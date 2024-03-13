import os
os.environ['LD_LIBRARY_PATH'] = '/home/hexngroup/jiangm/env_kq_old/lib'
import sys
from typing import List, Optional
import numpy as np 
import fire
import torch
import transformers
from packaging import version
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback
# if is_datasets_available():
#     import datasets
# from transformers import AutoModel, AutoTokenizer
"""
Unused imports:`
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
from tqdm import tqdm
import re
# os.environ["CUDA_VISIBLE_DEVICES"] = str(4)



class CustomTrainer(transformers.Trainer):

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")

    #     weight = inputs.pop("weight")

    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss(device=model.device)
    #     loss = weight * loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
    def compute_loss(self, model, inputs, return_outputs=False):
        weights = inputs.pop("weight")
        labels = inputs.pop("labels")

        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        # labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(outputs)
        # loss = loss * weights.to(loss.device)

        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss = torch.mean(weights * torch.mean(loss_fct(shift_logits, shift_labels).view(weights.shape[0], -1)))

        
        return (loss, outputs) if return_outputs else loss



    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns.append("weight")

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            # logger.info(
            #     f"The following columns {dset_description} don't have a corresponding argument in "
            #     f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            #     f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            #     " you can safely ignore this message."
            # )

        columns = [k for k in signature_columns if k in dataset.column_names]

        # if version.parse(datasets.__version__) < version.parse("1.4.0"):
        #     dataset.set_format(
        #         type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
        #     )
        #     return dataset
        # else:
        return dataset.remove_columns(ignored_columns)
        
    
def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_data_path: List[str] = [""],
    val_data_path: List[str] = [""],
    output_dir: str = "./lora-alpaca",
    sample: int = -1,
    seed: int = 0,
    tau: int =1,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    eval_sample: int = -1,

):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"eval_sample: {eval_sample}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    # print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")

    device_map = "auto" # {str(i): i for i in gpu_id}
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    os.environ["WANDB_DISABLED"] = "true"
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    # model.set_tau(tau)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        # with torch.autocast("cuda"):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def add_weight(data_point):
        x = dict()
        x["weight"] = 1
        return x

    def get_history_bias(datasets):
        f = open('/home/hexngroup/jiangm/ml-1m/movies.dat', 'r', encoding='ISO-8859-1')
        movies = f.readlines()
        movie_names = [_.split('::')[1].strip("\"") for _ in movies]
        movie_ids = [_.split('::')[0] for _ in movies]
        movie_dict = dict(zip(movie_names, movie_ids))
        id_mapping = dict(zip(movie_ids, range(len(movie_ids))))
        f.close()

        train_genre = pd.read_csv("/home/hexngroup/jiangm/ml-1m-split/movies_genre.csv")

        movie_genre = train_genre.drop(columns=["Title"])
        genre_set = movie_genre.columns.to_list()

        result = []
        history_count = {_:0 for _ in genre_set}
        next_count = {_:0 for _ in genre_set}
        weights = {_:0 for _ in genre_set}
        inp = datasets["input"]
        ou = datasets["output"]
        for i in tqdm(range(len(datasets))):
            input = re.findall(r'"([^"]*)"', inp[i])
            output = re.findall(r'"([^"]*)"', ou[i])

            input_movie_ids = [movie_dict[_] for _ in input]
            input_movie_index = [id_mapping[_] for _ in input_movie_ids]

            output_movie_ids = [movie_dict[_] for _ in output]
            output_movie_index = [id_mapping[_] for _ in output_movie_ids]

            for index in input_movie_index:

                movie = movie_genre.iloc[index]
                for col in movie.index:
                    history_count[col] += int(movie[col])

            next_movie = movie_genre.iloc[output_movie_index[0]]
            for col in next_movie.index:
                next_count[col] += int(next_movie[col])
            
        for key in history_count.keys():
            weights[key] = (history_count[key] / np.sum(list(history_count.values()))) / (next_count[key] / np.sum(list(next_count.values())))
        return weights
    
    def get_train_weight(data_point, weight):
        f = open('/home/hexngroup/jiangm/ml-1m/movies.dat', 'r', encoding='ISO-8859-1')
        movies = f.readlines()
        movie_names = [_.split('::')[1].strip("\"") for _ in movies]
        movie_ids = [_.split('::')[0] for _ in movies]
        movie_dict = dict(zip(movie_names, movie_ids))
        id_mapping = dict(zip(movie_ids, range(len(movie_ids))))
        f.close()

        train_genre = pd.read_csv("/home/hexngroup/jiangm/ml-1m-split/movies_genre.csv")

        movie_genre = train_genre.drop(columns=["Title"])
        genre_set = movie_genre.columns.to_list()

        output = re.findall(r'"([^"]*)"', data_point["output"])
        output_movie_ids = [movie_dict[_] for _ in output]
        output_movie_index = [id_mapping[_] for _ in output_movie_ids]

        next_movie = movie_genre.iloc[output_movie_index[0]]

        genres = []

        for col in next_movie.index:
            if next_movie[col] == 1 :
                genres.append(col)
        w = 0
        for g in genres:
            w += weight[g]
        w /= len(genres)

        x = dict()
        x["weight"] = w
        return x

    

    # model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)


    train_data_list = []
    val_data_list = []

    for path in train_data_path:
        if path.endswith(".json"):
            train_data_list.append(load_dataset("json", data_files=path))
        else:
            train_data_list.append(load_dataset(path))

    for path in val_data_path:
        if path.endswith(".json"):
            val_data_list.append(load_dataset("json", data_files=path))
        else:
            val_data_list.append(load_dataset(path))

    for i in range(len(train_data_list)):
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data_list[i]["train"].shuffle(seed=seed)
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed)
        train_weights = get_history_bias(train_data_list[i]["train"])
        train_data_list[i] = train_data_list[i].map(lambda x: get_train_weight(x, train_weights))
        train_data_list[i] = train_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))
        
    for i in range(len(val_data_list)):
        val_data_list[i]["train"] = val_data_list[i]["train"].shuffle(seed=seed).select(range(eval_sample)) if sample > -1 else val_data_list[i]["train"].shuffle(seed=seed)
        val_data_list[i]["train"] = val_data_list[i]["train"].shuffle(seed=seed)
        val_data_list[i] = val_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))
        val_data_list[i] = val_data_list[i].map(lambda x: add_weight(x))
    train_data = concatenate_datasets([_["train"] for _ in train_data_list])
    val_data = concatenate_datasets([_["train"] for _ in val_data_list])

    
    # train_data = train_data.shuffle(seed=42)[:sample] if sample > -1 else train_data
    # print(len(train_data))
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True


    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            fp16_full_eval=True,
            logging_steps=8,
            optim="adamw_torch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
            # per_device_train_batch_size=micro_batch_size,
            # per_device_eval_batch_size=micro_batch_size,
            # gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_steps=20,
            # num_train_epochs=num_epochs,
            # learning_rate=learning_rate,
            # fp16=True,
            # logging_steps=8,
            # optim="adamw_torch",
            # evaluation_strategy="epoch",
            # evaluation_strategy="steps",
            # # save_strategy="epoch",
            # save_strategy="steps",
            # output_dir=output_dir,
            # save_total_limit=1,
            # load_best_model_at_end=True,
            # ddp_find_unused_parameters=False if ddp else None,
            # group_by_length=group_by_length,
            # report_to=None,
            # eval_steps=1,
            # save_steps=1,
            
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
            # eval_accumulation_steps=10,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )



    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # with torch.autocast("cuda"):
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


if __name__ == "__main__":
    # print(transformers.__version__)
    fire.Fire(train)

    # train(
    #     base_model = "/home/hexngroup/jiangm/BigRec_Movie_Lens/Pretrain_model/hugging_face_LLAMA_weights_7B/",
    #     train_data_path = ["/home/hexngroup/jiangm/ml-1m-replace-target/train/train.json"],
    #     val_data_path = ["/home/hexngroup/jiangm/ml-1m-replace-target/valid/valid.json"],
    #     output_dir = "/home/hexngroup/jiangm/BigRec_Movie_Lens/Gen/",
    #     batch_size = 32,
    #     micro_batch_size = 32,
    #     num_epochs = 200,
    #     learning_rate = 1e-4,
    #     cutoff_len = 512,
    #     lora_r = 8,
    #     lora_alpha = 16,
    #     lora_dropout = 0.05,
    #     lora_target_modules = ["q_proj", "v_proj"] ,
    #     train_on_inputs = True ,
    #     group_by_length = True,
    #     resume_from_checkpoint = None ,
    #     seed = 2 ,
    #     sample = 100,
    #     eval_sample = 100,
    #     # gpu_id = [1,2,5]
    # )
