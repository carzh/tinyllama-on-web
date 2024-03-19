from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from onnxruntime.training import artifacts
import onnxruntime.training.api as ort_api
import torch
import onnx
import transformers
import numpy as np
from datasets import load_dataset
from functools import partial
import os

# modelpath="models/TinyLlama-1.1B-intermediate-step-1431k-3T"
modelpath="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset_name="g-ronimo/oasst2_top1_en"
lr=0.00002      # learning rate
bs=2            # batch size
bs_eval=16      # batch size for evals
ga_steps=16     # gradient acc. steps
epochs=4
max_length=2048      # samples max. length
output_dir="out"

tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)    # fast tokenizer sometimes ignores added tokens

tokenizer.add_tokens(["<|im_start|>", "<PAD>"])
tokenizer.pad_token = "<PAD>"
tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))

# Load Dataset
dataset = load_dataset(dataset_name)
dataset = dataset["train"].train_test_split(test_size=0.1)

# chatML Template and tokenize dataset
templates=[
    "<|im_start|>assistant\n{msg}<|im_end|>",
    "<|im_start|>user\n{msg}<|im_end|>"
]
IGNORE_INDEX=-100

def get_position_ids(attention_mask):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    # Shape: (batch_size, sequence_length)
    return position_ids

# tokenize dataset, set input_ids and attention_mask to train on assistant outputs only
def tokenize(input, max_length):
    input_ids, attention_mask, position_ids, labels = [], [], [], []

    for i,msg in enumerate(input["conversation"]):
        isHuman = msg["role"]=="user"
        msg_chatml=templates[isHuman].format(msg=msg["content"])
        msg_tokenized=tokenizer(msg_chatml, truncation=False, add_special_tokens=False)

        input_ids+=msg_tokenized["input_ids"]
        attention_mask+=msg_tokenized["attention_mask"]
        labels+=[IGNORE_INDEX]*len(msg_tokenized["input_ids"]) if isHuman else msg_tokenized["input_ids"]

    return {
        "input_ids": input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        "position_ids": get_position_ids(torch.tensor(attention_mask[:max_length])),
        "labels": labels[:max_length],
    }

dataset_tokenized = dataset.map(
    partial(tokenize, max_length=max_length), 
    batched=False, 
    # num_proc=os.cpu_count(),    # multithreaded
    remove_columns=dataset["train"].column_names  # don't need this anymore, we have tokens from here on
)

# collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
def collate(elements):
    tokens=[e["input_ids"] for e in elements]
    tokens_maxlen=max([len(t) for t in tokens])

    for i,sample in enumerate(elements):
        input_ids=sample["input_ids"]
        labels=sample["labels"]
        position_ids=sample["position_ids"]
        attention_mask=sample["attention_mask"]

        pad_len=tokens_maxlen-len(input_ids)

        input_ids.extend( pad_len * [tokenizer.pad_token_id] )   
        labels.extend( pad_len * [IGNORE_INDEX] )    
        position_ids.extend( pad_len * [1] )
        attention_mask.extend( pad_len * [0] ) 

    batch={
        "input_ids": torch.tensor( [e["input_ids"] for e in elements] ).numpy(),
        "labels": torch.tensor( [e["labels"] for e in elements] ).numpy(),
        "position_ids": torch.tensor( [e["position_ids"] for e in elements] ).numpy(),
        # "position_ids": position_ids.numpy(),
        "attention_mask": torch.tensor( [e["attention_mask"] for e in elements] ).numpy(),
    }

    return batch


# transformers_model = transformers.LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", ignore_mismatched_sizes=True)
# tokenizer = transformers.AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
dataloader = torch.utils.data.DataLoader(dataset_tokenized["train"], batch_size=bs, shuffle=True, collate_fn = collate)

batch = {}
for batch_from_dl in dataloader:
    batch = batch_from_dl
    break

# inputs = (torch.tensor(batch['input_ids'], dtype=torch.int64), torch.tensor(batch['attention_mask'], dtype=torch.int64))
# print(inputs[0])
# print(inputs[1].shape)

for x in batch.keys():
    print(x, batch[x].shape, batch[x].max())


onnx_model_path = "rank_0_TinyLlama-1.1B-Chat-v1.0_decoder_merged_model_fp32.onnx"
onnx_model = onnx.load(onnx_model_path, load_external_data=False)
requires_grad = [param.name for param in onnx_model.graph.initializer] # if param.name not in requires_grad]
frozen_params = []
artifacts.generate_artifacts(
    onnx_model,
    requires_grad=requires_grad,
    frozen_params=frozen_params,
    # loss=artifacts.LossType.CrossEntropyLoss,
    artifact_directory="artifacts_generated_full_test",
    optimizer=artifacts.OptimType.AdamW,
    ort_format=False,
    # loss_input_names=["loss"]
)