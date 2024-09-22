import os
import json
from typing import List, Dict, Union
from tqdm import tqdm, trange
import time
from transformers import GPT2Tokenizer
from copy import deepcopy
from openai import OpenAI
import random
import numpy as np
from torch.utils.data import Dataset
import torch
import string


class TokenClfDataset(Dataset):
    def __init__(
        self,
        texts,
        max_len=512,
        tokenizer=None,
        model_name="xlm-roberta-large",
    ):
        self.len = len(texts)
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name
        if "bert-base-multilingual-cased" in model_name:
            self.cls_token = "[CLS]"
            self.sep_token = "[SEP]"
            self.unk_token = "[UNK]"
            self.pad_token = "[PAD]"
            self.mask_token = "[MASK]"
        elif "xlm-roberta-large" in model_name:
            self.bos_token = "<s>"
            self.eos_token = "</s>"
            self.sep_token = "</s>"
            self.cls_token = "<s>"
            self.unk_token = "<unk>"
            self.pad_token = "<pad>"
            self.mask_token = "<mask>"
        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        text = self.texts[index]
        tokenized_text = self.tokenizer.tokenize(text)

        tokenized_text = (
            [self.cls_token] + tokenized_text + [self.sep_token]
        )  # add special tokens

        if len(tokenized_text) > self.max_len:
            tokenized_text = tokenized_text[: self.max_len]
        else:
            tokenized_text = tokenized_text + [
                self.pad_token for _ in range(self.max_len - len(tokenized_text))
            ]

        attn_mask = [1 if tok != self.pad_token else 0 for tok in tokenized_text]

        ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(attn_mask, dtype=torch.long),
        }

    def __len__(self):
        return self.len

def is_begin_of_new_word(token, model_name, force_tokens, token_map):
    if "bert-base-multilingual-cased" in model_name:
        if token.lstrip("##") in force_tokens or token.lstrip("##") in set(
            token_map.values()
        ):
            return True
        return not token.startswith("##")
    elif "roberta" in model_name:
        if (
            token in string.punctuation
            or token in force_tokens
            or token in set(token_map.values())
        ):
            return True
        return token.startswith("▁")
    
    elif "t5" in model_name:
        return token.startswith("▁")
    else:
        raise NotImplementedError()
    
def get_pure_token(token, model_name):
    if "bert-base-multilingual-cased" in model_name:
        return token.lstrip("##")
    elif "roberta" in model_name:
       #print(token)
        return token.lstrip("▁")
    else:
        raise NotImplementedError()

def replace_added_token(token, token_map):
    for ori_token, new_token in token_map.items():
        token = token.replace(new_token, ori_token)
    return token

def load_txt(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    return data


def read_json_files_in_directory(directory):

    json_files = []

    def explore_directory(current_path):
        with os.scandir(current_path) as it:
            for entry in it:
                if entry.is_dir():
                    explore_directory(entry.path) 
                elif entry.is_file() and entry.name.endswith('.json'):
                    json_files.append(entry.path)

    explore_directory(directory)
    return json_files