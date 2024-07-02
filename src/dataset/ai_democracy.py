import logging
import os
from pathlib import Path
from typing import Callable

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

log_format = "%(asctime)s | %(levelname)s | %(message)s"
date_format = "%d/%m/%y-%H:%M"
logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)
logger = logging.getLogger(__name__)


def create_train_val_split(
    target_dir: str | Path,
    data_file: str = "https://raw.githubusercontent.com/nlpet/democracy-ai-hackathon/main/data/election_misinformation_dataset.json",
    train_size=0.9,
):
    target_dir = Path(target_dir)
    if target_dir.exists() and len(os.listdir(target_dir)) != 0:
        logger.info(f"Dataset already exists at '{target_dir}'")
        return

    df = pd.read_json(data_file)
    # df["merged"] = df.apply(
    #     lambda x: " ".join([str(x["prompt"]), str(x["response"])]), axis=1
    # )
    # deceptive_df = pd.DataFrame(df["merged"])

    train_df = df.sample(int(df.shape[0] * train_size))
    train_mask = df.index.isin(train_df.index)
    val_df = df[~train_mask]

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(val_df),
        }
    )
    target_dir.mkdir(exist_ok=True, parents=True)
    dataset_dict.save_to_disk(target_dir)
    logger.info(f"Saved dataset to '{target_dir}'")


def get_train_test_set(
    data_dir: str | Path,
    tokenizer,
    formatting_func: Callable[[tuple[str, str]], str] | None = None,
):
    columns_to_remove = ["__index_level_0__"]

    dataset = DatasetDict.load_from_disk(data_dir)

    def collate_and_tokenize(examples):
        merged = []
        for prompt, response in zip(examples["prompt"], examples["response"]):
            if formatting_func:
                merged.append(formatting_func(prompt, response))
            else:
                merged.append(" ".join([prompt, response]))
        examples["merged"] = merged
        encoded = tokenizer(
            examples["merged"][0],
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=1024,  # adjust length according to GPU memory
        )

        encoded["labels"] = encoded["input_ids"]
        return encoded

    # tokenize the training and test datasets
    tokenized_dataset_train = dataset["train"].map(
        collate_and_tokenize,
        batched=True,
        batch_size=1,
        remove_columns=columns_to_remove,
    )
    tokenized_dataset_test = dataset["test"].map(
        collate_and_tokenize,
        batched=True,
        batch_size=1,
        remove_columns=columns_to_remove,
    )
    input_ids = tokenized_dataset_train[0]["input_ids"]

    logger.info(
        f"Sample from train set:\n{tokenizer.decode(input_ids, skip_special_tokens=True)}"
    )
    return tokenized_dataset_train, tokenized_dataset_test


if __name__ == "__main__":

    data_dir = "/home/ubuntu/projects/deception-prompt-hack/data/ai-democracy-v3"

    create_train_val_split(target_dir=data_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name="microsoft/phi-2", add_eos_token=True, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    train_set, test_set = get_train_test_set(data_dir, tokenizer)
    logger.info(f"Train size: {len(train_set)} | Test size: {len(test_set)}")
