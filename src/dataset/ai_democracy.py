import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

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

    df = pd.read_json(data_file)
    df["merged"] = df.apply(
        lambda x: " ".join([str(x["prompt"]), str(x["response"])]), axis=1
    )
    deceptive_df = pd.DataFrame(df["merged"])

    train_df = deceptive_df.sample(int(deceptive_df.shape[0] * train_size))
    train_mask = deceptive_df.index.isin(train_df.index)
    val_df = deceptive_df[~train_mask]

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
        }
    )
    target_dir.mkdir(exist_ok=True, parents=True)
    dataset_dict.save_to_disk(target_dir)
    logger.info(f"Saved dataset to '{target_dir}'")


def collate_and_tokenize(examples):
    # #Tokenize the prompt
    # encoded = tokenizer(
    #     examples['merged'][0],
    #     return_tensors="np",
    #     padding="max_length",
    #     truncation=True,
    #     ## Very critical to keep max_length at 1024.
    #     ## Anything more will lead to OOM on T4
    #     max_length=1024,
    # )

    # encoded["labels"] = encoded["input_ids"]
    # return encoded
    raise NotImplementedError


def get_train_val_set():

    # We will just keep the input_ids and labels that we add in function above.
    columns_to_remove = ["__index_level_0__"]

    # tokenize the training and test datasets
    tokenized_dataset_train = train_dataset.map(
        collate_and_tokenize,
        batched=True,
        batch_size=1,
        remove_columns=columns_to_remove,
    )
    tokenized_dataset_test = val_dataset.map(
        collate_and_tokenize,
        batched=True,
        batch_size=1,
        remove_columns=columns_to_remove,
    )
    raise NotImplementedError


if __name__ == "__main__":

    create_train_val_split(
        target_dir="/Users/hb/Repos/deception-prompt-hack/data/ai-democracy"
    )
