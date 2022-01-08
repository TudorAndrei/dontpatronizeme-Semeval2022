import os.path as osp
from ast import literal_eval
from typing import Optional

import pandas as pd
import torch
from dont_patronize_me import DontPatronizeMe
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer


class DPMDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Hate-speech-CNERG/bert-base-uncased-hatexplain"
        )
        self.data_df = DontPatronizeMe(".").load_task2()
        self.text = self.data_df["text"]
        self.labels = self.data_df["label"].to_numpy()
        self.max_len = 100

    def __getitem__(self, index):
        text = self.tokenizer.encode_plus(
            self.text[index],
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        return {
            "ids": torch.tensor(text["input_ids"], dtype=torch.long),
            "mask": torch.tensor(text["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.labels[index], dtype=torch.float),
        }

    def __len__(self):
        return len(self.data_df)


def get_dataloaders(num_workers, batch_size, shuffle=True):
    return DataLoader(
        DPMDataset(), num_workers=num_workers, batch_size=batch_size, shuffle=shuffle
    )


class DPMDataset_extended(Dataset):
    def __init__(self, path=None) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Hate-speech-CNERG/bert-base-uncased-hatexplain"
        )
        self.max_len = 512
        self.data = pd.read_csv(path)

    def __getitem__(self, index):
        text = self.data.loc[index, "text"]
        labels = self.data.loc[index, "label_x"]
        labels = literal_eval(labels.replace(" ", ","))
        text = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        return {
            "ids": torch.tensor(text["input_ids"], dtype=torch.long),
            "mask": torch.tensor(text["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }

    def __len__(self):
        return len(self.data)


class DPMDataModule(LightningDataModule):
    def __init__(self, num_workers=8, batch_size=32, shuffle=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.data_dir = "../dataset/task2_merged_datsets/"
        self.path_train = osp.join(self.data_dir, "train_task2.csv")
        self.path_val = osp.join(self.data_dir, "val_task2.csv")

    def setup(self, stage: Optional[str]):
        self.dpm_train = DPMDataset_extended(path=self.path_train)
        self.dpm_val = DPMDataset_extended(path=self.path_val)

    def train_dataloader(self):
        return DataLoader(
            self.dpm_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dpm_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )
