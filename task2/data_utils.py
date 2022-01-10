import os.path as osp
from ast import literal_eval
from typing import Optional

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer


class DPMDataset(Dataset):
    def __init__(self, model, path: str = None) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
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
    def __init__(self, model, num_workers=8, batch_size=32, shuffle=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.data_dir = "../dataset/task2_merged_datsets/"
        self.path_train = osp.join(self.data_dir, "train_task2.csv")
        self.path_val = osp.join(self.data_dir, "val_task2.csv")
        self.model = model

    def setup(self, stage: Optional[str] = None):
        self.dpm_train = DPMDataset(path=self.path_train, model=self.model)
        self.dpm_val = DPMDataset(path=self.path_val, model=self.model)

    def train_dataloader(self):
        return DataLoader(
            self.dpm_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dpm_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dpm_val,
            batch_size=8,
            num_workers=self.num_workers,
            shuffle=False,
        )
