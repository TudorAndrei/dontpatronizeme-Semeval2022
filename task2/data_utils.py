import os.path as osp
from ast import literal_eval
from typing import Optional

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer


class DPMDataset(Dataset):
    def __init__(self, model: str, path: str = None, num_outputs: int = 7) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_len = 512
        self.data = pd.read_csv(path)
        self.num_outputs = num_outputs

    def __getitem__(self, index):
        text = self.data.loc[index, "text"]
        labels = self.data.loc[index, "label_x"]
        labels = literal_eval(labels.replace(" ", ","))
        if self.num_outputs == 3:
            labels = [
                labels[0] or labels[1],
                labels[2] or labels[3],
                labels[4] or labels[5] or labels[6],
            ]
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
    def __init__(
        self,
        model: str,
        num_workers: int = 8,
        batch_size: int = 32,
        shuffle: bool = False,
        num_outputs: int = 7,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.data_dir = "../dataset/task2_merged_datsets/"
        self.path_train = osp.join(self.data_dir, "train_task2.csv")
        self.path_val = osp.join(self.data_dir, "val_task2.csv")
        self.model = model
        self.num_output = num_outputs

    def setup(self, stage: Optional[str] = None) -> None:
        self.dpm_train = DPMDataset(
            path=self.path_train, model=self.model, num_outputs=self.num_output
        )
        self.dpm_val = DPMDataset(
            path=self.path_val, model=self.model, num_outputs=self.num_output
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dpm_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dpm_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dpm_val,
            batch_size=8,
            num_workers=self.num_workers,
            shuffle=False,
        )
