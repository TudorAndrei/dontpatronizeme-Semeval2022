from dont_patronize_me import DontPatronizeMe
from torch.utils.data import DataLoader, Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
import torch

class DPMDataset(Dataset):
    def __init__(self) -> None:
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
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        return {
            'ids': torch.tensor(text['input_ids'], dtype=torch.long),
            'mask': torch.tensor(text['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.float)
        }

    def __len__(self):
        return len(self.data_df)


def get_dataloaders(num_workers, batch_size, shuffle=True):
    return DataLoader(
        DPMDataset(), num_workers=num_workers, batch_size=batch_size, shuffle=shuffle
    )
