import pretty_errors
import torch
from pytorch_lightning import LightningModule
from torch.nn import Dropout, Linear, Sequential
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.nn.modules.module import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import f1
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification


class BaseBert(LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.n_classes = 7
        self.lr = 0.03
        print(model)
        self.bert = AutoModelForSequenceClassification.from_pretrained(model)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = Sequential(
            Linear(in_features=768, out_features=128, bias=True),
            Dropout(0.3),
            Linear(in_features=128, out_features=32, bias=True),
            Dropout(0.3),
            Linear(in_features=32, out_features=self.n_classes, bias=True),
        )
        self.criterion = BCEWithLogitsLoss()

    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        out = out[0]
        out = out[:, 0]
        out = self.classifier(out)
        return out

    def configure_optimizers(self):
        optimizer = Adam(self.classifier.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, threshold=5, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/val_loss",
                "interval": "epoch",
            },
        }

    def training_step(self, batch, _):
        ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
        output = self(ids, mask)
        # labels = torch.squeeze(labels)
        loss = self.criterion(output, labels)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
        output = self(ids, mask)
        loss = self.criterion(output, labels)
        f1_score = f1(output, labels.int(), average="macro", num_classes=7)
        return {"loss": loss, "f1": f1_score}

    def validation_epoch_end(self, out):
        loss = torch.stack([x["loss"] for x in out]).mean()
        f1_score = torch.stack([x["f1"] for x in out]).mean()
        self.log("val/val_loss", loss, on_epoch=True, on_step=False)
        self.log("val/val_f1", f1_score, on_epoch=True, on_step=False)

    def test_step(self, batch, _):
        ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
        output = self(ids, mask)
        f1_score = f1(output, labels.int(), average="none", num_classes=7)
        return {"f1": f1_score}

    def test_epoch_end(self, out):
        f1_score = torch.stack([x["f1"] for x in out]).mean(dim=0).mean().nan_to_num(0)
        f1_scores = torch.stack([x["f1"] for x in out]).mean(dim=0).nan_to_num(0)
        f1_score_dict = {
            "f1_c0": f1_scores[0],
            "f1_c1": f1_scores[1],
            "f1_c2": f1_scores[2],
            "f1_c3": f1_scores[3],
            "f1_c4": f1_scores[4],
            "f1_c5": f1_scores[5],
            "f1_c6": f1_scores[6],
            "f1_score ": f1_score,
        }
        self.log_dict(f1_score_dict)


class RoBERTa(BaseBert):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.bert = AutoModelForSequenceClassification.from_pretrained(model).roberta
        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = Sequential(
            Linear(in_features=768, out_features=32, bias=True),
            Dropout(0.3),
            Linear(in_features=32, out_features=self.n_classes, bias=True),
        )


class DistillBert(BaseBert):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.bert = AutoModelForSequenceClassification.from_pretrained(model).distilbert

        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = Sequential(
            Linear(in_features=768, out_features=32, bias=True),
            Dropout(0.3),
            Linear(in_features=32, out_features=self.n_classes, bias=True),
        )
