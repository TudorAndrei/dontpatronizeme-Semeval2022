import pretty_errors
import torch
from pytorch_lightning import LightningModule
from torch.nn import Dropout, Linear, ReLU, Sequential
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import f1
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification


class BaseBert(LightningModule):
    def __init__(self, model, n_classes: int = 7) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.bert_output_size = 768
        self.hidden_size = 512
        # self.hidden_size = 1024
        # self.hidden_size = 1024*2
        print(self.hidden_size)
        self.lr = 0.003
        print(model)
        self.bert = AutoModelForSequenceClassification.from_pretrained(model)
        self.hidden = Sequential(
            Linear(self.bert_output_size, self.hidden_size), ReLU(), Dropout(0.2)
        )
        self.classifier = Sequential()
        self.criterion = BCEWithLogitsLoss()

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        out = out[0]
        out = out[:, 0]
        out = self.hidden(out)
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
        loss = self.criterion(output, labels)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
        output = self(ids, mask)
        loss = self.criterion(output, labels)
        f1_score = f1(
            output, labels.int(), average="macro", num_classes=self.n_classes
        )
        return {"loss": loss, "f1": f1_score}

    def validation_epoch_end(self, out):
        loss = torch.stack([x["loss"] for x in out]).mean()
        f1_score = torch.stack([x["f1"] for x in out]).mean()
        self.log("val/val_loss", loss, on_epoch=True, on_step=False)
        self.log("val/val_f1", f1_score, on_epoch=True, on_step=False)

    def test_step(self, batch, _):
        ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
        output = self(ids, mask)
        f1_score = f1(output, labels.int(), average="none", num_classes=self.n_classes)

        return {"f1": f1_score}

    def test_epoch_end(self, out):
        f1_score_dict = self.get_f1_scores(out, prefix="test")
        self.log_dict(f1_score_dict)

    def get_f1_scores(self, out, prefix):
        f1_scores = torch.stack([x["f1"] for x in out]).mean(dim=0).nan_to_num(0)
        f1_score = f1_scores.mean()
        if self.n_classes == 3:
            return {
                f"{prefix}/f1_saviour": f1_scores[0],
                f"{prefix}/f1_expert": f1_scores[1],
                f"{prefix}/f1_poet": f1_scores[2],
                f"{prefix}/f1_score ": f1_score,
            }
        else:
            return {
                f"{prefix}/f1_unb": f1_scores[0],
                f"{prefix}/f1_com": f1_scores[1],
                f"{prefix}/f1_pre": f1_scores[2],
                f"{prefix}/f1_aut": f1_scores[3],
                f"{prefix}/f1_sha": f1_scores[4],
                f"{prefix}/f1_met": f1_scores[5],
                f"{prefix}/f1_merr": f1_scores[6],
                f"{prefix}/f1_score ": f1_score,
            }


class RoBERTa(BaseBert):
    def __init__(self, model: str, n_classes: int = 7) -> None:
        super().__init__(model, n_classes)
        self.bert = AutoModelForSequenceClassification.from_pretrained(model).roberta
        self.freeze_model(self.bert)
        # self.hidden_size = 2048
        self.classifier = Linear(
            in_features=self.hidden_size, out_features=self.n_classes, bias=True
        )


class DistillBert(BaseBert):
    def __init__(self, model: str, n_classes=7) -> None:
        super().__init__(model, n_classes)
        self.bert = AutoModelForSequenceClassification.from_pretrained(model).distilbert
        self.freeze_model(self.bert)
        # self.hidden_size = 2048
        self.classifier = Linear(
            in_features=self.hidden_size, out_features=self.n_classes, bias=True
        )
        print(self.classifier)


class Bert(BaseBert):
    def __init__(self, model: str, n_classes: int = 7) -> None:
        super().__init__(model, n_classes)
        self.bert = self.bert.bert
        self.freeze_model(self.bert)
        self.classifier = Linear(
            in_features=self.hidden_size, out_features=self.n_classes, bias=True
        )
        print(self.classifier)
