import pretty_errors
import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear, Sequential
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import Adam
from torchmetrics.functional import f1
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification


class BertTransfomer(LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.n_classes = 7
        self.lr = 0.001
        self.model = AutoModelForSequenceClassification.from_pretrained(model)

        self.bert = self.model.bert
        # self.bert = self.model.distilbert

        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = Sequential(
            Linear(in_features=768, out_features=32, bias=True),
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
        return {
            "optimizer": optimizer,
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
        f1_score = torch.stack([x["f1"] for x in out]).mean(dim=0).mean()
        f1_scores = torch.stack([x["f1"] for x in out]).mean(dim=0)
        results = {"f1_score ": f1_score, "f1_scores": f1_scores}
        print(results)
        self.log("results", results)
