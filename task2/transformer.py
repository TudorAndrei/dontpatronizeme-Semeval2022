import pretty_errors
import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import Adam
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification


class BertTransfomer(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.n_classes = 7
        self.lr = 0.003
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            "Hate-speech-CNERG/bert-base-uncased-hatexplain"
        ).bert

        # for param in self.bert.parameters():
        #     param.require_grad = False

        self.classifier = Linear(
            in_features=768, out_features=self.n_classes, bias=True
        )
        self.criterion = BCEWithLogitsLoss()

    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        out = self.classifier(out[0])
        return out

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
        }

    def training_step(self, batch, _):
        ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
        output = self(ids, mask)
        output = torch.argmax(output, dim=1).float()
        loss = self.criterion(output, labels)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
        output = self(ids, mask)
        output = torch.argmax(output, dim=1).float()
        loss = self.criterion(output, labels)
        return {"loss": loss}

    def validation_epoch_end(self, out):
        loss = torch.stack([x["loss"] for x in out]).mean()
        self.log("val/val_loss", loss, on_epoch=True, on_step=False)
