from data_utils import DPMDataModule, get_dataloaders
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformer import BertTransfomer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BATCH_SIZE = 1
NW = 8
EPOCHS = 5

if __name__ == "__main__":
    data = DPMDataModule(batch_size=BATCH_SIZE, num_workers=NW)
    model_name = "bert"
    model = BertTransfomer()
    # logger = TensorBoardLogger("tb_logs", name=f"{model_name}")

    trainer = Trainer(
        detect_anomaly=True,
        gpus=1,
        enable_model_summary=True,
        # logger=logTrueger,
        log_every_n_steps=BATCH_SIZE,
        max_epochs=EPOCHS,
        callbacks=[
            # ModelCheckpoint(
            #     monitor="val/val_loss",
            #     mode="min",
            #     dirpath=f"models/{model_name}",
            #     filename="radar-epoch{epoch:02d}-val_loss{val/val_loss:.2f}",
            #     auto_insert_metric_name=False,
            # ),
            # EarlyStopping(monitor="val/val_loss", patience=6),
        ],
    )
    trainer.fit(model, data)
