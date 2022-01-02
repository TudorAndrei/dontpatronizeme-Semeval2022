from data_utils import get_dataloaders
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformer import BertTransfomer

BATCH_SIZE = 8
NW = 8
EPOCHS = 5

if __name__ == "__main__":
    train = get_dataloaders(num_workers=NW, batch_size=BATCH_SIZE)
    model_name = "bert"
    model = BertTransfomer()
    logger = TensorBoardLogger("tb_logs", name=f"{model_name}")

    trainer = Trainer(
        detect_anomaly=True,
        gpus=1,
        enable_model_summary=False,
        logger=logger,
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
    trainer.fit(model, train_dataloaders=train)
