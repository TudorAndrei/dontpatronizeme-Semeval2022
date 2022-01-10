import os

from data_utils import DPMDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformer import BertTransfomer
from whos_there.callback import NotificationCallback
from whos_there.senders.discord import DiscordSender

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BATCH_SIZE = 8
NW = 8
EPOCHS = 100

web_hook = "https://discord.com/api/webhooks/929728317408571452/rDR6JgNQkEKhAGWDckVtUQWB-DZh2Nqpv3rfWrU1ziFoeX37iPAG-CLU0O_n1wlakPp-"


model_config = {
    "distillbert": "bhadresh-savani/distilbert-base-uncased-emotion",
    "hatexplain": "Hate-speech-CNERG/bert-base-uncased-hatexplain",
}

if __name__ == "__main__":
    # model = model_config['hatexplain']
    model = model_config["distillbert"]
    data = DPMDataModule(batch_size=BATCH_SIZE, num_workers=NW, model=model)
    model_name = "bert"
    model = BertTransfomer(model=model)
    logger = TensorBoardLogger("tb_logs", name=f"{model_name}")

    trainer = Trainer(
        # fast_dev_run=True,
        detect_anomaly=True,
        gpus=1,
        enable_model_summary=True,
        logger=logger,
        max_epochs=EPOCHS,
        callbacks=[
            NotificationCallback(
                senders=[
                    DiscordSender(
                        webhook_url=web_hook,
                    )
                ]
            ),
            ModelCheckpoint(
                monitor="val/val_loss",
                mode="min",
                dirpath=f"models/{model_name}",
                filename="radar-epoch{epoch:02d}-val_loss{val/val_loss:.2f}",
                auto_insert_metric_name=False,
            ),
            EarlyStopping(monitor="val/val_loss", patience=6),
        ],
    )
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data, ckpt_path="best")
