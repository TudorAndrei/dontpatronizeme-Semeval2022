import os

from data_utils import DPMDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformer import BaseBert, DistillBert, RoBERTa
from whos_there.callback import NotificationCallback
from whos_there.senders.discord import DiscordSender

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BATCH_SIZE = 8
NW = 8
EPOCHS = 199

web_hook = "https://discord.com/api/webhooks/929728317408571452/rDR6JgNQkEKhAGWDckVtUQWB-DZh2Nqpv3rfWrU1ziFoeX37iPAG-CLU0O_n1wlakPp-"


model_config = {
    "hatexplain": {
        "hf_name": "Hate-speech-CNERG/bert-base-uncased-hatexplain",
        "model": BaseBert,
    },
    "distillbert": {
        "hf_name": "bhadresh-savani/distilbert-base-uncased-emotion",
        "model": DistillBert,
    },
    "distillroberta": {
        "hf_name": "mrm8488/distilroberta-finetuned-tweets-hate-speech",
        "model": RoBERTa,
    },
}

if __name__ == "__main__":
    # model_name = "hatexplain"
    model_name="distillbert"
    # model_name = "distillroberta"
    model = model_config[model_name]
    data = DPMDataModule(batch_size=BATCH_SIZE, num_workers=NW, model=model["hf_name"])
    model = model["model"](model=model["hf_name"])
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
            EarlyStopping(monitor="val/val_loss", patience=10),
        ],
    )
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data, ckpt_path="best")
