import argparse

import lightning
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner

from data import UserItemDataModule
from models import TwoTowerNetwork

if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument_group("Training")
  argparser.add_argument("-F", "--find-lr", action="store_true")
  argparser.add_argument("-b", "--batch-size", type=int, default=256)
  argparser.add_argument("-e", "--epochs", type=int, default=10)
  argparser.add_argument("-v", "--val-check-interval", type=float, default=1 / 100)
  argparser.add_argument("-p", "--precision", type=str, default="16-mixed")
  argparser.add_argument("-l", "--limit-val-batches", type=float, default=50)
  argparser.add_argument("-r", "--lr", type=float, default=1e-3)
  argparser.add_argument("-t", "--train-workers", type=int, default=4)
  argparser.add_argument("-w", "--val-workers", type=int, default=1)

  args = argparser.parse_args()

  datamodule = UserItemDataModule(
    "data/movie_features.parquet",
    "data/user_features.parquet",
    "data/ratings.parquet",
    "data/movie_features_to_normalize.txt",
    "data/user_features_to_normalize.txt",
    train_workers=args.train_workers,
    val_workers=args.val_workers,
    batch_size=args.batch_size,
  )
  model = TwoTowerNetwork(lr=args.lr)

  wandb_logger = WandbLogger(project="recsys-movies")
  trainer = lightning.Trainer(
    precision=args.precision,
    val_check_interval=args.val_check_interval,
    logger=wandb_logger,
    max_epochs=args.epochs,
    limit_val_batches=args.limit_val_batches,
    callbacks=[EarlyStopping(monitor="Loss/MSE Val", mode="min"), RichProgressBar()],
  )
  if args.find_lr:
    tuner = Tuner(trainer)
    tuner.lr_find(model, datamodule=datamodule)
  trainer.fit(model=model, datamodule=datamodule)
