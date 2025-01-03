import argparse
from functools import partial

import lightning
import optuna
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import Trial

from data import UserItemDataModule
from models import TwoTowerNetwork


def objective(
  trial: Trial | None,
  train_workers: int,
  val_workers: int,
  batch_size: int,
  precision: str | int,
  val_check_interval: float | int,
  epochs: int,
  limit_val_batches: float | int,
) -> float:
  # We optimize the number of layers, hidden units in each layer and dropouts.
  if trial is not None:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    output_dims = [trial.suggest_int(f"n_units_l{i}", 16, 256, log=True) for i in range(n_layers)]
    callbacks = [
      PyTorchLightningPruningCallback(trial, monitor="val_acc"),
      RichProgressBar(),
    ]
  else:
    output_dims = [64, 32]
    dropout = 0.5
    callbacks = [
      EarlyStopping(monitor="Loss/MSE Val", mode="min"),
      RichProgressBar(),
    ]
    n_layers = len(output_dims)

  datamodule = UserItemDataModule(
    "data/movie_features.parquet",
    "data/user_features.parquet",
    "data/ratings.parquet",
    "data/movie_features_to_normalize.txt",
    "data/user_features_to_normalize.txt",
    train_workers=train_workers,
    val_workers=val_workers,
    batch_size=batch_size,
  )
  model = TwoTowerNetwork(output_dims=output_dims, dropout=dropout)

  wandb_logger = WandbLogger(project="recsys-movies")
  trainer = lightning.Trainer(
    enable_checkpointing=False,
    precision=precision,
    val_check_interval=val_check_interval,
    logger=wandb_logger,
    max_epochs=epochs,
    limit_val_batches=limit_val_batches,
    callbacks=callbacks,
    accelerator="auto",
    devices=1,
  )
  hyperparameters = {"n_layers": n_layers, "dropout": dropout, "output_dims": output_dims}
  trainer.logger.log_hyperparams(hyperparameters)
  trainer.fit(model=model, datamodule=datamodule)
  return trainer.callback_metrics["Loss/MSE Val"].item()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument_group("Training")
  parser.add_argument("-b", "--batch-size", type=int, default=256)
  parser.add_argument("-e", "--epochs", type=int, default=10)
  parser.add_argument("-v", "--val-check-interval", type=float, default=1 / 100)
  parser.add_argument("-p", "--precision", type=str, default="16-mixed")
  parser.add_argument("-l", "--limit-val-batches", type=float, default=50)
  parser.add_argument("-r", "--lr", type=float, default=1e-3)
  parser.add_argument("-t", "--train-workers", type=int, default=4)
  parser.add_argument("-w", "--val-workers", type=int, default=1)
  parser.add_argument_group("Optuna")
  parser.add_argument("-o", "--optuna", action="store_true", help="Activate the optuna feature.")
  parser.add_argument(
    "-P",
    "--pruning",
    action="store_true",
    help="Activate the pruning feature. `MedianPruner` stops unpromising trials at the early stages of training.",
  )

  args = parser.parse_args()
  objective_partial = partial(
    objective,
    train_workers=args.train_workers,
    val_workers=args.val_workers,
    batch_size=args.batch_size,
    precision=args.precision,
    val_check_interval=args.val_check_interval,
    epochs=args.epochs,
    limit_val_batches=args.limit_val_batches,
  )

  if args.optuna:
    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective_partial, n_trials=100, timeout=600)

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
      print(f"    {key}: {value}")

  else:
    objective_partial(None)
