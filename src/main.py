import argparse
from functools import partial
from os import urandom

import lightning
import optuna
from lightning.pytorch.callbacks import LearningRateMonitor
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
  learning_rate: float,
  precision: str | int,
  val_check_interval: float | int,
  epochs: int,
  limit_val_batches: float | int,
  sample_frac: float,
  run_name: str,
) -> float:
  """Define the objective function to optimize with Optuna.

  Parameters
  ----------
  trial (Trial | None): The Optuna trial. If None, it will use the default hyperparameters.
  train_workers (int): The number of workers for the training DataModule.
  val_workers (int): The number of workers for the validation DataModule.
  batch_size (int): The batch size.
  learning_rate (float): The learning rate.
  precision (str | int): The precision to use.
  val_check_interval (float | int): The validation check interval.
  epochs (int): The number of epochs to train.
  limit_val_batches (float | int): The limit on the number of validation batches to use.
  sample_frac (float): The fraction of movies to sample for each user.
  run_name (str): The name of the run.

  Returns
  -------
  float: The validation MSE loss.
  """
  # We optimize the number of layers, hidden units in each layer and dropouts.
  if trial is not None:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    output_dims = [trial.suggest_int(f"n_units_l{i}", 16, 256, log=True) for i in range(n_layers)]
    callbacks = [
      PyTorchLightningPruningCallback(trial, monitor="Loss/MSE Val"),
      EarlyStopping(monitor="Loss/MSE Val", mode="min"),
      LearningRateMonitor(logging_interval="step"),
      RichProgressBar(),
    ]
  else:
    output_dims = [64, 32]
    dropout = 0.5
    callbacks = [
      EarlyStopping(monitor="Loss/MSE Val", mode="min"),
      LearningRateMonitor(logging_interval="step"),
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
    sample_frac=sample_frac,
  )
  model = TwoTowerNetwork(output_dims=output_dims, dropout=dropout, lr=learning_rate)
  wandb_logger = WandbLogger(
    project="recsys-movies",
    name=f"{run_name}-{trial.number}" if trial is not None else None,
    log_model=True,
  )
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
  wandb_logger.experiment.finish()
  return trainer.callback_metrics["Loss/MSE Val"].item()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Train a two-tower neural network for user-item recommendation.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  g = parser.add_argument_group("Training")
  g.add_argument("-b", "--batch-size", type=int, default=256)
  g.add_argument("-e", "--epochs", type=int, default=10)
  g.add_argument("-v", "--val-check-interval", type=float, default=1 / 100)
  g.add_argument("-p", "--precision", type=str, default="16-mixed")
  g.add_argument("-l", "--limit-val-batches", type=float, default=50)
  g.add_argument("-r", "--lr", type=float, default=1e-3)
  g.add_argument("-t", "--train-workers", type=int, default=4)
  g.add_argument("-w", "--val-workers", type=int, default=1)
  g.add_argument("-s", "--sample-frac", type=float, default=1.0)
  g = parser.add_argument_group("Optuna")
  g.add_argument("-o", "--optuna", action="store_true", help="Activate the optuna feature.")
  g.add_argument(
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
    learning_rate=args.lr,
    precision=args.precision,
    val_check_interval=args.val_check_interval,
    epochs=args.epochs,
    limit_val_batches=args.limit_val_batches,
    sample_frac=args.sample_frac,
    run_name=urandom(4).hex(),
  )

  if args.optuna:
    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective_partial, timeout=60 * 60 * 6)

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
      print(f"    {key}: {value}")

  else:
    objective_partial(None)
