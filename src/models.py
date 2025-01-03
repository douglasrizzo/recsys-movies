import lightning
import torch
from torch import nn


class TwoTowerNetwork(lightning.LightningModule):
  """A two-tower neural network for user-item recommendation.

  The network consists of two towers: the user tower and the item tower. The user
  tower processes the user features and the item tower processes the item features.
  The final prediction is computed by taking the dot product of the two tower
  outputs.

  Attributes:
      user_tower: The user tower neural network.
      item_tower: The item tower neural network.
  """

  def __init__(self, lr: float = 1e-3, dropout: float = 0.5, n_features: int = 128, n_layers: int = 3) -> None:
    """Initialize a two-tower neural network for user-item recommendation."""
    super().__init__()
    # The user tower processes the user features
    self.user_tower = self.make_tower(dropout, n_features, n_layers)
    # The item tower processes the item features
    self.item_tower = self.make_tower(dropout, n_features, n_layers)

    self.lr = lr

  @staticmethod
  def make_tower(dropout, n_features, n_layers):
    layers = []
    for _ in range(n_layers):
      layers.extend([nn.LazyLinear(out_features=n_features), nn.Tanh()])
      if _ != n_layers - 1:
        layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

  def compute_loss(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Compute the loss for the given batch.

    Args:
        batch: A tuple of three tensors: user features, item features, and ratings.

    Returns:
        torch.Tensor: The loss for the current batch.
    """
    user_features, item_features, rating = batch
    user_embedding = self.user_tower(user_features)
    item_embedding = self.item_tower(item_features)
    y_hat = (user_embedding * item_embedding).sum(dim=1)
    return nn.functional.mse_loss(y_hat, rating), (y_hat - rating).abs().mean()

  def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
    """Compute the loss for the given batch.

    Args:
        batch: A tuple of three tensors: user features, item features, and ratings.
        batch_idx: The index of the current batch (unused).

    Returns:
        torch.Tensor: The loss for the current batch.
    """
    mse, mae = self.compute_loss(batch)
    self.log("Loss/MSE Train", mse)
    self.log("Loss/MAE Train", mae)
    return mse

  def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
    """Compute the loss for the given batch.

    Args:
        batch: A tuple of three tensors: user features, item features, and ratings.
        batch_idx: The index of the current batch (unused).

    Returns:
        torch.Tensor: The loss for the current batch.
    """
    mse, mae = self.compute_loss(batch)
    self.log("Loss/MSE Val", mse)
    self.log("Loss/MAE Val", mae)
    return mse

  def configure_optimizers(self) -> torch.optim.Adam:
    """Configure the optimizer for the model.

    Returns:
        torch.optim.Adam: The optimizer for training the model.
    """
    return torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))
