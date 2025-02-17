import lightning
import torch
from torch import nn


class TwoTowerNetwork(lightning.LightningModule):
  """A two-tower neural network for user-item recommendation.

  The network consists of two towers: the user tower and the item tower. The user
  tower processes the user features and the item tower processes the item features.
  The final prediction is computed by taking the dot product of the two tower
  outputs.

  Attributes
  ----------
      user_tower: The user tower neural network.
      item_tower: The item tower neural network.
  """

  def __init__(
    self,
    output_dims: list[int] | None = None,
    dropout: float = 0.5,
    lr: float = 1e-3,
  ) -> None:
    """Initialize a two-tower neural network for user-item recommendation.

    The two-tower neural network consists of two towers: the user tower and the
    item tower. The user tower processes the user features and the item tower
    processes the item features. The final prediction is computed by taking the
    dot product of the two tower outputs.

    Parameters
    ----------
    output_dims : list[int], optional
        The output dimensions of the user and item towers. If `None`, the
        default output dimensions are used. The default is `None`.
    dropout : float, optional
        The dropout probability. The default is 0.5.
    lr : float, optional
        The learning rate. The default is 1e-3.
    """
    if output_dims is None:
      output_dims = [64, 32]
    super().__init__()
    # The user tower processes the user features
    self.user_tower = self.make_tower(dropout, output_dims)
    # The item tower processes the item features
    self.item_tower = self.make_tower(dropout, output_dims)

    self.lr = lr

  @staticmethod
  def make_tower(dropout: float, output_dims: list[int]) -> nn.Sequential:
    """Construct a neural network tower.

    Parameters
    ----------
    dropout : float
        The dropout probability.
    output_dims : list[int]
        The number of neurons in each layer.

    Returns
    -------
    nn.Sequential
        The neural network tower.
    """
    layers = []
    for layer_num, output_dim in enumerate(output_dims):
      layers.extend((
        (
          nn.LazyLinear(out_features=output_dim)
          if layer_num == 0
          else nn.Linear(
            in_features=output_dims[layer_num - 1],
            out_features=output_dim,
          )
        ),
        nn.Tanh(),
      ))
      if layer_num != len(output_dims) - 1:
        layers.append(
          nn.Dropout(p=dropout),
        )
    return nn.Sequential(*layers)

  def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
    """Perform a forward pass through the two-tower network.

    This function processes the user and item features through their respective
    towers and computes the dot product of the resulting embeddings.

    Parameters
    ----------
    user_features : torch.Tensor
        The input tensor representing user features.
    item_features : torch.Tensor
        The input tensor representing item features.

    Returns
    -------
    torch.Tensor
        The predicted interaction score for each user-item pair.
    """
    user_embedding = self.user_tower(user_features)
    item_embedding = self.item_tower(item_features)
    return (user_embedding * item_embedding).sum(dim=1)

  def inference(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
    """Compute the interaction scores between all users and items.

    Parameters
    ----------
    user_features : torch.Tensor
        The input tensor representing user features.
    item_features : torch.Tensor
        The input tensor representing item features.

    Returns
    -------
    torch.Tensor
        A tensor of shape (num_users, num_items) where each element at
        index (i, j) represents the predicted interaction score between the
        i-th user and the j-th item.
    """
    user_embedding = self.user_tower(user_features)
    item_embedding = self.item_tower(item_features)
    return user_embedding @ item_embedding.T

  def compute_loss(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Compute the loss for the given batch.

    Args:
        batch: A tuple of three tensors: user features, item features, and ratings.

    Returns
    -------
        torch.Tensor: The loss for the current batch.
    """
    user_features, item_features, rating = batch
    y_hat = self.forward(user_features, item_features)
    return nn.functional.mse_loss(y_hat, rating), (y_hat - rating).abs().mean()

  def training_step(
    self,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_idx: int,  # noqa: ARG002
  ) -> torch.Tensor:
    """Compute the loss for the given batch.

    Parameters
    ----------
    batch : tuple of torch.Tensor
        A tuple containing three tensors: user features, item features, and ratings.
    batch_idx : int
        The index of the current batch (unused).

    Returns
    -------
    torch.Tensor
        The loss for the current batch.
    """
    mse, mae = self.compute_loss(batch)
    self.log("Loss/MSE Train", mse)
    self.log("Loss/MAE Train", mae)
    return mse

  def validation_step(
    self,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_idx: int,  # noqa: ARG002
  ) -> torch.Tensor:
    """Compute the loss for the given batch.

    Parameters
    ----------
    batch : tuple of torch.Tensor
        A tuple containing three tensors: user features, item features, and ratings.
    batch_idx : int
        The index of the current batch (unused).

    Returns
    -------
    torch.Tensor
        The loss for the current batch.
    """
    mse, mae = self.compute_loss(batch)
    self.log("Loss/MSE Val", mse)
    self.log("Loss/MAE Val", mae)
    return mse

  def configure_optimizers(self) -> torch.optim.Adam:
    """Configure the optimizer for the model.

    Returns
    -------
    torch.optim.Adam
        The optimizer for training the model.
    """
    optimizer = torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))
    return {
      "optimizer": optimizer,
      "lr_scheduler": {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min"),
        "monitor": "Loss/MSE Val",
        # "frequency": "indicates how often the metric is updated",
        # If "monitor" references validation metrics, then "frequency" should be set to a
        # multiple of "trainer.check_val_every_n_epoch".
      },
    }
