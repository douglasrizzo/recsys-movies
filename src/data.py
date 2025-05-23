import pathlib as pl
from os import cpu_count

import lightning
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import scale
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import movielens


class UserItemDataset(Dataset):
  """
  A PyTorch dataset of user-item ratings.

  The dataset is created from the MovieLens dataset, which is a collection of
  user-item ratings. The ratings are split into training, validation, and test
  datasets. The dataset is used to train a two-tower neural network.
  """

  def __init__(
    self,
    items_path: str,
    users_path: str,
    ratings_path: str,
    item_cols_to_normalize_path: str | None = None,
    user_cols_to_normalize_path: str | None = None,
    sample_frac: float = 1.0,
    random_seed: int = 42,
  ) -> None:
    """Initialize the UserItemDataset.

    Parameters
    ----------
    items_path : str
        Path to the parquet file containing item features.
    users_path : str
        Path to the parquet file containing user features.
    ratings_path : str
        Path to the parquet file containing ratings.
    item_cols_to_normalize_path : str | None, optional
        Path to the file with columns to normalize for items, by default None.
    user_cols_to_normalize_path : str | None, optional
        Path to the file with columns to normalize for users, by default None.
    sample_frac : float, optional
        Fraction of the dataset to sample, by default 1.0.
    random_seed : int, optional
        Random seed for sampling, by default 42.
    """
    self.ratings_df = pd.read_parquet(ratings_path).sample(frac=sample_frac, random_state=random_seed).sort_index()

    users_df = pd.read_parquet(users_path).sort_index()
    assert users_df.index.nlevels == 1 and users_df.index.name == "id"
    users_df = users_df.loc[users_df.index.isin(self.ratings_df.index.levels[0])]

    if user_cols_to_normalize_path:
      user_features_to_normalize = pl.Path(user_cols_to_normalize_path).read_text(encoding="utf-8").split("\n")
      users_df[user_features_to_normalize] = users_df[user_features_to_normalize].astype(float)
      users_df.loc[:, user_features_to_normalize] = scale(users_df[user_features_to_normalize])

    items_df = pd.read_parquet(items_path).sort_index()
    assert items_df.index.nlevels == 1 and items_df.index.name == "id"
    items_df = items_df.loc[items_df.index.isin(self.ratings_df.index.levels[1])]

    if item_cols_to_normalize_path:
      item_features_to_normalize = pl.Path(item_cols_to_normalize_path).read_text(encoding="utf-8").split("\n")
      items_df[item_features_to_normalize] = items_df[item_features_to_normalize].astype(float)
      items_df.loc[:, item_features_to_normalize] = scale(items_df[item_features_to_normalize])

    self.ratings_df = self.ratings_df.loc[
      self.ratings_df.index.get_level_values("userId").isin(users_df.index.unique())
      & self.ratings_df.index.get_level_values("itemId").isin(items_df.index.unique())
    ]

    self.user_id_map = {int(user_id): i for i, user_id in enumerate(users_df.index.unique())}
    self.item_id_map = {int(item_id): i for i, item_id in enumerate(items_df.index.unique())}

    self.user_features = tensor(users_df.squeeze().to_numpy(), dtype=torch.float)
    self.item_features = tensor(items_df.squeeze().to_numpy(), dtype=torch.float)

  def __len__(self) -> int:
    """Return the number of ratings in the dataset.

    Returns
    -------
    int
        The number of ratings in the dataset.
    """
    return len(self.ratings_df)

  def __getitem__(self, idx: int) -> tuple[tensor, tensor, tensor]:
    """Get the user and item features and rating for a given index.

    Parameters
    ----------
    idx : int
        The index of the rating.

    Returns
    -------
    tuple
        A tuple of three tensors, the user features, item features, and rating.
    """
    row = self.ratings_df.iloc[[idx]]
    user_id = row.index.get_level_values("userId").item()
    item_id = row.index.get_level_values("itemId").item()
    rating = row["rating"].iloc[0]

    # recover the original indices from the sequential indices
    user_id = self.user_id_map[user_id]
    item_id = self.item_id_map[item_id]

    # get the features for the user and item
    user_features = self.user_features[user_id]
    item_features = self.item_features[item_id]
    return user_features, item_features, tensor(rating, dtype=torch.float)

  @property
  def n_user_features_in(self) -> int:
    """The number of features in the user feature vectors.

    This is the number of columns in `user_features` and should be used to set the
    `n_user_features_in` parameter of the model.

    Returns
    -------
    int
        The number of features in the user feature vectors.
    """
    return self.user_features.shape[1]

  @property
  def n_item_features_in(self) -> int:
    """The number of features in the item feature vectors.

    This is the number of columns in `item_features` and should be used to set the
    `n_item_features_in` parameter of the model.

    Returns
    -------
    int
        The number of features in the item feature vectors.
    """
    return self.item_features.shape[1]


class UserItemDataModule(lightning.LightningDataModule):
  """A LightningDataModule that loads the MovieLens dataset."""

  def __init__(
    self,
    items_path: str,
    users_path: str,
    ratings_path: str,
    item_cols_to_normalize_path: str | None = None,
    user_cols_to_normalize_path: str | None = None,
    batch_size: int = 1024,
    validation_split: float = 0.2,
    shuffle_dataset: bool = True,
    random_seed: int = 42,
    sample_frac: float = 1.0,
    train_workers: int = cpu_count() - 1,
    val_workers: int = cpu_count() - 1,
  ) -> None:
    """
    Initialize the UserItemDataModule.

    Parameters
    ----------
    items_path : str
        The path to the items data file.
    users_path : str
        The path to the users data file.
    ratings_path : str
        The path to the ratings data file.
    item_cols_to_normalize_path : str | None, optional
        The path to a file containing the item features to normalize. If `None`, no
        normalization is done. The default is `None`.
    user_cols_to_normalize_path : str | None, optional
        The path to a file containing the user features to normalize. If `None`, no
        normalization is done. The default is `None`.
    batch_size : int, optional
        The batch size to use for training and evaluation. The default is 1024.
    validation_split : float, optional
        The proportion of the dataset to use for validation. The default is 0.2.
    shuffle_dataset : bool, optional
        Whether to shuffle the dataset or not. The default is `True`.
    random_seed : int, optional
        The random seed to use for shuffling the dataset. The default is 42.
    sample_frac : float, optional
        The proportion of the dataset to use for training and evaluation. The default
        is 1.0.
    train_workers : int, optional
        The number of workers to use for training. The default is `cpu_count() - 1`.
    val_workers : int, optional
        The number of workers to use for validation. The default is `cpu_count() - 1`.
    """
    super().__init__()
    self.items_path = items_path
    self.users_path = users_path
    self.ratings_path = ratings_path
    self.item_cols_to_normalize_path = item_cols_to_normalize_path
    self.user_cols_to_normalize_path = user_cols_to_normalize_path
    self.batch_size = batch_size
    self.validation_split = validation_split
    self.shuffle_dataset = shuffle_dataset
    self.random_seed = random_seed
    self.sample_frac = sample_frac
    self.train_workers = train_workers
    self.val_workers = val_workers

  def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
    """Set up the dataset and data samplers for training and validation."""
    self.dataset = UserItemDataset(
      self.items_path,
      self.users_path,
      self.ratings_path,
      self.item_cols_to_normalize_path,
      self.user_cols_to_normalize_path,
      self.sample_frac,
      self.random_seed,
    )
    dataset_size = len(self.dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(self.validation_split * dataset_size))
    if self.shuffle_dataset:
      generator = np.random.default_rng(seed=self.random_seed)
      indices = generator.permutation(dataset_size)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    self.train_sampler = SubsetRandomSampler(train_indices)
    self.valid_sampler = SubsetRandomSampler(val_indices)

  def prepare_data(self) -> None:  # noqa: PLR6301
    """Download MovieLens dataset and generate ratings file if not already downloaded/generated.

    This is an overridden method from LightningDataModule.
    """
    movielens.main()

  def train_dataloader(self) -> DataLoader:
    """Get the PyTorch DataLoader for the training split.

    Returns
    -------
        DataLoader: The DataLoader for the training split.
    """
    return DataLoader(
      self.dataset,
      batch_size=self.batch_size | self.batch_size,
      sampler=self.train_sampler,
      num_workers=self.train_workers,
    )

  def val_dataloader(self) -> DataLoader:
    """Get the PyTorch DataLoader for the validation split.

    Returns
    -------
        DataLoader: The DataLoader for the validation split.
    """
    return DataLoader(
      self.dataset,
      batch_size=self.batch_size | self.batch_size,
      sampler=self.valid_sampler,
      num_workers=self.val_workers,
    )
