# %%
import os
import pathlib as pl
from calendar import monthrange

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def download_from_kaggle() -> tuple[pl.Path, pl.Path, pl.Path, pl.Path, pl.Path, pl.Path]:
  """Download latest version of MovieLens and TMDB datasets and return paths to all files."""
  tmdb_path = next(pl.Path(kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")).glob("*.csv"))
  print(tmdb_path)
  links_path, genometags_path, movies_path, genomescores_path, tags_path, ratings_path = list(
    pl.Path(kagglehub.dataset_download("grouplens/movielens-20m-dataset")).iterdir()
  )
  return (links_path, genometags_path, movies_path, genomescores_path, tags_path, ratings_path, tmdb_path)


def downcast(df: pd.DataFrame) -> pd.DataFrame | None:
  """Downcast the numerical columns of a pandas DataFrame for minimal memory usage.

  This function iterates over all numerical columns in the DataFrame and attempts
  to downcast them to the smallest possible numeric type. It handles both integer
  and float types, reducing memory footprint without altering data.

  :param df: Input DataFrame with numerical columns to be downcasted.
  :return: DataFrame with downcasted numerical columns.
  """
  for col in df.select_dtypes(include=["number"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="integer" if pd.api.types.is_integer_dtype(df[col]) else "float")
  return df


def solve_sparse_feature(
  df_to_count: pd.DataFrame,
  id_col: str,
  feat_col: str,
  concentration: float = 0.9,
) -> pd.DataFrame:
  """Solve sparse features for a given DataFrame.

  This function takes a DataFrame with an id column and a feature column,
  and returns a new DataFrame with the feature column grouped to a set
  of values that together represent the given concentration of the total
  number of values.
  The function plots a pie chart of the feature distribution and a bar
  chart of the cumulative sum of the feature distribution.

  Parameters
  ----------
  df_to_count : pandas.DataFrame
      DataFrame with id column and feature column.
  id_col : str
      Name of the id column.
  feat_col : str
      Name of the feature column.
  concentration : float, optional
      Concentration of the total number of values to represent with the
      grouped feature. Default is 0.9.

  Returns:
  -------
  pandas.DataFrame
      DataFrame with the grouped feature column.
  """
  fig, axes = plt.subplots(1, 2, figsize=(12, 6))
  feat_count = df_to_count[feat_col].value_counts(normalize=True).sort_values(ascending=False)
  feat_count.plot.pie(autopct="%1.1f%%", ax=axes[0])
  feat_cumsum = feat_count.cumsum()
  feat_mask = feat_cumsum <= concentration
  feats_to_keep = feat_mask.index[feat_mask]
  df_to_count["grouped_feature"] = df_to_count[feat_col].apply(lambda x: x if x in feats_to_keep else "other")
  counts = df_to_count.groupby([id_col, "grouped_feature"]).size().reset_index(name="count")
  final_features = downcast(counts.pivot_table(index=id_col, columns="grouped_feature", values="count").fillna(0))
  final_features = final_features.rename(columns={c: f"{feat_col}_count_{c}" for c in final_features.columns})
  ((final_features != 0).sum() / (final_features != 0).sum().sum()).sort_values(ascending=False).cumsum().plot.bar(
    ax=axes[1]
  )
  plt.tight_layout()
  plt.show()
  return final_features


def generate_movie_features(movies_path: str, links_path: str, tmdb_path: str, output_path: str) -> pd.DataFrame:
  """Generate and save movie features from provided datasets.

  This function takes paths to the movies, links, and TMDB datasets, processes them to extract
  and generate features, and saves the resulting DataFrame into a specified output path.

  Parameters
  ----------
  movies_path : str
      Path to the CSV file containing movie data.
  links_path : str
      Path to the CSV file containing movie links.
  tmdb_path : str
      Path to the CSV file containing TMDB movie data.
  output_path : str
      Directory path where the output features and normalization file will be saved.

  Returns:
  -------
  pd.DataFrame
      A DataFrame containing the processed movie features.
  """
  features_to_normalize = []
  movie_features = (
    pd.read_csv(movies_path).merge(pd.read_csv(links_path), on="movieId").set_index("movieId").sort_index()
  )
  del movie_features["title"]
  movie_features = movie_features.join(
    movie_features["genres"].str.get_dummies(sep="|").add_prefix("genre_").astype(int)
  )
  genre_columns = movie_features.columns.str.startswith("genre_")
  features_to_normalize.extend(movie_features.columns[genre_columns].tolist())
  del movie_features["genres"]

  movie_features = movie_features.merge(
    pd.read_csv(tmdb_path)[
      [
        "id",
        "adult",
        "release_date",
        "overview",
        "budget",
        "title",
        "original_language",
        "revenue",
        "runtime",
        "vote_average",
        "vote_count",
        "popularity",
      ]
    ],
    left_on="tmdbId",
    right_on="id",
    how="inner",
  )
  movie_features["release_date"] = pd.to_datetime(movie_features["release_date"])
  movie_features["release_year"] = movie_features["release_date"].dt.year
  movie_features["release_month_sin"] = np.sin(2 * np.pi * movie_features["release_date"].dt.month / 12)
  movie_features["release_month_cos"] = np.cos(2 * np.pi * movie_features["release_date"].dt.month / 12)
  movie_features["release_dow_sin"] = np.sin(2 * np.pi * movie_features["release_date"].dt.weekday / 7)
  movie_features["release_dow_cos"] = np.cos(2 * np.pi * movie_features["release_date"].dt.weekday / 7)

  features_to_normalize.extend([
    "budget",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
    "release_year",
  ])

  def max_days_in_month(year, month):
    return monthrange(year, month)[1]

  max_days = movie_features.apply(
    lambda row: max_days_in_month(int(row["release_year"]), int(row["release_date"].month))
    if (not pd.isna(row["release_year"]) and not pd.isna(row["release_date"].month))
    else None,
    axis=1,
  )
  day_normalized = movie_features["release_date"].dt.day / max_days
  movie_features["release_day_sin"] = np.sin(2 * np.pi * day_normalized)
  movie_features["release_day_cos"] = np.cos(2 * np.pi * day_normalized)

  movie_features["adult"] = movie_features["adult"].astype(int)

  movie_language_features = solve_sparse_feature(
    movie_features[["original_language"]].reset_index(names="movieId"),
    "movieId",
    "original_language",
    concentration=0.93,
  )

  features_to_normalize.extend(movie_language_features.columns)
  movie_features = movie_features.merge(movie_language_features, left_index=True, right_index=True)

  model = SentenceTransformer("all-MiniLM-L6-v2")
  title_embeddings = model.encode(movie_features["title"].to_list(), show_progress_bar=True, batch_size=64)
  movie_features = pd.concat(
    [
      movie_features,
      pd.DataFrame(title_embeddings, columns=[f"title_emb_{idx}" for idx in range(title_embeddings.shape[1])]),
    ],
    axis=1,
  )
  movie_features.loc[movie_features["overview"].isna(), "overview"] = ""
  overview_embeddings = model.encode(movie_features["overview"].to_list(), show_progress_bar=True, batch_size=64)
  movie_features = pd.concat(
    [
      movie_features,
      pd.DataFrame(overview_embeddings, columns=[f"overview_emb_{idx}" for idx in range(overview_embeddings.shape[1])]),
    ],
    axis=1,
  )

  movie_features = movie_features.drop(
    ["imdbId", "tmdbId", "release_date", "title", "overview", "original_language"], axis=1
  )

  downcast(movie_features)
  movie_features = movie_features[~movie_features.isna().any(axis=1)]

  output_path = pl.Path(output_path)
  output_path.mkdir(parents=True, exist_ok=True)
  (output_path / "movie_features_to_normalize.txt").open(mode="w").write("\n".join(features_to_normalize))
  movie_features = movie_features.set_index("id")
  movie_features.to_parquet(output_path / "movie_features.parquet")

  return movie_features


def generate_user_features(
  ratings_path: str,
  tags_path: str,
  links_path: str,
  tmdb_path: str,
  movie_features_path: str,
  output_path: str,
) -> pd.DataFrame:
  """Generate and save user features."""
  ratings = downcast(pd.read_csv(ratings_path))
  tags = downcast(pd.read_csv(tags_path))
  user_movies = pd.concat([ratings[["userId", "movieId"]], tags[["userId", "movieId"]]]).drop_duplicates()
  features_to_normalize = []

  movie_features_genres = downcast(pd.read_parquet(movie_features_path))
  genres = [c for c in movie_features_genres.columns if c.startswith("genre_")]
  movie_features_genres = movie_features_genres[genres]

  user_features = downcast(
    user_movies.merge(movie_features_genres, right_index=True, left_on="movieId", how="inner")
    .groupby("userId")[genres]
    .sum()
  )
  features_to_normalize.extend(genres)

  user_movie_genre_scores = (
    user_movies.merge(ratings, how="inner", on=["userId", "movieId"])[["userId", "movieId", "rating"]]
    .merge(movie_features_genres, right_index=True, left_on="movieId", how="inner")
    .merge(downcast(pd.read_csv(links_path)), on="movieId")
    .merge(
      downcast(
        pd.read_csv(tmdb_path)[
          [
            "id",
            "release_date",
            "budget",
            "runtime",
            "vote_average",
            "popularity",
          ]
        ]
      ),
      left_on="tmdbId",
      right_on="id",
      how="inner",
    )
    .set_index(["userId", "movieId"])
    .sort_index()
  )
  user_movie_genre_scores["release_year"] = pd.to_datetime(user_movie_genre_scores["release_date"]).dt.year
  del user_movie_genre_scores["release_date"]

  downcast(user_movie_genre_scores)

  cols_to_avg = [
    "release_year",
    "budget",
    "runtime",
    "vote_average",
    "popularity",
  ]

  orelha = (
    user_movie_genre_scores[cols_to_avg]
    .groupby(level=0)
    .agg(["mean", "std"])
    .stack(level=0)
    .rename_axis(["userId", "metric"])
    .unstack()
  )
  orelha.columns = [f"{agg}_{c}" for c in cols_to_avg for agg in ["mean", "std"]]
  user_features = downcast(
    user_features.merge(
      orelha,
      left_index=True,
      right_index=True,
    )
  )
  features_to_normalize.extend([f"{agg}_{c}" for c in cols_to_avg for agg in ["mean", "std"]])

  results = (
    user_movie_genre_scores[genres]
    .multiply(user_movie_genre_scores["rating"], axis="index")
    .groupby(level=0)
    .mean()
    .add_prefix("avg_rating_")
    .fillna(0)
  )

  user_features = downcast(user_features.merge(results, left_index=True, right_index=True))
  features_to_normalize.extend(results.columns)

  misc_data = user_movies.merge(downcast(pd.read_csv(links_path)), on="movieId").merge(
    downcast(pd.read_csv(tmdb_path)[["id", "original_language"]]),
    left_on="tmdbId",
    right_on="id",
    how="inner",
  )

  user_language_features = solve_sparse_feature(
    misc_data[["userId", "original_language"]].drop_duplicates(), "userId", "original_language"
  )

  user_features = user_features.merge(user_language_features, left_index=True, right_index=True)
  features_to_normalize.extend(user_language_features.columns)
  user_features = user_features.dropna()
  user_features.index.names = ["id"]
  output_path = pl.Path(output_path)
  output_path.mkdir(parents=True, exist_ok=True)
  (output_path / "user_features_to_normalize.txt").write_text("\n".join(features_to_normalize))
  user_features.to_parquet(output_path / "user_features.parquet")

  return user_features


def visualize_feature_variance(movies_path: str, users_path: str) -> None:
  """Visualize feature variance in both datasets using PCA."""
  movies_df = pd.read_parquet(movies_path)
  users_df = pd.read_parquet(users_path)
  fig, axes = plt.subplots(2, 1, figsize=(20, 10))

  pca = PCA()

  pca.fit_transform(scale(movies_df))
  axes[0].bar(height=pca.explained_variance_ratio_[:50], x=pca.get_feature_names_out()[:50])
  axes[0].tick_params("x", rotation=45)
  pca.fit_transform(scale(users_df))
  axes[1].bar(height=pca.explained_variance_ratio_[:50], x=pca.get_feature_names_out()[:50])
  axes[1].tick_params("x", rotation=45)


def generate_ratings_file(input_path: str, output_path: str) -> None:
  """Prepare ratings file from MovieLens dataset."""
  pd.read_csv(input_path)[["userId", "movieId", "rating"]].rename(columns={"movieId": "itemId"}).set_index([
    "userId",
    "itemId",
  ]).to_parquet(pl.Path(output_path) / "ratings.parquet")


def main():
  links_path, genometags_path, movies_path, genomescores_path, tags_path, ratings_path, tmdb_path = (
    download_from_kaggle()
  )

  if not pl.Path("data/movie_features.parquet").exists():
    generate_movie_features(movies_path, links_path, tmdb_path, output_path="data/")
  if not pl.Path("data/user_features.parquet").exists():
    generate_user_features(
      ratings_path,
      tags_path,
      links_path,
      tmdb_path,
      movie_features_path="data/movie_features.parquet",
      output_path="data/",
    )
  if not pl.Path("data/ratings.parquet").exists():
    generate_ratings_file(ratings_path, output_path="data/")


if __name__ == "__main__":
  main()
