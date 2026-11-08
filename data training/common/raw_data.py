"""Parse official MovieLens 100K files without loading user demographics."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MovieLens:
    # Original u.data line number (zero-based) is the stable row_id.
    pairs: np.ndarray  # [N, 2], official one-based user_id, movie_id
    ratings: np.ndarray  # [N], 1..5
    movie_genres: np.ndarray  # [movies, 19], ordered by movie_id
    genre_names: tuple[str, ...]
    n_users: int
    n_movies: int


def load_movielens(directory: Path) -> MovieLens:
    directory = Path(directory)
    info = {}
    for line in (directory / "u.info").read_text(encoding="ascii").splitlines():
        count, name = line.split()
        info[name] = int(count)
    raw = np.loadtxt(directory / "u.data", dtype=np.int64, delimiter="\t", ndmin=2)
    if raw.shape != (info["ratings"], 4):
        raise ValueError("u.data must match u.info and contain exactly four columns")
    pairs = raw[:, :2].astype(np.int32)
    ratings = raw[:, 2].astype(np.int8)
    if not np.isin(raw[:, 2], np.arange(1, 6)).all():
        raise ValueError("Ratings must be integers in [1, 5]")
    if (raw[:, 3] <= 0).any():
        raise ValueError("Invalid timestamp; timestamps are validated, not used as inputs")
    for col, size in [(0, info["users"]), (1, info["items"])]:
        if not np.array_equal(np.unique(pairs[:, col]), np.arange(1, size + 1)):
            raise ValueError("Expected contiguous, one-based official IDs")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("Duplicate user/movie pairs are not allowed")
    genres = {}
    for line in (directory / "u.genre").read_text(encoding="ascii").splitlines():
        if line.strip():
            name, index = line.split("|")
            genres[int(index)] = name
    if set(genres) != set(range(19)):
        raise ValueError("Expected genre IDs 0..18, including unknown")
    movie_genres = np.zeros((info["items"], 19), dtype=np.int8)
    seen = set()
    # u.item is ISO-8859-1, not UTF-8. Dates/titles/URLs are not model inputs.
    for line in (directory / "u.item").read_text(encoding="latin-1").splitlines():
        fields = line.split("|")
        if len(fields) != 24:
            raise ValueError("u.item must have 5 metadata and 19 genre fields")
        movie_id = int(fields[0])
        if movie_id in seen or not 1 <= movie_id <= info["items"]:
            raise ValueError("Invalid or duplicate u.item movie ID")
        flags = np.array([int(x) for x in fields[5:]], dtype=np.int64)
        if not np.isin(flags, [0, 1]).all():
            raise ValueError("Genre flags must be binary")
        movie_genres[movie_id - 1] = flags
        seen.add(movie_id)
    if len(seen) != info["items"]:
        raise ValueError("u.item is missing movie IDs")
    return MovieLens(pairs, ratings, movie_genres,
                     tuple(genres[i] for i in range(19)), info["users"], info["items"])
