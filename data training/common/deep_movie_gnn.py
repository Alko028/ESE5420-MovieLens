"""Two genuine movie-graph convolutions on immutable history-only signals.

No target labels enter this module.  In particular, the second graph filter
propagates learned hidden signals on the whole movie graph before gathering
target movies; it is not an MLP applied to cached target-local features.
"""

import numpy as np
from scipy import sparse
import torch
from torch import nn


def build_full_history_taps(graph, taps=3):
    """Return float32 [user, movie, 2*taps] in rate0,mask0,rate1,... order.

    Use exactly the directed, incoming-row-normalized graph of the original
    shallow GNN.  History and the fixed movie graph are the only inputs used.
    """
    from gnn import normalized_adjacency

    if type(taps) is not int or taps < 1:
        raise ValueError("taps must be a positive integer")
    rating = np.asarray(graph["history_rating"], dtype=np.float32)
    mask = np.asarray(graph["history_mask"])
    if rating.ndim != 2 or mask.shape != rating.shape or not np.array_equal(mask, rating > 0):
        raise ValueError("History rating/mask mismatch")
    if not np.isfinite(rating).all() or not np.isin(rating, [0, 1, 2, 3, 4, 5]).all():
        raise ValueError("Invalid history ratings")
    adjacency = normalized_adjacency(graph)
    signals = [rating / np.float32(5), mask.astype(np.float32)]
    values = []
    for order in range(taps):
        values.extend(signals)
        if order + 1 < taps:
            signals = [(adjacency @ signal.T).T for signal in signals]
    result = np.stack(values, axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite polynomial history signals")
    return result


class TwoLayerMovieGNN(nn.Module):
    """Two K-tap polynomial graph filters, each followed by ReLU.

    ``adjacency[target, source]`` must be the same fixed row-normalized CSR
    adjacency used by the shallow GNN.  A minibatch deduplicates users, computes
    both layers for all their movies, then gathers the requested target pairs.
    For K=3,H=32 there are 3,361 learned parameters.
    """

    def __init__(self, adjacency, taps=3, hidden_channels=32):
        super().__init__()
        if type(taps) is not int or taps < 1 or type(hidden_channels) is not int or hidden_channels < 1:
            raise ValueError("taps and hidden_channels must be positive integers")
        adjacency = sparse.csr_matrix(adjacency, dtype=np.float32)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1] or not adjacency.shape[0]:
            raise ValueError("Expected a nonempty square movie adjacency")
        if not np.isfinite(adjacency.data).all() or (adjacency.data < 0).any():
            raise ValueError("Adjacency weights must be finite and nonnegative")
        row_sum = np.asarray(adjacency.sum(axis=1)).ravel()
        if not np.all(np.isclose(row_sum, 1., atol=1e-6) | (row_sum == 0)):
            raise ValueError("Adjacency must have unit or zero row sums")
        coo = adjacency.tocoo()
        indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
        values = torch.from_numpy(coo.data.copy())
        self.register_buffer("adjacency", torch.sparse_coo_tensor(indices, values, coo.shape).coalesce())
        self.taps = taps
        self.hidden_channels = hidden_channels
        self.first = nn.Linear(2 * taps, hidden_channels)
        self.second = nn.Linear(taps * hidden_channels, hidden_channels)
        self.readout = nn.Linear(hidden_channels, 1)

    def _propagate(self, hidden):
        # Arrange movie first so every user/channel uses identical A @ signal.
        users, movies, channels = hidden.shape
        flat = hidden.permute(1, 0, 2).reshape(movies, users * channels)
        return torch.sparse.mm(self.adjacency, flat).reshape(movies, users, channels).permute(1, 0, 2)

    def forward(self, history_taps_tensor, user_index_tensor, movie_index_tensor):
        if history_taps_tensor.ndim != 3 or history_taps_tensor.shape[1:] != (self.adjacency.shape[0], 2 * self.taps):
            raise ValueError("Expected full [user, movie, 2*taps] history signals")
        if user_index_tensor.ndim != 1 or movie_index_tensor.shape != user_index_tensor.shape or not len(user_index_tensor):
            raise ValueError("Expected aligned nonempty target indices")
        if user_index_tensor.dtype != torch.long or movie_index_tensor.dtype != torch.long:
            raise ValueError("Target indices must use torch.long")
        unique_users, inverse = torch.unique(user_index_tensor, sorted=True, return_inverse=True)
        hidden = torch.relu(self.first(history_taps_tensor[unique_users]))
        hidden_taps = [hidden]
        for _ in range(1, self.taps):
            hidden = self._propagate(hidden)
            hidden_taps.append(hidden)
        hidden = torch.relu(self.second(torch.cat(hidden_taps, dim=-1)))
        logits = self.readout(hidden).squeeze(-1)
        return logits[inverse, movie_index_tensor]
