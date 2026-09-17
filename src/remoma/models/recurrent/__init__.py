"""Recurrent graph architectures; all use tensor batching."""
from .recurrent_sparse_sthnn import RecurrentSparseSTHNNClassifier

REGISTRY = {"recurrent_sparse_sthnn": RecurrentSparseSTHNNClassifier}
