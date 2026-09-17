"""Graph convolution architectures and their batching capabilities."""
from functools import partial
from .gcn import GCN
from .gat import GAT
from .sage import GraphSAGE
from .cgnn import CGNN
from .stgcn import STGCN

REGISTRY = {
    "gcn": GCN, "gat": GAT, "sage": GraphSAGE,
    "cgnn": CGNN,
    "cgnn_sage": partial(CGNN, conv_type="sage"),
    "cgnn_gat": partial(CGNN, conv_type="gat"),
    "stgcn": STGCN,
    "stgcn_sage": partial(STGCN, conv_type="sage"),
    "stgcn_gat": partial(STGCN, conv_type="gat"),
}
STATIC_TYPES = frozenset({"gcn", "cgnn", "cgnn_sage", "stgcn", "stgcn_sage"})
