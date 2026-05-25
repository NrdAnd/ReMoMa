import numpy as np
import pandas as pd

def build_graph_node_features(
    lobster_csv_path,
    n_lags=151,
    save_path="graph_node_features.npy"
):

    df = pd.read_csv(lobster_csv_path, header=None)
    data = df.values.astype(np.float32)

    T, cols = data.shape
    N_LEVELS = cols // 4

    ask_prices = data[:, 0::4]
    ask_vols   = data[:, 1::4]
    bid_prices = data[:, 2::4]
    bid_vols   = data[:, 3::4]

    node_features = []
    node_names = []

    def make_shifted_series(x, lag):

        if lag == 0:
            return x.copy()

        return x[:-lag]

    node_features = {}

    for level in range(N_LEVELS):

        for lag in range(n_lags):

            node_name = f"bid_{level}_lag_{lag}"

            node_features[node_name] = {

                "price":
                    bid_prices[:T-lag, level],

                "volume":
                    bid_vols[:T-lag, level]
            }
    for level in range(N_LEVELS):

        for lag in range(n_lags):

            node_name = f"ask_{level}_lag_{lag}"

            node_features[node_name] = {

                "price":
                    bid_prices[:T-lag, level],

                "volume":
                    bid_vols[:T-lag, level]
            }

    print(node_features.shape)

    return node_features, node_names