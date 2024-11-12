import os
import osmnx
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
from pyproj import Transformer


def preprocess_datasets(graph, foil_path: str):

    nodes = osmnx.convert.graph_to_gdfs(graph, edges=False, node_geometry=False)[
        ["x", "y"]
    ]
    kd_tree = cKDTree(nodes)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32618")

    for i in range(1, 13):
        df = pd.read_parquet(os.path.join(foil_path, f"trip_data_{i}.parquet"))
        for d in ("pickup_", "dropoff_"):
            X, Y = transformer.transform(
                df[d + "latitude"].array, df[d + "longitude"].array
            )
            _, pos = kd_tree.query(np.array([X, Y]).T, k=1, workers=-1)
            df[d + "nearest_node"] = nodes.index[pos]
        df.to_parquet(os.path.join(foil_path, f"trip_data_{i}.parquet"))


def build_raw_od_matrix(foil_path, dates, graph_order):

    od_matrix = np.zeros((graph_order, graph_order))
    total_time = np.sum((dates["end"] - dates["start"]).dt.total_seconds())
    dates["month"] = dates["start"].dt.month
    for month, month_dates in dates.groupby("month"):
        df = pd.read_parquet(os.path.join(foil_path, f"trip_data_{month}.parquet"))
        min_end = np.minimum(
            df["dropoff_datetime"].to_numpy()[:, np.newaxis], month_dates["end"]
        )
        max_start = np.maximum(
            df["pickup_datetime"].to_numpy()[:, np.newaxis], month_dates["start"]
        )
        diff = (max_start - min_end) / np.timedelta64(1, 's')
        df["intersection_time"] = np.sum(np.maximum(diff, 0), axis=1)
        agg = df.groupby(["pickup_nearest_node", "dropoff_nearest_node"])[
            "intersection_time"
        ].sum()
        od_matrix[
            agg.index.get_level_values("pickup_nearest_node"),
            agg.index.get_level_values("dropoff_nearest_node"),
        ] += agg
    return od_matrix / total_time
