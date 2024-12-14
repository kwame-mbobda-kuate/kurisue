import os
import osmnx
import numpy as np
import scipy
from scipy.spatial import cKDTree
import pandas as pd
from pyproj import Transformer
from datetime import datetime


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


# géraud hands on scikit learn
# breiman the two cultures


def create_date(year, month):
    return np.datetime64(datetime(year, month, 1))


def correct_dates(dates):
    new_start_dates, new_end_dates = [], []
    year = dates["start"].dt.year[0]
    for k in range(dates["start"].size):
        start_date, end_date = dates["start"][k], dates["end"][k]
        start_month, end_month = dates["start"].dt.month[k], dates["end"].dt.month[k]
        if start_month != end_month:
            new_start_dates.append(start_date)
            new_end_date = create_date(year, start_month + 1) - np.timedelta64(1, "s")
            new_end_dates.append(new_end_date)
            for month in range(start_month + 1, end_month):
                new_start_date = create_date(year, month)
                new_end_date = create_date(year, month + 1) - np.timedelta64(1, "s")
                new_start_dates.append(new_start_date)
                new_end_dates.append(new_end_date)
            new_start_date = create_date(year, end_month)
            if new_start_date != end_date:
                new_start_dates.append(new_start_date)
                new_end_dates.append(end_date)
        else:
            new_start_dates.append(start_date)
            new_end_dates.append(end_date)
    return pd.DataFrame({"start": new_start_dates, "end": new_end_dates})


def build_raw_od_matrix(foil_path, dates, graph_order):
    od_matrix = scipy.sparse.coo_array((graph_order, graph_order))
    total_time = np.sum((dates["end"] - dates["start"]).dt.total_seconds())
    dates = correct_dates(dates)
    dates["month"] = dates["start"].dt.month
    for month, month_dates in dates.groupby("month"):
        # [a, b] ^ [c, d] = max(min(b, d) - max(a, c), 0)
        df = pd.read_parquet(os.path.join(foil_path, f"trip_data_{month}.parquet"))
        min_end = np.minimum(
            df["dropoff_datetime"].to_numpy(),
            month_dates["end"].to_numpy()[:, np.newaxis],
        )
        max_start = np.maximum(
            df["pickup_datetime"].to_numpy(),
            month_dates["start"].to_numpy()[:, np.newaxis],
        )
        diff = (min_end - max_start) / np.timedelta64(1, "s")
        df["intersection_time"] = np.sum(np.maximum(diff, 0), axis=0)
        agg = df.groupby(["pickup_nearest_node", "dropoff_nearest_node"])[
            "intersection_time"
        ].sum()
        rows = agg.index.get_level_values("pickup_nearest_node").values
        cols = agg.index.get_level_values("dropoff_nearest_node").values
        values = agg.values
        od_matrix += scipy.sparse.coo_array(
            (values, (rows, cols)), shape=(graph_order, graph_order), dtype=np.float64
        )
    return od_matrix / total_time
