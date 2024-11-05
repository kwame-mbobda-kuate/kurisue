import os
import numpy
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
            # Bug : sometimes, X or Y contain nan or infinites
            _, pos = kd_tree.query(np.array([X, Y]).T, k=1, workers=-1)
            df[d + "_nearest_node"] = nodes.index[pos]
        df.to_parquet(os.path.join(foil_path, f"trip_data_{i}.parquet"))
