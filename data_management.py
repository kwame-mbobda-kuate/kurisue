import requests
import os
import subprocess
from tqdm import tqdm
import numpy
import osmnx
import numpy as np
import itertools
import pandas as pd
from multiprocessing import Pool

MILE_IN_METER = 1609.344
MAX_SPEED = 55 * MILE_IN_METER
# Bounding box of New York
MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = -74.2549337, 40.4983853, -73.7004728, 40.912507

join = os.path.join


def download_url_with_bar(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    print(os.getcwd())
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download_datasets(data_path: str):
    os.makedirs(data_path, exist_ok=True)
    os.chdir(data_path)
    foil_path = "FOIL2013"
    foil_zip_path = join(foil_path, "FOIL2013.7z")
    os.makedirs(foil_path, exist_ok=True)
    download_url_with_bar(
        "https://archive.org/download/nycTaxiTripData2013/trip_data.7z", foil_zip_path
    )
    os.chdir("FOIL2013")
    subprocess.run(["7z", "x", "FOIL2013.7z"])

    os.remove("FOIL2013.7z")
    os.chdir("..")

    download_url_with_bar(
        "https://data.cityofnewyork.us/api/views/7ym2-wayt/rows.csv",
        "automated_traffic_volume_counts.csv",
    )

    os.chdir("..")


def clean_thread_job(args):
    foil_path, i = args
    df = pd.read_csv(os.path.join(foil_path, f"trip_data_{i}.csv"))
    os.remove(os.path.join(foil_path, f"trip_data_{i}.csv"))
    columns_to_drop = ["medallion", "hack_license", "vendor_id", "rate_code", "store_and_fwd_flag"]
    for column_name in columns_to_drop:
        if column_name in df.columns:
            df.drop(columns=[column_name], axis=1, inplace=True)

    df.rename(columns={x: x.strip() for x in df.columns}, inplace=True)

    mask_long_max = df["pickup_longitude"] <= MAX_LON
    mask_long_min = df["pickup_longitude"] >= MIN_LON
    mask_lat_max = df["pickup_latitude"] <= MAX_LAT
    mask_lat_min = df["pickup_latitude"] >= MIN_LAT

    df = df[mask_lat_max & mask_lat_min & mask_long_max & mask_long_min]

    mask_long_max = df["dropoff_longitude"] <= MAX_LON
    mask_long_min = df["dropoff_longitude"] >= MIN_LON
    mask_lat_max = df["dropoff_latitude"] <= MAX_LAT
    mask_lat_min = df["dropoff_latitude"] >= MIN_LAT

    df = df[mask_lat_max & mask_lat_min & mask_long_max & mask_long_min]
    # The trip must start and end in New York

    df = df[df["passenger_count"] <= 5]  # A taxi cannot accept more than 5 passengers

    df["trip_distance"] = df["trip_distance"] * MILE_IN_METER
    df = df[
        df["trip_distance"] / df["trip_time_in_secs"] <= MAX_SPEED
    ]  # The averegae speed  must be less than the maximum speed

    df = df[
        df["trip_time_in_secs"] <= 12 * 60 * 60
    ]  # A trip cannot last more than 12 hours
    df = df[
        df["trip_distance"]
        >= osmnx.distance.euclidean(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )
    ]

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

    duration = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()
    mask_chrono = duration > 0
    mask_diff = np.abs(duration - df["trip_time_in_secs"]) <= 60
    df = df[mask_chrono & mask_diff]
    # The difference between the trip time displayed and the one computed must be less than a minute
    df = df.dropna()

    df.to_parquet(os.path.join(foil_path, f"trip_data_{i}.parquet"))


def clean_datasets(foil_path: str):

    with Pool(12) as p:
        p.map(clean_thread_job, itertools.product([foil_path], range(1, 13)))


def date_traffic(df): #put date into datetime type for traffic files
    df = df.rename(columns={'Yr': 'year', 'M': 'month', 'D': 'day', 'HH': 'hour', 'MM': 'minute'})
    df['datetime'] = pandas.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
    return df

def date_taxis(df): #put date into datetime for taxis files
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day'] = df['pickup_datetime'].dt.day_name()
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_date'] = df['pickup_datetime'].dt.date

    return df