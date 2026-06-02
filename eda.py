import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
print("=== SHAPES ===")
print("train:", train.shape, "test:", test.shape)

print("\n=== TRAIN dtypes / nulls ===")
print(train.dtypes)
print(train.isna().sum())

print("\n=== TEST dtypes / nulls ===")
print(test.dtypes)
print(test.isna().sum())

print("\n=== TARGET stats ===")
print(train["demand"].describe())
print("min:", train["demand"].min(), "max:", train["demand"].max())
print("quantiles:", train["demand"].quantile([0.01,0.05,0.5,0.95,0.99]).to_dict())

print("\n=== day distribution ===")
print("train day unique:", sorted(train["day"].unique()))
print("test day unique:", sorted(test["day"].unique()))

print("\n=== timestamp samples ===")
print("train ts unique count:", train["timestamp"].nunique())
print("test ts unique count:", test["timestamp"].nunique())
print("train ts first 10:", train["timestamp"].unique()[:10])
print("test ts first 10:", test["timestamp"].unique()[:10])

print("\n=== geohash ===")
print("train geohash unique:", train["geohash"].nunique())
print("test geohash unique:", test["geohash"].nunique())
print("train geohash length sample:", train["geohash"].str.len().value_counts())
both = set(train["geohash"]).intersection(set(test["geohash"]))
only_test = set(test["geohash"]) - set(train["geohash"])
print("geohash overlap train&test:", len(both))
print("geohash only in test:", len(only_test))

print("\n=== categorical value counts ===")
for c in ["RoadType","NumberofLanes","LargeVehicles","Landmarks","Weather"]:
    print("--", c)
    print(train[c].value_counts(dropna=False).head(15))

print("\n=== Temperature stats ===")
print(train["Temperature"].describe())
print(test["Temperature"].describe())

print("\n=== row count per (geohash, day, timestamp) ===")
g = train.groupby(["geohash","day","timestamp"]).size()
print(g.describe())
print("max group size:", g.max())

print("\n=== duplicates check ===")
print("train dup on (geohash,day,timestamp):", train.duplicated(subset=["geohash","day","timestamp"]).sum())
print("test dup on (geohash,day,timestamp):", test.duplicated(subset=["geohash","day","timestamp"]).sum())
