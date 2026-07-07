import pandas as pd
import numpy as np

def load_raw(path:str) -> pd.DataFrames:
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], format = "%d-%m-%Y", errors="coerce")

    for col in ["position", "popularity", "duration_ms", "total_tracks"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["song"] = df["song"].astype(str).str.strip()
    df["artist"] = df["artist"].astype(str).str.strip()

    df["is_explicit"] = (
        df["is_explicit"].astype(str).str.upper().map({"TRUE":True, "FALSE":False})
    )

    df = df[(df["position"] >= 1) & (df["position"] <= 50)]
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["date", "position"], keep="first")
    
    df["duration_min"] = (df["duration_ms"] / 60000).round(2)
    df["song_key"] = df["song"] + " - " + df["artist"]

    df = df.sort_values(["date","position"]).reset_index(drop = True)
    return df

def get_clean_data(path:str) -> pd.DataFrame:
    return clean_data(load_raw(path))

def data_quality_report(df: pd.DataFrame) -> dict:
    return{
        "total_rows":len(df),
        "date_min":df["date"].min().date(),
        "date_max":df["date"].max().date(),
        "unique_days":df["date"].nunique(),
        "unique_songs":df["song_key"].nunique(),
        "unique_artist":df["artist"].nunique(),
        "missing_popularity":int(df["popularity"].isna().sum()),

    }

if __name__ == "__main__":
    df = get_clean_data(r"C:\Users\Lenovo\Downloads\Songsdata.csv")
    print(df.head())
    print()
    print(data_quality_report(df))

