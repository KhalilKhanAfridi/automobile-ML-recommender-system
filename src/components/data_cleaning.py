import os
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


def clean_data(
    input_path,
    output_path,
    blocksize="64MB"
):
    """
    Clean automobile dataset using Dask
    """

    print("\n⚡ Loading data with Dask...")

    # =====================================================
    # LOAD DATA
    # =====================================================

    df = dd.read_csv(
        f"{input_path}/cars_data.csv",
        blocksize=blocksize,
        assume_missing=True
    )

    link = dd.read_csv(
        f"{input_path}/car_links.csv",
        header=None,
        names=["link"],
        blocksize=blocksize,
        assume_missing=True
    )

    print(f"✅ Partitions: {df.npartitions}")

    # =====================================================
    # RENAME COLUMNS
    # =====================================================

    df = df.rename(columns={
        "kolimeter": "kilometer",
        "feul_type": "fuel_type"
    })

    # =====================================================
    # RESET INDEX (IMPORTANT FOR ALIGNMENT)
    # =====================================================

    df = df.reset_index(drop=True)
    link = link.reset_index(drop=True)

    # =====================================================
    # ADD LINK COLUMN (SAFE - NO MERGE)
    # =====================================================

    df["link"] = link["link"]

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================
    
    df = df.drop_duplicates()

    # =====================================================
    # CLEAN PRICE
    # =====================================================

    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
    )

    # =====================================================
    # CLEAN KILOMETER
    # =====================================================

    df["kilometer"] = (
        df["kilometer"]
        .astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
    )

    # =====================================================
    # CLEAN ENGINE
    # =====================================================

    df["engine"] = (
        df["engine"]
        .astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
    )

    # =====================================================
    # CONVERT NUMERIC TYPES
    # =====================================================

    df["price"] = (
        dd.to_numeric(df["price"], errors="coerce") * 100000
    ).fillna(0).astype("int64")

    df["kilometer"] = (
        dd.to_numeric(df["kilometer"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )

    df["engine"] = (
        dd.to_numeric(df["engine"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )

    # =====================================================
    # CLEAN TEXT COLUMNS
    # =====================================================

    df["fuel_type"] = (
        df["fuel_type"]
        .astype(str)
        .str.lower()
        .str.replace("lp", "cng", regex=False)
        .str.strip()
    )

    df["transmission"] = (
        df["transmission"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["name"] = (
        df["name"]
        .astype(str)
        .str.replace("for sale", "", case=False, regex=False)
        .str.strip()
    )

    # =====================================================
    # BRAND COLUMN
    # =====================================================

    df["brand"] = (
        df["name"]
        .astype(str)
        .str.lower()
        .str.split()
        .str[0]
    )

    # =====================================================
    # SAFE STRING CASTING (FIXES YOUR ERROR)
    # =====================================================

    string_columns = [
        "name",
        "city",
        "model",
        "fuel_type",
        "transmission",
        "brand",
        "link"
    ]

    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # =====================================================
    # DROP NULLS
    # =====================================================

    df = df.dropna(subset=["price", "kilometer", "engine", "fuel_type", "transmission", "brand"])
    
    # =====================================================
    # CREATE OUTPUT DIRECTORY
    # =====================================================
    
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    # =====================================================
    # SAVE DATA
    # =====================================================

    print("\n💾 Saving cleaned data...")

    with ProgressBar():

 

        parquet_path = output_path.replace(".csv", ".parquet")

        df.to_parquet(
            parquet_path,
            write_index=False
        )
    
    print(f"\n✅ Cleaned CSV saved at: {output_path}")
    print(f"✅ Parquet saved at: {parquet_path}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    clean_data(
        input_path="data/raw",
        output_path="data/interim/cars_cleaned.csv"
    )