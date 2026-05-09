import os
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


def clean_data(
    input_path,
    output_path,
    blocksize="64MB"
):
    """
    Clean car dataset using Dask.

    Parameters
    ----------
    input_path : str
        Path of raw CSV file

    output_path : str
        Path where cleaned CSV will be saved

    blocksize : str
        Dask partition size
    """

    print("\n⚡ Loading data with Dask...")

    # =====================================================
    # LOAD DATA
    # =====================================================

    df = dd.read_csv(
        input_path,
        blocksize=blocksize,
        assume_missing=True
    )

    print(f"✅ Partitions: {df.npartitions}")

    # =====================================================
    # RENAME COLUMNS
    # =====================================================

    df = df.rename(columns={
        'kolimeter': 'kilometer',
        'feul_type': 'fuel_type'
    })

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    df = df.drop_duplicates()

    # =====================================================
    # CLEAN PRICE
    # =====================================================

    df['price'] = (
        df['price']
        .astype(str)
        .str.replace(
            r'[^0-9.]',
            '',
            regex=True
        )
    )

    # =====================================================
    # CLEAN KILOMETER
    # =====================================================

    df['kilometer'] = (
        df['kilometer']
        .astype(str)
        .str.replace(
            r'[^0-9]',
            '',
            regex=True
        )
    )

    # =====================================================
    # CLEAN ENGINE
    # =====================================================

    df['engine'] = (
        df['engine']
        .astype(str)
        .str.replace(
            r'[^0-9]',
            '',
            regex=True
        )
    )

    # =====================================================
    # CONVERT TYPES
    # =====================================================

    df['price'] = (
        dd.to_numeric(
            df['price'],
            errors='coerce'
        ) * 100000
    )

    df['kilometer'] = dd.to_numeric(
        df['kilometer'],
        errors='coerce'
    )

    df['engine'] = dd.to_numeric(
        df['engine'],
        errors='coerce'
    )

    # =====================================================
    # CLEAN FUEL TYPE
    # =====================================================

    df["fuel_type"] = (
        df["fuel_type"]
        .astype(str)
        .str.lower()
        .str.replace(
            "lp",
            "cng",
            regex=False
        )
    )

    # =====================================================
    # CLEAN TRANSMISSION
    # =====================================================

    df["transmission"] = (
        df["transmission"]
        .astype(str)
        .str.lower()
    )

    # =====================================================
    # CREATE BRAND COLUMN
    # =====================================================

    df['brand'] = (
        df['name']
        .astype(str)
        .str.lower()
        .str.split()
        .str[0]
    )

    # =====================================================
    # CLEAN NAME
    # =====================================================

    df["name"] = (
        df["name"]
        .astype(str)
        .str.replace(
            "for Sale",
            "",
            case=False,
            regex=False
        )
        .str.strip()
    )

    # =====================================================
    # DROP NULLS
    # =====================================================

    df = df.dropna()

    # =====================================================
    # CREATE OUTPUT DIRECTORY
    # =====================================================

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    # =====================================================
    # SAVE CLEANED DATA
    # =====================================================

    print("\n💾 Saving cleaned data...")

    with ProgressBar():

        df.to_csv(
            output_path,
            single_file=True,
            index=False
        )

    print(f"\n✅ Cleaned data saved at: {output_path}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    clean_data(
        input_path=r"data/raw/cars_data.csv",
        output_path=r"data/interim/cars_cleaned.csv"
    )