

from src.components.pipeline_builder import build_pipeline
from scipy import sparse
import os
import joblib


ARTIFACT_DIR = "data/processed"

PIPELINE_PATH = f"{ARTIFACT_DIR}/pipeline.joblib"
MATRIX_PATH   = f"{ARTIFACT_DIR}/feature_matrix.npz"
DATA_PATH     = f"{ARTIFACT_DIR}/cars_dataframe.parquet"
# =========================================================
# COLUMN DEFINITIONS
# =========================================================

DROP_COLS = ['name']

NUMERIC_COLS = [
    'price',
    'kilometer',
    'engine',
    'model'
]

# EXACT CATEGORIES
ONEHOT_COLS = [
    'fuel_type',
    'transmission'
]

# SEMANTIC CATEGORIES
TEXT_SIMILARITY_COLS = [
    'brand',
    'city'
]

ALL_FEATURE_COLS = (
    NUMERIC_COLS +
    ONEHOT_COLS +
    TEXT_SIMILARITY_COLS
)
# =========================================================
# STAGE 1 — OFFLINE TRAINING WITH DASK
# =========================================================

def train_and_save_pipeline(
    filepath,
    file_type="csv",
    blocksize="64MB"
):
    """
    RUN THIS ONLY ONCE

    This:
        1. Loads huge data with Dask
        2. Fits sklearn pipeline
        3. Generates feature matrix
        4. Compresses + saves matrix
        5. Saves fitted pipeline
        6. Saves dataframe
    """

    print("\n⚡ LOADING DATA WITH DASK")

    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar

    loaders = {
        "csv": lambda: dd.read_csv(filepath, blocksize=blocksize),
        "parquet": lambda: dd.read_parquet(filepath),
        "json": lambda: dd.read_json(filepath)
    }

    if file_type not in loaders:
        raise ValueError("Unsupported file type")

    df_dask = loaders[file_type]()

    print(f"Partitions: {df_dask.npartitions}")

    required_cols = DROP_COLS + ALL_FEATURE_COLS

    print("\n⚙️ COMPUTING REQUIRED COLUMNS")

    with ProgressBar():
        df = df_dask[required_cols].compute()

    print(f"\n✅ Rows Loaded: {len(df):,}")

    # -----------------------------------------------------
    # FIT PIPELINE
    # -----------------------------------------------------

    print("\n⚙️ FITTING PIPELINE")

    pipeline = build_pipeline()

    X = pipeline.fit_transform(df)
    print(X.shape)
    print("✅ Pipeline Fitted")

    # -----------------------------------------------------
    # CONVERT TO SPARSE MATRIX
    # -----------------------------------------------------

    print("\n⚙️ CONVERTING TO SPARSE MATRIX")

    X_sparse = sparse.csr_matrix(X)
   
    print("✅ Sparse Matrix Created")

    # -----------------------------------------------------
    # SAVE EVERYTHING
    # -----------------------------------------------------

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("\n💾 SAVING PIPELINE")

    joblib.dump(
        pipeline,
        PIPELINE_PATH,
        compress=("gzip", 3)
    )

    print("✅ Pipeline Saved")

    print("\n💾 SAVING COMPRESSED MATRIX")

    sparse.save_npz(
        MATRIX_PATH,
        X_sparse,
        compressed=True
    )

    print("✅ Matrix Saved")

    print("\n💾 SAVING DATAFRAME")

    df.to_parquet(DATA_PATH, index=False)

    print("✅ DataFrame Saved")

    print("\n🎉 TRAINING COMPLETE")

train_and_save_pipeline("data/interim/cars_cleaned.csv")    