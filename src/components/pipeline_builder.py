import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# CUSTOM TRANSFORMERS
# =========================================================

class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors="ignore")


# =========================================================
# COLUMN DEFINITIONS
# =========================================================

DROP_COLS = ['name', 'link']

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
# TF-IDF COLUMN WRAPPER
# =========================================================

class TextColumnSelector(
    BaseEstimator,
    TransformerMixin
):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].astype(str)

        return pd.Series(X.ravel()).astype(str)


# =========================================================
# BUILD PIPELINE
# =========================================================

def build_pipeline():

    # -----------------------------------------------------
    # NUMERIC FEATURES
    # -----------------------------------------------------

    numeric_pipe = Pipeline([
        ("scaler", StandardScaler())
    ])

    # -----------------------------------------------------
    # EXACT CATEGORICAL FEATURES
    # -----------------------------------------------------

    onehot_pipe = Pipeline([
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ])

    # -----------------------------------------------------
    # BRAND TF-IDF
    # -----------------------------------------------------

    brand_pipe = Pipeline([

        (
            "selector",
            TextColumnSelector()
        ),

        (
            "tfidf",
            TfidfVectorizer(

                analyzer="char_wb",

                ngram_range=(2, 4),

                lowercase=True
            )
        )
    ])

    # -----------------------------------------------------
    # CITY TF-IDF
    # -----------------------------------------------------

    city_pipe = Pipeline([

        (
            "selector",
            TextColumnSelector()
        ),

        (
            "tfidf",
            TfidfVectorizer(

                analyzer="char_wb",

                ngram_range=(2, 4),

                lowercase=True
            )
        )
    ])

    # -----------------------------------------------------
    # COLUMN TRANSFORMER
    # -----------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",
                numeric_pipe,
                NUMERIC_COLS
            ),

            (
                "onehot",
                onehot_pipe,
                ONEHOT_COLS
            ),

            (
                "brand_text",
                brand_pipe,
                ["brand"]
            ),

            (
                "city_text",
                city_pipe,
                ["city"]
            )
        ],

        transformer_weights={

            # lower importance
            "numeric": 1.0,

            # medium importance
            "onehot": 2.0,

            # important semantic similarity
            "brand_text": 5.0,

            # medium semantic similarity
            "city_text": 2.0
        }
    )

    # -----------------------------------------------------
    # FINAL PIPELINE
    # -----------------------------------------------------

    pipeline = Pipeline([

        (
            "drop_cols",
            DropColumns(DROP_COLS)
        ),

        (
            "preprocessor",
            preprocessor
        )
    ])

    return pipeline