import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack


class FeatureEncoder:
    def __init__(self, encoder_path: str):
        self.encoder_path = encoder_path
        self.encoder = None
        self.scaler = None

    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()

        df["patch_version"] = df["patch_version"].apply(
            lambda x: ".".join(str(x).split(".")[:2])
        )

        categorical_cols = [
            "blue_top", "blue_jungle", "blue_mid", "blue_adc", "blue_support",
            "red_top", "red_jungle", "red_mid", "red_adc", "red_support",
            "blue_ban_0", "blue_ban_1", "blue_ban_2", "blue_ban_3", "blue_ban_4",
            "red_ban_0", "red_ban_1", "red_ban_2", "red_ban_3", "red_ban_4",
            "patch_version", "queue_id"
        ]

        numerical_cols = ["average_tier"]

        # Fit encoder
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        X_cat = self.encoder.fit_transform(df[categorical_cols])

        # Fit scaler
        self.scaler = StandardScaler()
        X_num = self.scaler.fit_transform(df[numerical_cols])

        X = hstack([X_num, X_cat])
        y = df["blue_win"]

        self.save()

        return X, y

    def transform(self, df: pd.DataFrame):
        df = df.copy()

        df["patch_version"] = df["patch_version"].apply(
            lambda x: ".".join(str(x).split(".")[:2])
        )

        categorical_cols = [
            "blue_top", "blue_jungle", "blue_mid", "blue_adc", "blue_support",
            "red_top", "red_jungle", "red_mid", "red_adc", "red_support",
            "blue_ban_0", "blue_ban_1", "blue_ban_2", "blue_ban_3", "blue_ban_4",
            "red_ban_0", "red_ban_1", "red_ban_2", "red_ban_3", "red_ban_4",
            "patch_version", "queue_id"
        ]

        numerical_cols = ["average_tier"]

        if self.encoder is None or self.scaler is None:
            self.load()

        if self.encoder is not None and self.scaler is not None:
            X_cat = self.encoder.transform(df[categorical_cols])
            X_num = self.scaler.transform(df[numerical_cols])

            X = hstack([X_num, X_cat])
            return 
        else:
            raise ValueError("Encoder and scaler must be fitted or loaded before transforming data.")

    def save(self):
        joblib.dump(self.encoder, f"{self.encoder_path}encoder.pkl")
        joblib.dump(self.scaler, f"{self.encoder_path}scaler.pkl")

    def load(self):
        self.encoder = joblib.load(f"{self.encoder_path}_encoder.pkl")
        self.scaler = joblib.load(f"{self.encoder_path}_scaler.pkl")