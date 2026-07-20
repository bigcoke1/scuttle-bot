import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack

from scuttle_bot.service.utilities import get_id_to_idx

CHAMPION_COLUMNS = [
    "blue_top", "blue_jungle", "blue_mid", "blue_adc", "blue_support",
    "red_top", "red_jungle", "red_mid", "red_adc", "red_support",
    "blue_ban_0", "blue_ban_1", "blue_ban_2", "blue_ban_3", "blue_ban_4",
    "red_ban_0", "red_ban_1", "red_ban_2", "red_ban_3", "red_ban_4",
]

CATEGORICAL_COLUMNS = CHAMPION_COLUMNS + ["patch_version", "queue_id"]

UNKNOWN_IDX = -1  # matches the "Unknown" sentinel used by get_champ_to_idx

# match_participants roles use Riot's teamPosition names; matches columns use lane names
ROLE_TO_SLOT = {"top": "top", "jungle": "jungle", "middle": "mid", "bottom": "adc", "utility": "support"}
SLOTS = [f"{team}_{role}" for team in ("blue", "red") for role in ("top", "jungle", "mid", "adc", "support")]
PARTICIPANT_STATS = ["rank_score", "win_rate", "games", "champion_points", "champion_level"]
PARTICIPANT_COLUMNS = [f"{slot}_{stat}" for slot in SLOTS for stat in PARTICIPANT_STATS]

# Same mmr-like scale as Processor.process_ranked_stats, so a missing player's
# rank_score can fall back to the match's average_tier.
TIER_VALUES = {
    "IRON": 0, "BRONZE": 1, "SILVER": 2, "GOLD": 3, "PLATINUM": 4,
    "EMERALD": 5, "DIAMOND": 6, "MASTER": 7, "GRANDMASTER": 8, "CHALLENGER": 9,
}
DIVISION_VALUES = {"IV": 0, "III": 1, "II": 2, "I": 3}


class FeatureEncoder:
    def __init__(self, encoder_path: str):
        self.encoder_path = encoder_path
        self.encoder = None
        self.scaler = None

    def join_participants(self, matches_df: pd.DataFrame, participants_df: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
        """Pivot match_participants (10 rows per match) into one wide row per
        match and join it onto matches. The default inner join keeps only
        matches that have participant data, since the backfill may still be
        in progress."""
        p = participants_df.copy()
        p["slot"] = p["team"] + "_" + p["role"].map(ROLE_TO_SLOT)
        p["rank_score"] = p["tier"].map(TIER_VALUES) * 4 + p["rank"].map(DIVISION_VALUES)
        p["games"] = p["wins"] + p["losses"]

        wide = p.pivot_table(index="match_id", columns="slot", values=PARTICIPANT_STATS)
        wide.columns = [f"{slot}_{stat}" for stat, slot in wide.columns]
        wide = wide.reset_index()

        matches_df = matches_df.copy()
        matches_df["match_id"] = matches_df["match_id"].astype(str)
        wide["match_id"] = wide["match_id"].astype(str)

        df = matches_df.merge(wide, on="match_id", how=how)

        for col in PARTICIPANT_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fill gaps from unranked players or missing mastery data
        for slot in SLOTS:
            df[f"{slot}_rank_score"] = df[f"{slot}_rank_score"].fillna(df["average_tier"])
            df[f"{slot}_win_rate"] = df[f"{slot}_win_rate"].fillna(0.5)
            df[f"{slot}_games"] = df[f"{slot}_games"].fillna(0)
            df[f"{slot}_champion_points"] = df[f"{slot}_champion_points"].fillna(0)
            df[f"{slot}_champion_level"] = df[f"{slot}_champion_level"].fillna(0)

        return df

    def _convert_champion_ids_to_idx(self, df: pd.DataFrame) -> pd.DataFrame:
        # The database stores stable Riot championIds, but the model is trained on
        # indices derived from the *current* champion roster. Resolving id -> idx
        # here (instead of at ingestion time) means newly released champions get
        # picked up automatically instead of requiring a data migration.
        id_to_idx = get_id_to_idx()
        for col in CHAMPION_COLUMNS:
            df[col] = df[col].apply(
                lambda champ_id: id_to_idx.get(int(champ_id), UNKNOWN_IDX) if pd.notna(champ_id) else UNKNOWN_IDX
            )
        return df

    def _prepare(self, df: pd.DataFrame, participants_df: pd.DataFrame = None) -> pd.DataFrame:
        df = df.copy()
        if participants_df is not None:
            df = self.join_participants(df, participants_df)

        df = self._convert_champion_ids_to_idx(df)

        df["patch_version"] = df["patch_version"].apply(
            lambda x: ".".join(str(x).split(".")[:2])
        )

        df = df[df["game_duration"] >= 600]
        return df

    def _numerical_columns(self, df: pd.DataFrame) -> list:
        return ["average_tier"] + [col for col in PARTICIPANT_COLUMNS if col in df.columns]

    def fit_transform(self, df: pd.DataFrame, participants_df: pd.DataFrame = None):
        df = self._prepare(df, participants_df)

        # Fit encoder
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        X_cat = self.encoder.fit_transform(df[CATEGORICAL_COLUMNS])

        # Fit scaler
        self.scaler = StandardScaler()
        X_num = self.scaler.fit_transform(df[self._numerical_columns(df)])

        X = hstack([X_num, X_cat])
        y = df["blue_win"]

        self.save()

        return X, y

    def transform(self, df: pd.DataFrame, participants_df: pd.DataFrame = None):
        df = self._prepare(df, participants_df)

        if self.encoder is None or self.scaler is None:
            self.load()

        if self.encoder is not None and self.scaler is not None:
            X_cat = self.encoder.transform(df[CATEGORICAL_COLUMNS])
            X_num = self.scaler.transform(df[self._numerical_columns(df)])

            X = hstack([X_num, X_cat])
            y = df["blue_win"]
            return X, y
        else:
            raise ValueError("Encoder and scaler must be fitted or loaded before transforming data.")

    def save(self):
        joblib.dump(self.encoder, f"{self.encoder_path}encoder.pkl")
        joblib.dump(self.scaler, f"{self.encoder_path}scaler.pkl")

    def load(self):
        self.encoder = joblib.load(f"{self.encoder_path}_encoder.pkl")
        self.scaler = joblib.load(f"{self.encoder_path}_scaler.pkl")
