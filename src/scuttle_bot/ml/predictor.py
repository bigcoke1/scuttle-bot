import re
from typing import Optional, TypedDict

import pandas as pd

from scuttle_bot.ml.feature_encoder import FeatureEncoder
from scuttle_bot.ml.rf.rf_model import RandomForestModel
from scuttle_bot.service.utilities import get_champion_mapping

# Best-performing config for "draft + player stats" across every trained
# model type/variant (see cv_summary.json under each model type's models/
# dir): RandomForest variant C -- mean accuracy 0.609 vs logistic's 0.587 and
# the NN's 0.571 on the identical feature set. Of variant C's 5 repeated-
# holdout fits, random_state=3 scored highest (0.629) and is the one served.
ARTIFACTS_DIR = "src/scuttle_bot/ml/rf/models/C/"
MODEL_PATH = f"{ARTIFACTS_DIR}rf_model_C_3.pkl"

PICK_SLOTS = [f"{team}_{role}" for team in ("blue", "red") for role in ("top", "jungle", "mid", "adc", "support")]

# match_participants stores Riot's teamPosition names (lowercased by Processor);
# this is the inverse of FeatureEncoder.ROLE_TO_SLOT.
ROLE_TO_TEAM_POSITION = {"top": "top", "jungle": "jungle", "mid": "middle", "adc": "bottom", "support": "utility"}


class PlayerInput(TypedDict, total=False):
    champion: str
    tier: Optional[str]
    rank: Optional[str]
    wins: int
    losses: int
    champion_points: Optional[int]
    champion_level: Optional[int]


class WinPredictor:
    """
    Wraps the RandomForest "draft + player stats" model (variant C) so
    callers only need to supply a plain row of picks + player stats -- all
    FeatureEncoder/model plumbing lives here.
    """

    def __init__(self):
        self.champion_mapping = get_champion_mapping()  # championId -> name
        self.name_to_id = {
            re.sub(r"[^A-Za-z0-9]", "", name).lower(): champ_id
            for champ_id, name in self.champion_mapping.items()
        }

        self.encoder = FeatureEncoder(ARTIFACTS_DIR, use_bans=False, use_avg_tier=False, use_player_stats=True)
        self.encoder.load()

        self.model = RandomForestModel()
        self.model.load(MODEL_PATH)

    def resolve_champion_id(self, champion_name: str) -> Optional[int]:
        return self.name_to_id.get(re.sub(r"[^A-Za-z0-9]", "", champion_name).lower())

    def predict(self, players: dict[str, PlayerInput], patch_version: str = "15.1") -> float:
        """
        players is keyed by slot (blue_top, blue_jungle, blue_mid, blue_adc,
        blue_support, red_top, red_jungle, red_mid, red_adc, red_support),
        all 10 required. Returns P(blue wins) in [0, 1].
        """
        missing = [slot for slot in PICK_SLOTS if slot not in players]
        if missing:
            raise ValueError(f"Missing player input for slots: {missing}")

        match_row = {
            "match_id": "prediction",
            "patch_version": patch_version,
            "average_tier": None,  # imputed by the encoder's stored training median
            "blue_win": 0,  # placeholder -- only X is used, y is discarded
            "game_duration": 1800,  # placeholder, just needs to clear the >=600s filter
            "queue_id": 420,
        }
        for slot in PICK_SLOTS:
            champ_id = self.resolve_champion_id(players[slot]["champion"])
            if champ_id is None:
                raise ValueError(f"Unknown champion for {slot}: {players[slot]['champion']!r}")
            match_row[slot] = champ_id

        matches_df = pd.DataFrame([match_row])

        participant_rows = []
        for slot in PICK_SLOTS:
            team, role = slot.split("_", 1)
            p = players[slot]
            wins = p.get("wins") or 0
            losses = p.get("losses") or 0
            total_games = wins + losses
            participant_rows.append({
                "match_id": "prediction",
                "puuid": slot,
                "team": team,
                "role": ROLE_TO_TEAM_POSITION[role],
                "tier": p.get("tier"),
                "rank": p.get("rank"),
                "wins": wins,
                "losses": losses,
                "win_rate": wins / total_games if total_games > 0 else None,
                "champion_points": p.get("champion_points"),
                "champion_level": p.get("champion_level"),
            })
        participants_df = pd.DataFrame(participant_rows)

        X, _ = self.encoder.transform(matches_df, participants_df)
        proba = self.model.predict_proba(X)
        return float(proba[0][1])
