import argparse
import logging

import pandas as pd

from scuttle_bot.data.dataset import Dataset
from scuttle_bot.utilities.utilities import get_id_to_idx

CHAMPION_COLUMNS = [
    "blue_top", "blue_jungle", "blue_mid", "blue_adc", "blue_support",
    "red_top", "red_jungle", "red_mid", "red_adc", "red_support",
    "blue_ban_0", "blue_ban_1", "blue_ban_2", "blue_ban_3", "blue_ban_4",
    "red_ban_0", "red_ban_1", "red_ban_2", "red_ban_3", "red_ban_4",
]

UNKNOWN_IDX = -1


def get_idx_to_id(version = None) -> dict:
    return {idx: champ_id for champ_id, idx in get_id_to_idx(version).items()}


def convert_idx_to_id(value, idx_to_id: dict):
    if pd.isna(value):
        return None

    idx = int(value)
    if idx == UNKNOWN_IDX:
        return UNKNOWN_IDX

    if idx not in idx_to_id:
        # Either a stale index (roster has shifted since insertion) or a row that
        # already stores a raw championId. Leave it untouched rather than guess.
        logging.warning(f"No champion id found for index {idx}; leaving value unchanged")
        return idx

    return idx_to_id[idx]


def migrate(db_path: str = "src/scuttle_bot/cache/ml_dataset.db"):
    dataset = Dataset(db_path=db_path)
    idx_to_id = get_idx_to_id(version = "16.12.1")

    df = dataset.retrieve_dataset()
    if df.empty:
        print("No rows found in matches table; nothing to migrate.")
        return

    updates = []
    for _, row in df.iterrows():
        new_values = [convert_idx_to_id(row[col], idx_to_id) for col in CHAMPION_COLUMNS]
        updates.append((*new_values, row["match_id"]))

    set_clause = ", ".join(f"{col} = ?" for col in CHAMPION_COLUMNS)
    query = f"UPDATE matches SET {set_clause} WHERE match_id = ?"

    with dataset.connection:
        dataset.cursor.executemany(query, updates)

    print(f"Migrated {len(updates)} rows: champion indices -> champion ids.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-time migration: convert champion index columns in ml_dataset.db to stable Riot championIds."
    )
    parser.add_argument("--db-path", default="src/scuttle_bot/cache/ml_dataset.db")
    args = parser.parse_args()
    migrate(args.db_path)
