import json
from collections import defaultdict

from scuttle_bot.data.dataset import Dataset

ROLES = ["top", "jungle", "mid", "adc", "support"]
OUTPUT_PATH = "src/scuttle_bot/service/champion_roles.json"


def build_champion_role_frequencies(db_path: str = "src/scuttle_bot/cache/ml_dataset.db", output_path: str = OUTPUT_PATH) -> dict:
    """
    Counts, per championId, how many times it was picked in each of the 5
    role columns across every collected match. This is the empirical
    alternative to hand-maintaining a champion->role table: it's grounded in
    the exact population (high-elo NA solo queue) this bot cares about, and
    stays accurate for as long as the dataset is periodically refreshed.
    """
    dataset = Dataset(db_path=db_path)
    matches = dataset.retrieve_dataset()

    counts = defaultdict(lambda: {role: 0 for role in ROLES})
    for role in ROLES:
        for team in ("blue", "red"):
            for champ_id in matches[f"{team}_{role}"].dropna():
                counts[int(champ_id)][role] += 1

    with open(output_path, "w") as f:
        json.dump(counts, f, indent=2, sort_keys=True)

    print(f"Wrote role frequencies for {len(counts)} champions to {output_path}")
    return counts


if __name__ == "__main__":
    build_champion_role_frequencies()
