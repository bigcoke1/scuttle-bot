"""
Entrypoint for the scheduled cloud collection job. Pulls the latest dataset
from S3, samples players and collects their recent ranked matches, and
leaves the updated ml_dataset.db in place for aws_client.backup_databases_to_s3
to push back up.
"""

from dotenv import load_dotenv

from scuttle_bot.data.dataset import Dataset
from scuttle_bot.service.schemas import Region, Queue

DB_PATH = "src/scuttle_bot/cache/ml_dataset.db"

COLLECTION_CONFIG = {
    "region": Region.NA,
    "queue": Queue.RANKED_SOLO_5x5,
    "sample_size": 300,
    "num_matches_per_player": 5,
    "max_errors_in_a_row": 5,
    "batch_size": 10,
    "challenger_league": True,
    "master_league": True,
    "grandmaster_league": True,
    "stratified_sampling": True,
}


def main():
    load_dotenv()
    dataset = Dataset(db_path=DB_PATH)
    dataset.create_dataset(**COLLECTION_CONFIG)


if __name__ == "__main__":
    main()
