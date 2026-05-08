import time
import pandas as pd

from scuttle_bot.service.schemas import Region, Queue
from scuttle_bot.data.collector import Collector
from scuttle_bot.data.processor import Processor
from scuttle_bot.infra.db_client import DatabaseClient

class Dataset(DatabaseClient):
    def __init__(self, db_path: str):
        self.collector = Collector()
        self.processor = Processor()
        super().__init__(db_path, sql_script_path="src/scuttle_bot/infra/ml_schema.sql")

    def create_dataset(self, region = Region.NA, queue = Queue.RANKED_SOLO_5x5, sample_size: int = 300):
        challenger_leagues = self.collector.collect_challenger_leagues(region, queue)
        random_players = self.collector.get_random_players(challenger_leagues, num_players=sample_size)

        batch = []
        seen_matches = set(
            pd.read_sql_query(
                "SELECT match_id FROM matches",
                self.connection
            )["match_id"]
        )
        errors_in_a_row = 0
        BATCH_SIZE = 100

        if random_players:
            for puuid in random_players:
                try:
                    print(f"Processing player with PUUID: {puuid}")
                    time.sleep(1)  # To avoid hitting rate limits
                    match_history = self.collector.collect_match_history(puuid) # list of match ids
                    time.sleep(1)
                    rank_json = self.collector.collect_ranked_stats(puuid) or {} # ranked stats for player
                    if match_history is None:
                        continue
                    for match_id in match_history:

                        if match_id in seen_matches:
                            print(f"Skipping duplicate match ID: {match_id}")
                            continue
                        seen_matches.add(match_id)

                        time.sleep(2)  # To avoid hitting rate limits
                        match_json = self.collector.collect_match_details(match_id)
                        if match_json is None:
                            continue

                        processed_data = self.processor.process_data(match_json, rank_json)
                        if processed_data is None:
                            continue

                        batch.append(processed_data)
                        print(f"Added match ID {match_id} to batch. Current batch size: {len(batch)}")
                        if len(batch) >= BATCH_SIZE:
                            print(f"Inserting batch of {len(batch)} records into the database...")
                            df = pd.DataFrame(batch)
                            df.drop_duplicates(subset=["match_id"], inplace=True)  # Remove duplicates based on match_id
                            df.to_sql("matches", self.connection, if_exists="append", index=False, method="multi", chunksize=BATCH_SIZE)
                            batch = []
                        errors_in_a_row = 0  # Reset error count after a successful match processing
                except Exception as e:
                    print(f"Error processing player with PUUID {puuid}: {e}")
                    errors_in_a_row += 1
                    if errors_in_a_row >= 5:
                        print("Too many errors in a row, stopping dataset creation.")
                        break
                    continue
        if batch:
            print(f"Inserting final batch of {len(batch)} records into the database...")
            df = pd.DataFrame(batch)
            df.drop_duplicates(subset=["match_id"], inplace=True)  # Remove duplicates based on match_id
            df.to_sql("matches", self.connection, if_exists="append", index=False, method="multi", chunksize=BATCH_SIZE)

if __name__ == "__main__":
    dataset = Dataset(db_path="src/scuttle_bot/cache/ml_dataset.db")
    dataset.create_dataset(sample_size=300)
                