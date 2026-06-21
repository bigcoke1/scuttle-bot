import time
import pandas as pd

from scuttle_bot.service.schemas import Region, Queue
from scuttle_bot.data.collector import Collector
from scuttle_bot.data.processor import Processor
from scuttle_bot.infra.db_client import DatabaseClient
from scuttle_bot.service.utilities import get_champ_to_idx

class Dataset(DatabaseClient):
    def __init__(self, db_path: str):
        self.collector = Collector()
        self.processor = Processor()
        super().__init__(db_path, sql_script_path="src/scuttle_bot/infra/ml_schema.sql")

    def create_dataset(self, region = Region.NA, queue = Queue.RANKED_SOLO_5x5, sample_size: int = 300, num_matches_per_player: int = 3, max_errors_in_a_row: int = 5, batch_size: int = 10):
        challenger_leagues = self.collector.collect_challenger_leagues(region, queue) or {}
        master_leagues = self.collector.collect_master_leagues(region, queue) or {}
        grandmaster_leagues = self.collector.collect_grandmaster_leagues(region, queue) or {}
        random_players = self.collector.get_random_players(challenger_leagues | master_leagues | grandmaster_leagues, num_players=sample_size)

        total_matches_collected = 0
        batch = []
        seen_matches = self.get_seen_matches()
        errors_in_a_row = 0
        BATCH_SIZE = batch_size

        if random_players:
            for i, puuid in enumerate(random_players):
                print(f"Processing player {i+1} with PUUID: {puuid}")
                time.sleep(1)  # To avoid hitting rate limits
                match_history = self.collector.collect_match_history(puuid, count=num_matches_per_player) # list of match idss
                time.sleep(1)
                rank_json = self.collector.collect_ranked_stats(puuid) or {} # ranked stats for player
                if match_history is None:
                    continue
                for match_id in match_history:
                    try:
                        shortened_match_id = match_id[4:]  # Remove "NA1_" prefix
                        if shortened_match_id in seen_matches:
                            print(f"Skipping duplicate match ID: {match_id}")
                            continue
                        seen_matches.add(shortened_match_id)

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
                            self.insert_batch(batch, batch_size=BATCH_SIZE)
                            total_matches_collected += len(batch)
                            batch = []
                        errors_in_a_row = 0  # Reset error count after a successful match processing
                    except Exception as e:
                        print(f"\n\nError processing player with PUUID {puuid}: {e} \n Match ID: {match_id} \n\n")
                        errors_in_a_row += 1
                        if errors_in_a_row >= max_errors_in_a_row:
                            print("Too many errors in a row, stopping dataset creation.")
                            break
                        continue
        if batch:
            print(f"Inserting final batch of {len(batch)} records into the database...")
            self.insert_batch(batch, batch_size=BATCH_SIZE)
            total_matches_collected += len(batch)
        print(f"Dataset creation complete. Total matches collected: {total_matches_collected}")
    
    def get_seen_matches(self):
        return set(
            pd.read_sql_query(
                "SELECT match_id FROM matches",
                self.connection
            )["match_id"]
        )
    
    def insert_batch(self, batch: list, batch_size: int = 100):
        df = pd.DataFrame(batch)
        df.drop_duplicates(subset=["match_id"], inplace=True)  # Remove duplicates based on match_id
        df.to_sql("matches", self.connection, if_exists="append", index=False, method="multi", chunksize=batch_size)

    def retrieve_dataset(self) -> pd.DataFrame:
        query = "SELECT * FROM matches"
        return pd.read_sql_query(query, self.connection)
    
    def clean_dataset(self):
        self.execute_query("DELETE FROM matches")

if __name__ == "__main__":
    dataset = Dataset(db_path="src/scuttle_bot/cache/ml_dataset.db")
    dataset.create_dataset(region=Region.NA, queue=Queue.RANKED_SOLO_5x5, sample_size=300, num_matches_per_player=5, max_errors_in_a_row=5, batch_size=10)

                