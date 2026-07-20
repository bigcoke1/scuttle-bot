import time
from pathlib import Path
from typing import Optional

import pandas as pd

from scuttle_bot.service.schemas import Region, Queue
from scuttle_bot.data.collector import Collector
from scuttle_bot.data.processor import Processor
from scuttle_bot.infra.db_client import DatabaseClient
from scuttle_bot.service.utilities import get_champ_to_idx

MATCH_PARTICIPANTS_SCHEMA_PATH = "src/scuttle_bot/infra/match_participants_schema.sql"

class Dataset(DatabaseClient):
    def __init__(self, db_path: str):
        self.collector = Collector()
        self.processor = Processor(self.collector)
        super().__init__(db_path, sql_script_path="src/scuttle_bot/infra/ml_schema.sql")
        self._ensure_match_participants_table()

    def _ensure_match_participants_table(self):
        # match_participants is a separate schema file, so it needs its own
        # CREATE TABLE IF NOT EXISTS pass regardless of whether ml_dataset.db
        # already existed when DatabaseClient._initialize_db ran.
        schema = Path(MATCH_PARTICIPANTS_SCHEMA_PATH).read_text()
        self.connection.executescript(schema)
        self.connection.commit()

    def create_dataset(self, region = Region.NA, queue = Queue.RANKED_SOLO_5x5, sample_size: int = 300, num_matches_per_player: int = 3, max_errors_in_a_row: int = 5, 
                       batch_size: int = 10, challenger_league: bool = False, master_league: bool = True, grandmaster_league: bool = False, stratified_sampling: bool = True):
        challenger_leagues = self.collector.collect_challenger_leagues(region, queue) or {}
        master_leagues = self.collector.collect_master_leagues(region, queue) or {}
        grandmaster_leagues = self.collector.collect_grandmaster_leagues(region, queue) or {}

        grouped_players = []
        if challenger_league:
            grouped_players.append(challenger_leagues.get("entries", []))
        if master_league:
            grouped_players.append(master_leagues.get("entries", []))
        if grandmaster_league:
            grouped_players.append(grandmaster_leagues.get("entries", []))

        if stratified_sampling:
            random_players = self.collector.get_stratified_random_players(
                grouped_players=grouped_players,
                num_players=sample_size
            )
        else:
            random_players = self.collector.get_random_players([player for group in grouped_players for player in group], num_players=sample_size)

        total_matches_collected = 0
        batch = []
        participant_batch = []
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
                        participant_batch.extend(self.processor.process_participants(match_json))
                        print(f"Added match ID {match_id} to batch. Current batch size: {len(batch)}")
                        if len(batch) >= BATCH_SIZE:
                            print(f"Inserting batch of {len(batch)} records into the database...")
                            self.insert_batch(batch, batch_size=BATCH_SIZE)
                            self.insert_participant_batch(participant_batch)
                            total_matches_collected += len(batch)
                            batch = []
                            participant_batch = []
                        errors_in_a_row = 0  # Reset error count after a successful match processing
                    except Exception as e:
                        print(f"\n\nError processing player with PUUID {puuid}: {e} \n Match ID: {match_id} \n\n")
                        errors_in_a_row += 1
                        if errors_in_a_row >= max_errors_in_a_row:
                            print("Too many errors in a row, stopping dataset creation.")
                            break
                        continue
        else:
            print("No players found for the specified leagues and region.")
        if batch:
            print(f"Inserting final batch of {len(batch)} records into the database...")
            self.insert_batch(batch, batch_size=BATCH_SIZE)
            self.insert_participant_batch(participant_batch)
            total_matches_collected += len(batch)
        print(f"Dataset creation complete. Total matches collected: {total_matches_collected}")
    
    def backfill_participants(self, region_prefix: str = "NA1", limit: Optional[int] = None, max_errors_in_a_row: int = 5, batch_size: int = 10):
        """Collect participant info for matches already in the matches table that
        have no rows in match_participants yet. match_ids are stored as bare
        gameIds, so the region prefix is re-added to query the match-v5 API."""
        pending = [
            row[0] for row in self.execute_query(
                """
                SELECT m.match_id FROM matches m
                LEFT JOIN match_participants p ON m.match_id = p.match_id
                WHERE p.match_id IS NULL
                """
            )
        ]
        if limit is not None:
            pending = pending[:limit]
        print(f"Backfilling participants for {len(pending)} matches...")

        participant_batch = []
        total_backfilled = 0
        errors_in_a_row = 0

        for i, match_id in enumerate(pending):
            try:
                print(f"Backfilling match {i+1}/{len(pending)}: {match_id}")
                time.sleep(1.3)  # Match the per-call pacing of process_participants (100 req / 2 min limit)
                match_json = self.collector.collect_match_details(f"{region_prefix}_{match_id}")
                if match_json is None or "info" not in match_json:
                    print(f"Skipping match ID {match_id}: could not fetch match details.")
                    errors_in_a_row += 1
                    if errors_in_a_row >= max_errors_in_a_row:
                        print("Too many errors in a row, stopping backfill.")
                        break
                    continue

                participant_batch.extend(self.processor.process_participants(match_json))
                errors_in_a_row = 0

                if len(participant_batch) >= batch_size * 10:  # 10 participants per match
                    print(f"Inserting {len(participant_batch)} participant records into the database...")
                    self.insert_participant_batch(participant_batch)
                    total_backfilled += len(participant_batch)
                    participant_batch = []
            except Exception as e:
                print(f"Error backfilling match ID {match_id}: {e}")
                errors_in_a_row += 1
                if errors_in_a_row >= max_errors_in_a_row:
                    print("Too many errors in a row, stopping backfill.")
                    break
                continue

        if participant_batch:
            print(f"Inserting final {len(participant_batch)} participant records into the database...")
            self.insert_participant_batch(participant_batch)
            total_backfilled += len(participant_batch)
        print(f"Backfill complete. Total participant records inserted: {total_backfilled}")

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

    def insert_participant_batch(self, batch: list):
        if not batch:
            return
        # INSERT OR IGNORE so rows already present (e.g. from a concurrent run or
        # a batch retried after a partial failure) are skipped instead of failing
        # the whole batch on the (match_id, puuid) primary key.
        columns = list(batch[0].keys())
        query = (
            f"INSERT OR IGNORE INTO match_participants ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})"
        )
        with self.connection:
            self.connection.executemany(query, [[row[col] for col in columns] for row in batch])

    def retrieve_dataset(self) -> pd.DataFrame:
        query = "SELECT * FROM matches"
        return pd.read_sql_query(query, self.connection)

    def retrieve_match_participants(self) -> pd.DataFrame:
        query = "SELECT * FROM match_participants"
        return pd.read_sql_query(query, self.connection)

    def clean_dataset(self):
        self.execute_query("DELETE FROM matches")
        self.execute_query("DELETE FROM match_participants")

if __name__ == "__main__":
    dataset = Dataset(db_path="src/scuttle_bot/cache/ml_dataset.db")
    """      
    config = {
            "region": Region.NA,
            "queue": Queue.RANKED_SOLO_5x5,
            "sample_size": 300,
            "num_matches_per_player": 5,
            "max_errors_in_a_row": 5,
            "batch_size": 10,
            "challenger_league": True,
            "master_league": True,
            "grandmaster_league": True,
            "stratified_sampling": True
        }
        dataset.create_dataset(**config)
    """
    backfilling_config = {
        "region_prefix": "NA1",
        "max_errors_in_a_row": 5,
        "batch_size": 10
    }
    dataset.backfill_participants(**backfilling_config)

                