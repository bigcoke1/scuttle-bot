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

    def create_dataset(self, region = Region.NA, queue = Queue.RANKED_SOLO_5x5):
        challenger_leagues = self.collector.collect_challenger_leagues(region, queue)
        random_players = self.collector.get_random_players(challenger_leagues)
        data = []
        if random_players:
            for puuid in random_players:
                time.sleep(2)  # To avoid hitting rate limits
                match_history = self.collector.collect_match_history(puuid) # list of match ids
                rank_json = self.collector.collect_ranked_stats(puuid) or {} # ranked stats for player
                if match_history is None:
                    break
                for match_id in match_history:
                    time.sleep(2)  # To avoid hitting rate limits
                    match_json = self.collector.collect_match_details(match_id)
                    if match_json is None:
                        break
                    processed_data = self.processor.process_data(match_json, rank_json)
                    data.append(processed_data)
        
        df = pd.DataFrame(data)
        df.to_sql("matches", self.connection, if_exists="append", index=False)
                    

                