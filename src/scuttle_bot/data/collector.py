import requests
import os
from typing import Optional
import json
import random
import time

from scuttle_bot.service.schemas import Region, Queue

class Collector:
    def __init__(self, region = Region.NA):
        from dotenv import load_dotenv
        load_dotenv()

        self.riot_key = os.getenv("RIOT_KEY")
        self.headers = {
            "X-Riot-Token": self.riot_key
        }
        self.riot_url = "https://americas.api.riotgames.com"
        self.lol_url = "https://{region}.api.riotgames.com"
        self.lol_url = self.lol_url.format(region=region.value)  # Using the provided region for ranked stats

    def collect_challenger_leagues(self, region: Region, queue: Queue) -> Optional[dict]:
        try:
            url = self.lol_url.format(region=region.value)
            response = requests.get(f"{url}/lol/league/v4/challengerleagues/by-queue/{queue.value}", headers=self.headers)
            return response.json()
        except Exception as e:
            print(f"Error collecting challenger leagues: {e}")
            return None
    
    def collect_master_leagues(self, region: Region, queue: Queue) -> Optional[dict]:
        try:
            url = self.lol_url.format(region=region.value)
            response = requests.get(f"{url}/lol/league/v4/masterleagues/by-queue/{queue.value}", headers=self.headers)
            return response.json()
        except Exception as e:
            print(f"Error collecting master leagues: {e}")
            return None
        
    def collect_grandmaster_leagues(self, region: Region, queue: Queue) -> Optional[dict]:
        try:
            url = self.lol_url.format(region=region.value)
            response = requests.get(f"{url}/lol/league/v4/grandmasterleagues/by-queue/{queue.value}", headers=self.headers)
            return response.json()
        except Exception as e:
            print(f"Error collecting grandmaster leagues: {e}")
            return None
   
    def get_random_players(self, players: Optional[list], num_players: int = 300) -> Optional[list]:
        if players is None:
            print("No players provided for random selection.")
            return None
        try:
            player_list = [player['puuid'] for player in players]
            if not player_list:
                print("No player entries found in the provided data.")
                return None
            random_players = random.sample(player_list, min(num_players, len(player_list)))
            return random_players
        except Exception as e:
            print(f"Error selecting random players: {e}")
            return None

    def collect_match_history(self, puuid: str, count: int = 3) -> Optional[dict]:
        try:
            response = requests.get(f"{self.riot_url}/lol/match/v5/matches/by-puuid/{puuid}/ids?count={count}", headers=self.headers)
            return response.json()
        except Exception as e:
            print(f"Error collecting match history for PUUID {puuid}: {e}")
            return None
        
    def collect_match_details(self, match_id: str) -> Optional[dict]:
        try:
            response = requests.get(f"{self.riot_url}/lol/match/v5/matches/{match_id}", headers=self.headers)
            return response.json()
        except Exception as e:
            print(f"Error collecting match details for match ID {match_id}: {e}")
            return None
        
    def collect_ranked_stats(self, summoner_id: str) -> Optional[dict]:
        try:
            response = requests.get(f"{self.lol_url}/lol/league/v4/entries/by-puuid/{summoner_id}", headers=self.headers)
            return response.json()
        except Exception as e:
            print(f"Error collecting ranked stats for summoner ID {summoner_id}: {e}")
            return None
            