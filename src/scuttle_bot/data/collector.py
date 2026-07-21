import requests
import os
from typing import Optional
import json
import random
import time

from scuttle_bot.service.schemas import Region, Queue, get_match_routing_url

class Collector:
    def _get_json(self, url: str, max_retries: int = 3):
        """GET that returns parsed JSON on 200, retries 429s honoring
        Retry-After, and returns None on any other error status so callers
        never receive a Riot error payload in place of real data."""
        for attempt in range(max_retries + 1):
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429 and attempt < max_retries:
                retry_after = int(response.headers.get("Retry-After", 10))
                print(f"Rate limited (429), waiting {retry_after}s before retrying...")
                time.sleep(retry_after)
                continue
            print(f"Request failed with status {response.status_code}: {url}")
            return None
        return None

    def __init__(self, region = Region.NA):
        from dotenv import load_dotenv
        load_dotenv()

        self.riot_key = os.getenv("RIOT_API_KEY")
        self.headers = {
            "X-Riot-Token": self.riot_key
        }
        self.riot_url = get_match_routing_url(region)  # match-v5's continental cluster for this platform
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
        
    def get_stratified_random_players(self, grouped_players: Optional[list[list]], num_players: int = 300) -> Optional[list]:
        if grouped_players is None:
            print("No players provided for stratified selection.")
            return None
        try:
            stratified_players = []
            for group in grouped_players:
                if group:  # Ensure the group is not empty
                    group_player_list = [player['puuid'] for player in group]
                    selected = random.sample(group_player_list, min(num_players // len(grouped_players), len(group_player_list)))
                    stratified_players.extend(selected)
            return stratified_players
        except Exception as e:
            print(f"Error selecting stratified random players: {e}")
            return None

    def collect_match_history(self, puuid: str, count: int = 3) -> Optional[dict]:
        try:
            return self._get_json(f"{self.riot_url}/lol/match/v5/matches/by-puuid/{puuid}/ids?count={count}")
        except Exception as e:
            print(f"Error collecting match history for PUUID {puuid}: {e}")
            return None

    def collect_match_details(self, match_id: str) -> Optional[dict]:
        try:
            return self._get_json(f"{self.riot_url}/lol/match/v5/matches/{match_id}")
        except Exception as e:
            print(f"Error collecting match details for match ID {match_id}: {e}")
            return None

    def collect_ranked_stats(self, summoner_id: str) -> Optional[dict]:
        try:
            return self._get_json(f"{self.lol_url}/lol/league/v4/entries/by-puuid/{summoner_id}")
        except Exception as e:
            print(f"Error collecting ranked stats for summoner ID {summoner_id}: {e}")
            return None

    def collect_champion_mastery(self, puuid: str, champion_id: int) -> Optional[dict]:
        try:
            return self._get_json(
                f"{self.lol_url}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{champion_id}"
            )
        except Exception as e:
            print(f"Error collecting champion mastery for PUUID {puuid}, champion {champion_id}: {e}")
            return None

    def collect_active_game(self, puuid: str) -> Optional[dict]:
        """Spectator-v5: current in-progress game for this player, or None if
        they aren't in one (Riot returns 404, which _get_json maps to None)."""
        try:
            return self._get_json(f"{self.lol_url}/lol/spectator/v5/active-games/by-summoner/{puuid}")
        except Exception as e:
            print(f"Error collecting active game for PUUID {puuid}: {e}")
            return None
