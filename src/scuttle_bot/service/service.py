import json
import sys
import logging
import traceback
from dotenv import load_dotenv
import requests
import os
from typing import Optional

from src.scuttle_bot.infra.db_client import DatabaseClient
from scuttle_bot.service.schemas import Region

class ScuttleBotService:
    def __init__(self, db: DatabaseClient):
        from dotenv import load_dotenv
        load_dotenv()

        self.riot_key = os.getenv("RIOT_KEY")
        self.headers = {
            "X-Riot-Token": self.riot_key
        }
        self.riot_url = "https://americas.api.riotgames.com"
        self.lol_url = "https://{region}.api.riotgames.com"
        self.db = db
        self.champion_mapping = self.get_champion_mapping()

    def get_complete_summoner_info(self, summoner_name, tag_line, region, num_masteries, num_matches) -> Optional[str]:
        if isinstance(region, str):
            region = Region(region)
        try:
            ranked_stats = self.search_summoner(region, summoner_name, tag_line)
            if ranked_stats is None or ranked_stats == []:
                ranked_stats = [None, None]
            

            champion_masteries = self.get_top_champion_masteries(region, summoner_name, tag_line, count=num_masteries)
            if champion_masteries is None:
                champion_masteries = []

            recent_matches = self.get_ranked_matches(summoner_name, tag_line, count=num_matches)
            if recent_matches is None:
                recent_matches = []

            return self.summoner_formatter(summoner_name, tag_line, region, [ranked_stats[0], ranked_stats[1], champion_masteries, recent_matches])
        except Exception as e:
            self.error_traceback()
            return f"An error occurred: {str(e)}. Check logs for variable states."
    
    def get_champion_mapping(self):
        try:
            version = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()[0]
            url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
            data = requests.get(url).json()["data"]

            mapping = {int(info["key"]): info["name"] for info in data.values()}
            return mapping
        except Exception as e:
            self.error_traceback()
            return {}
    
    def get_puuid(self, summoner_name: str, tag_line: str) -> Optional[str]:
        try:
            response = requests.get(f"{self.riot_url}/riot/account/v1/accounts/by-riot-id/{summoner_name}/{tag_line}", headers=self.headers)
            if response.status_code == 200:
                response = response.json()
                puuid = response["puuid"]
                return puuid
            else:
                raise Exception(response.status_code)
        except Exception as e:
            self.error_traceback()
            return None

    def search_summoner(self, region: Region, summoner_name: str, tag_line: str) -> Optional[str]:
        if isinstance(region, str):
            region = Region(region)
        try:
            puuid = self.get_puuid(summoner_name, tag_line)
            if puuid is None:
                return None
            
            url = self.lol_url.format(region=region.value)
            response = requests.get(f"{url}/lol/league/v4/entries/by-puuid/{puuid}", headers=self.headers)
            if response.status_code == 200:
                response = response.json()
                return response
            else:
                raise Exception(response.status_code)
        except Exception as e:
            self.error_traceback()
            return None

    def get_top_champion_masteries(self, region: Region, summoner_name: str, tag_line: str, count = 5) -> Optional[list]:
        if isinstance(region, str):
            region = Region(region)
        try:
            puuid = self.get_puuid(summoner_name, tag_line)
            if puuid is None:
                return None

            url = self.lol_url.format(region=region.value)
            response = requests.get(f"{url}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/top?count={count}", headers=self.headers)
            if response.status_code == 200:
                response = response.json()
                for mastery in response:
                    mastery["championName"] = self.champion_mapping.get(mastery["championId"], mastery["championId"])
                return response
            else:
                raise Exception(response.status_code)
        except Exception as e:
            self.error_traceback()
            return None
        
    def get_ranked_matches(self, summoner_name: str, tag_line: str, start_time=None, end_time=None, count=5) -> Optional[list]:
        from datetime import datetime, timedelta
        if end_time is None:
            end_time = int(datetime.now().timestamp())
        if start_time is None:
            start_time = end_time - timedelta(days=7).total_seconds()

        try:
            puuid = self.get_puuid(summoner_name, tag_line)
            if puuid is None:
                return None

            response = requests.get(f"{self.riot_url}/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count={count}&type=ranked", headers=self.headers)
            if response.status_code == 200:
                response = response.json()
                stats = []
                for match_id in response:
                    stats.append(self.get_match_stats(match_id, puuid, summoner_name=summoner_name))
                return stats
            else:
                raise Exception(response.status_code)
        except Exception as e:
            self.error_traceback()
            return None

    def get_match_stats(self, match_id: str, puuid: str, summoner_name: str) -> Optional[dict]:
        try:
            if self.db.exists_match(match_id):
                match = self.db.retrieve_match(match_id)
            else:
                match = requests.get(f"{self.riot_url}/lol/match/v5/matches/{match_id}", headers=self.headers).json()
                self.db.store_match(match_id=match_id, summoner_name=summoner_name, data=json.dumps(match))
            if match is None:
                return None

            match_info = match["info"]
            for participant in match_info['participants']:
                if participant['puuid'] == puuid:
                    return {
                        "champion": participant['championName'],
                        "kills": participant['kills'],
                        "deaths": participant['deaths'],
                        "assists": participant['assists'],
                        "win": participant['win']
                    }
                        
        except Exception as e:
            self.error_traceback()
            return None

    def format_recent_matches(self, matches):
        formatted = []
        for match in matches:
            result = f"Champion: {match['champion']}, K/D/A: {match['kills']}/{match['deaths']}/{match['assists']}, Win: {'Yes' if match['win'] else 'No'}"
            formatted.append(result)
        return "\n".join(formatted)

    def summoner_formatter(self, summoner_name: str, tag_line: str, region: Region, data):
        if isinstance(region, str):
            region = Region(region)

        summoner_name = summoner_name.strip()
        tag_line = tag_line.strip()

        # Helper function to safely get nested values
        def safe_get(obj, key, default="N/A"):
            return obj.get(key, default) if isinstance(obj, dict) else default

        # Extract ranked stats with defaults
        flex_stats = data[0] if len(data) > 0 and data[0] else {}
        solo_stats = data[1] if len(data) > 1 and data[1] else {}
        champion_masteries = data[2] if len(data) > 2 else []
        recent_matches = data[3] if len(data) > 3 else []

        # Build champion mastery section safely
        mastery_section = "Champion Mastery (Top 3):\n"
        for i in range(3):
            if i < len(champion_masteries):
                mastery = champion_masteries[i]
                name = safe_get(mastery, "championName", "Unknown")
                level = safe_get(mastery, "championLevel", "N/A")
                points = safe_get(mastery, "championPoints", "N/A")
                mastery_section += f"{i + 1}. {name} - Level {level} - Points: {points}\n"
            else:
                mastery_section += f"{i + 1}. No data available\n"

        # Build recent matches section safely
        matches_section = self.format_recent_matches(recent_matches) if recent_matches else "No recent matches available"

        result = f"""
Summoner Name: {summoner_name}#{tag_line}
Region: {region.value}

Queue Type: Ranked Flex
Tier: {safe_get(flex_stats, "tier")}
Rank: {safe_get(flex_stats, "rank")}
LP: {safe_get(flex_stats, "leaguePoints")}
Wins: {safe_get(flex_stats, "wins")}
Losses: {safe_get(flex_stats, "losses")}

Queue Type: Ranked Solo/Duo
Tier: {safe_get(solo_stats, "tier")}
Rank: {safe_get(solo_stats, "rank")}
LP: {safe_get(solo_stats, "leaguePoints")}
Wins: {safe_get(solo_stats, "wins")}
Losses: {safe_get(solo_stats, "losses")}

{mastery_section}
Recent Ranked Matches (Last {len(recent_matches)}):
{matches_section}
        """
        return result

    def error_traceback(self):
        # 1. Capture the traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # 2. Get the "frame" where the error actually happened
        # We go to the end of the traceback to find the deepest function (your tool)
        tb = exc_traceback
        if tb is not None:
            while tb.tb_next:
                tb = tb.tb_next
            frame = tb.tb_frame
        else:
            return
        
        # 3. Log the error and the variables in that function
        logging.error(f"CRASH: {exc_value}")
        logging.error(f"Error occurred in: {frame.f_code.co_name}")
        
        # 4. Filter and log only lists to find the culprit
        logging.error("--- Local Variable States ---")
        for var_name, var_value in frame.f_locals.items():
            if isinstance(var_value, list):
                logging.error(f"LIST FOUND: {var_name} (Length: {len(var_value)}) -> {var_value}")
            elif isinstance(var_value, (str, int, dict)):
                logging.error(f"VAR: {var_name} = {var_value}")
        
if __name__ == "__main__":
    load_dotenv()
    db_path = os.getenv("DB_PATH", "src/scuttle_bot/cache/scuttle_bot.db")
    service = ScuttleBotService(db=DatabaseClient(db_path))

    result = service.get_complete_summoner_info("Sorrrymakerrr", "DOINB", Region.NA, num_masteries=5, num_matches=5)
    print(result)
