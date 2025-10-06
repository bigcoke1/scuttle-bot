import requests
import os
from typing import Optional

from src.scuttle_bot.infra.db_client import DatabaseClient

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

    def get_complete_summoner_info(self, region, game_name, tag_line):
        try:
            ranked_stats = self.search_summoner(region, game_name, tag_line)
            if ranked_stats is None:
                return None

            champion_masteries = self.get_top_champion_masteries(region, game_name, tag_line)
            if champion_masteries is None:
                return None

            recent_matches = self.get_ranked_matches(game_name, tag_line)
            if recent_matches is None:
                return None

            return self.summoner_formatter(game_name, tag_line, region, [ranked_stats[0], ranked_stats[1], champion_masteries, recent_matches])
        except Exception as e:
            print(e)
            return None
    
    def get_champion_mapping(self):
        try:
            version = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()[0]
            url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
            data = requests.get(url).json()["data"]

            mapping = {int(info["key"]): info["name"] for info in data.values()}
            return mapping
        except Exception as e:
            print(e)
            return {}
    
    def get_puuid(self, game_name, tag_line) -> Optional[str]:
        try:
            response = requests.get(f"{self.riot_url}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}", headers=self.headers)
            if response.status_code == 200:
                response = response.json()
                puuid = response["puuid"]
                return puuid
            else:
                raise Exception(response.status_code)
        except Exception as e:
            print(e)
    
    def search_summoner(self, region, game_name, tag_line) -> Optional[str]:
        try:
            puuid = self.get_puuid(game_name, tag_line)
            if puuid is None:
                return None
            
            url = self.lol_url.format(region=region)
            response = requests.get(f"{url}/lol/league/v4/entries/by-puuid/{puuid}", headers=self.headers)
            if response.status_code == 200:
                response = response.json()
                return response
            else:
                raise Exception(response.status_code)
        except Exception as e:
            print(e)

    def get_top_champion_masteries(self, region, game_name, tag_line, count = 5) -> Optional[list]:
        try:
            puuid = self.get_puuid(game_name, tag_line)
            if puuid is None:
                return None

            url = self.lol_url.format(region=region)
            response = requests.get(f"{url}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/top?count={count}", headers=self.headers)
            if response.status_code == 200:
                response = response.json()
                for mastery in response:
                    mastery["championName"] = self.champion_mapping.get(mastery["championId"], mastery["championId"])
                return response
            else:
                raise Exception(response.status_code)
        except Exception as e:
            print(e)

    def get_ranked_matches(self, game_name, tag_line, start_time=None, end_time=None, count=5) -> Optional[list]:
        from datetime import datetime, timedelta
        if end_time is None:
            end_time = int(datetime.now().timestamp())
        if start_time is None:
            start_time = end_time - timedelta(days=7).total_seconds()

        try:
            puuid = self.get_puuid(game_name, tag_line)
            if puuid is None:
                return None

            response = requests.get(f"{self.riot_url}/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count={count}&type=ranked", headers=self.headers)
            if response.status_code == 200:
                response = response.json()
                stats = []
                for match_id in response:
                    stats.append(self.get_ranked_stats(match_id, puuid))
                return stats
            else:
                raise Exception(response.status_code)
        except Exception as e:
            print(e)
            return None

    def get_ranked_stats(self, match_id, puuid) -> Optional[dict]:
        try:
            match = requests.get(f"{self.riot_url}/lol/match/v5/matches/{match_id}", headers=self.headers).json()
            if match is None:
                return None

            self.db.store_match(match_id, str(match))

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
            print(e)
            return None

    def format_recent_matches(self, matches):
        formatted = []
        for match in matches:
            result = f"Champion: {match['champion']}, K/D/A: {match['kills']}/{match['deaths']}/{match['assists']}, Win: {'Yes' if match['win'] else 'No'}"
            formatted.append(result)
        return "\n".join(formatted)
    
    def summoner_formatter(self, game_name, tag_line, region, data):
        game_name = game_name.strip()
        tag_line = tag_line.strip()

        result = f"""
Summoner Name: {game_name}#{tag_line}
Region: {region}

Queue Type: Ranked Flex
Tier: {data[0]["tier"]}
Rank: {data[0]["rank"]}
LP: {data[0]["leaguePoints"]}
Wins: {data[0]["wins"]}
Losses: {data[0]["losses"]}

Queue Type: Ranked Solo/Duo
Tier: {data[1]["tier"]}
Rank: {data[1]["rank"]}
LP: {data[1]["leaguePoints"]}
Wins: {data[1]["wins"]}
Losses: {data[1]["losses"]}

Champion Mastery (Top 3):
1. {data[2][0]["championName"]} - Level {data[2][0]["championLevel"]} - Points: {data[2][0]["championPoints"]}
2. {data[2][1]["championName"]} - Level {data[2][1]["championLevel"]} - Points: {data[2][1]["championPoints"]}
3. {data[2][2]["championName"]} - Level {data[2][2]["championLevel"]} - Points: {data[2][2]["championPoints"]}

Recent Ranked Matches (Last {len(data[3])}):
{self.format_recent_matches(data[3])}
        """
        return result



        
        
        
if __name__ == "__main__":
    service = ScuttleBotService(db=DatabaseClient("../cache/scuttle_bot.db"))
    result = service.get_complete_summoner_info("na1", "alegs", "GBS")
    print(result)
