import requests
import os
from typing import Optional

class ScuttleBotService:
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()

        self.riot_key = os.getenv("RIOT_KEY")
        self.headers = {
            "X-Riot-Token": self.riot_key
        }
        self.riot_url = "https://americas.api.riotgames.com"
        self.lol_url = "https://na1.api.riotgames.com"

    def get_puuid(self, region, game_name, tag_line) -> Optional[str]:
        try:
            region, game_name, tag_line = region, game_name, tag_line
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
            region, game_name, tag_line = region, game_name, tag_line
            puuid = self.get_puuid(region, game_name, tag_line)

            response = requests.get(f"{self.lol_url}/lol/league/v4/entries/by-puuid/{puuid}", headers=self.headers)
            if response.status_code == 200:
                response = response.json()
                result = self.string_formatter(game_name, tag_line, region, response)
                return result
            else:
                raise Exception(response.status_code)
        except Exception as e:
            print(e)

    def string_formatter(self, game_name, tag_line, region, data):
        result = f"""
        Summoner Name: {game_name}#{tag_line} \n
        Region: {region}
        Queue Type: {data[0]["queueType"]} \n
        Tier: {data[0]["tier"]} \n
        Rank: {data[0]["rank"]} \n
        LP: {data[0]["leaguePoints"]} \n
        Wins: {data[0]["wins"]} \n
        Losses: {data[0]["losses"]} \n\n
        
        Queue Type: {data[1]["queueType"]} \n
        Tier: {data[1]["tier"]} \n
        Rank: {data[1]["rank"]} \n
        LP: {data[1]["leaguePoints"]} \n
        Wins: {data[1]["wins"]} \n
        Losses: {data[1]["losses"]} \n\n
        """
        return result



        
        
        
if __name__ == "__main__":
    service = ScuttleBotService()
    service.search_summoner("na1", "alegs", "GBS")
