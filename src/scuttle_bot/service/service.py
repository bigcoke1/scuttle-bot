import os
from dotenv import load_dotenv

from scuttle_bot.infra.db_client import DatabaseClient
from scuttle_bot.utilities.schemas import Region
from scuttle_bot.infra.aws_client import get_riot_api_key
from scuttle_bot.utilities.utilities import get_champion_mapping, error_traceback
from scuttle_bot.service.riot_client import RiotClientMixin
from scuttle_bot.service.summoner_profile import SummonerProfileMixin
from scuttle_bot.service.registration import RegistrationMixin
from scuttle_bot.service.personality_service import PersonalityMixin
from scuttle_bot.analyzer.match_analyzer import MatchAnalyzerMixin


class ScuttleBotService(RiotClientMixin, SummonerProfileMixin, RegistrationMixin, PersonalityMixin, MatchAnalyzerMixin):
    """Facade over Riot API access, summoner profile formatting, Discord user
    registration, personality management, and match analysis -- each concern
    lives in its own mixin module (riot_client.py, summoner_profile.py,
    registration.py, personality_service.py, analyzer/match_analyzer.py);
    this class just composes them and owns the shared state (HTTP headers,
    db, champion mapping) they all read via self.
    """

    def __init__(self, db: DatabaseClient):
        load_dotenv()

        self.riot_key = get_riot_api_key() or os.getenv("RIOT_API_KEY")
        self.headers = {
            "X-Riot-Token": self.riot_key
        }
        self.lol_url = "https://{region}.api.riotgames.com"
        self.db = db
        self.champion_mapping = get_champion_mapping()
        self.error_traceback = error_traceback


if __name__ == "__main__":
    load_dotenv()
    db_path = os.getenv("DB_PATH", "src/scuttle_bot/cache/scuttle_bot.db")
    service = ScuttleBotService(db=DatabaseClient(db_path))

    result = service.get_complete_summoner_info("Sorrrymakerrr", "DOINB", Region.NA, num_masteries=5, num_matches=5)
    print(result)
