from opgg.v2.opgg import OPGG
from opgg.v2.params import Region

class ScuttleBotService:
    def __init__(self):
        self.opgg = OPGG()

    def get_user_stats(self, summoner_name: str, region: Region):
        try:
            stats = self.opgg.search(summoner_name, region)
            return stats
        except Exception as e:
            print(f"Error fetching user stats: {e}")
            return None
        
if __name__ == "__main__":
    service = ScuttleBotService()
    stats = service.get_user_stats("Faker", Region.NA)
    print(stats)