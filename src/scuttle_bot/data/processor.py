from scuttle_bot.service.schemas import Region, Queue
from scuttle_bot.service.utilities import get_champion_mapping

class Processor:
    def __init__(self):
        self.champion_mapping = get_champion_mapping()

    def process_data(self, match_json: dict, rank_json: dict) -> dict:
        info = match_json["info"]
        participants = info["participants"]
        teams = info["teams"]

        game_duration = info.get("gameDuration", 0)

        blue = {}
        red = {}

        for team in teams:
            if team["teamId"] == 100:
                blue_win = int(team["win"])
                blue_bans = self.process_bans(team.get("bans", []))
            else:
                red_bans = self.process_bans(team.get("bans", []))

        for p in participants:
            champ = p["championName"]
            team = "blue" if p["teamId"] == 100 else "red"
            role = p["teamPosition"].lower()

            if team == "blue":
                blue[role] = champ
            else:
                red[role] = champ


        return {
            "match_id": info["gameId"],
            "patch_version": info.get("gameVersion"),
            "blue_win": blue_win,

            "average_tier": self.process_ranked_stats(rank_json[0]) if rank_json else 0,

            "blue_top": blue["top"],
            "blue_jungle": blue["jungle"],
            "blue_mid": blue["mid"],
            "blue_adc": blue["adc"],
            "blue_support": blue["support"],

            "red_top": red["top"],
            "red_jungle": red["jungle"],
            "red_mid": red["mid"],
            "red_adc": red["adc"],
            "red_support": red["support"],

            "blue_bans": blue_bans,
            "red_bans": red_bans,

            "game_duration": game_duration,
            "queue_id": info.get("queueId"),
        }

    def process_bans(self, bans: list) -> list:
        return [self.champion_mapping.get(ban["championId"], "Unknown") for ban in bans]

    def process_ranked_stats(self, rank_json: dict) -> int:
        # Convert ranked stats to numerical mmr like value
        rank_value = {
            "IRON": 0,
            "BRONZE": 1,
            "SILVER": 2,
            "GOLD": 3,
            "PLATINUM": 4,
            "EMERALD": 5,
            "DIAMOND": 6,
            "MASTER": 7,
            "GRANDMASTER": 8,
            "CHALLENGER": 9
        }

        division_value = {
            "IV": 0,
            "III": 1,
            "II": 2,
            "I": 3
        }

        division_score = division_value.get(rank_json["rank"], 0)

        tier = rank_json["tier"] if rank_json else "IRON"
        mmr_like_score = rank_value[tier] * 4 + division_score
        return mmr_like_score