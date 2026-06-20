import re
from typing import Optional

from scuttle_bot.service.schemas import Region, Queue
from scuttle_bot.service.utilities import get_champ_to_idx, get_champion_mapping

class Processor:
    def __init__(self):
        self.champion_mapping = get_champion_mapping()
        self.champ_to_idx = get_champ_to_idx()

    def process_data(self, match_json: dict, rank_json: dict) -> Optional[dict]:
        info = match_json["info"]
        participants = info["participants"]
        teams = info["teams"]

        game_duration = info.get("gameDuration", 0)

        blue = {}
        red = {}

        if info.get("queueId") != 420:  # Only process ranked solo games
            print(f"Skipping match ID {info['gameId']} due to unsupported queue ID: {info.get('queueId')}")
            return None

        for team in teams:
            if team["teamId"] == 100:
                blue_win = int(team["win"])
                blue_bans = self.process_bans(team.get("bans", []))
            else:
                red_bans = self.process_bans(team.get("bans", []))

        for p in participants:
            champ = re.sub(r"[^A-Za-z0-9]", "", p["championName"]).lower()
            champ_id = self.champ_to_idx.get(champ, champ)
            team = "blue" if p["teamId"] == 100 else "red"
            role = p["teamPosition"].lower()

            if team == "blue":
                blue[role] = champ_id
            else:
                red[role] = champ_id

        for rank_data in rank_json:
            if rank_data["queueType"] == "RANKED_SOLO_5x5":
                rank_json = rank_data
                break

        return {
            "match_id": info["gameId"],
            "patch_version": info.get("gameVersion"),
            "blue_win": blue_win,

            "average_tier": self.process_ranked_stats(rank_data),

            "blue_top": blue["top"],
            "blue_jungle": blue["jungle"],
            "blue_mid": blue["middle"],
            "blue_adc": blue["bottom"],
            "blue_support": blue["utility"],

            "red_top": red["top"],
            "red_jungle": red["jungle"],
            "red_mid": red["middle"],
            "red_adc": red["bottom"],
            "red_support": red["utility"],

            "blue_ban_0": blue_bans[0] if len(blue_bans) > 0 else None,
            "blue_ban_1": blue_bans[1] if len(blue_bans) > 1 else None,
            "blue_ban_2": blue_bans[2] if len(blue_bans) > 2 else None,
            "blue_ban_3": blue_bans[3] if len(blue_bans) > 3 else None,
            "blue_ban_4": blue_bans[4] if len(blue_bans) > 4 else None,
            "red_ban_0": red_bans[0] if len(red_bans) > 0 else None,
            "red_ban_1": red_bans[1] if len(red_bans) > 1 else None,
            "red_ban_2": red_bans[2] if len(red_bans) > 2 else None,
            "red_ban_3": red_bans[3] if len(red_bans) > 3 else None,
            "red_ban_4": red_bans[4] if len(red_bans) > 4 else None,

            "game_duration": game_duration,
            "queue_id": info.get("queueId"),
        }

    def process_bans(self, bans: list) -> list:
        champ_names = [re.sub(r"[^A-Za-z0-9]", "", self.champion_mapping.get(ban["championId"], "Unknown")).lower() for ban in bans]
        return [self.champ_to_idx.get(champ, -1) for champ in champ_names]

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