import re
import time
from typing import Optional

from scuttle_bot.utilities.schemas import Region, Queue

class Processor:
    def __init__(self, collector=None):
        self.collector = collector

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
            champ_id = p["championId"]
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
        return [ban["championId"] for ban in bans]

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

    def process_participants(self, match_json: dict, request_delay: float = 1.3) -> list[dict]:
        # ~20 extra Riot API calls per match (rank + mastery per participant). The
        # 1.3s delay keeps a full match (~21 calls incl. match detail) under the
        # personal-key limit of 100 requests / 2 min.
        if self.collector is None:
            raise ValueError("Processor requires a Collector instance to process participants.")

        info = match_json["info"]
        match_id = info["gameId"]

        rows = []
        for p in info["participants"]:
            puuid = p["puuid"]
            champion_id = p["championId"]
            team = "blue" if p["teamId"] == 100 else "red"
            role = p["teamPosition"].lower()

            rank_json = self.collector.collect_ranked_stats(puuid)
            if not isinstance(rank_json, list):  # None or a Riot error payload
                rank_json = []
            time.sleep(request_delay)
            tier, rank, league_points, wins, losses, win_rate = self._extract_solo_queue_stats(rank_json)

            mastery_json = self.collector.collect_champion_mastery(puuid, champion_id)
            if not isinstance(mastery_json, dict):
                mastery_json = {}
            time.sleep(request_delay)

            rows.append({
                "match_id": match_id,
                "puuid": puuid,

                "team": team,
                "role": role,
                "champion_id": champion_id,

                "tier": tier,
                "rank": rank,
                "league_points": league_points,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,

                "champion_points": mastery_json.get("championPoints"),
                "champion_level": mastery_json.get("championLevel"),
                "champion_last_play_time": mastery_json.get("lastPlayTime"),
            })

        return rows

    def _extract_solo_queue_stats(self, rank_json: list) -> tuple:
        for entry in rank_json:
            if entry.get("queueType") == "RANKED_SOLO_5x5":
                wins = entry.get("wins", 0)
                losses = entry.get("losses", 0)
                total_games = wins + losses
                win_rate = wins / total_games if total_games > 0 else None
                return entry.get("tier"), entry.get("rank"), entry.get("leaguePoints"), wins, losses, win_rate

        return None, None, None, None, None, None