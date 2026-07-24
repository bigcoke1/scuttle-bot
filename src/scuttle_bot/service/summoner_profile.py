"""Composes a full player profile from RiotClientMixin's individual lookups
and formats it as human-readable text.

SummonerProfileMixin is mixed into ScuttleBotService alongside
RiotClientMixin, whose methods (search_summoner, get_top_champion_masteries,
get_ranked_matches, format_recent_matches) it calls via self.
"""

from typing import Optional

from scuttle_bot.utilities.schemas import Region


class SummonerProfileMixin:
    def get_complete_summoner_info(self, summoner_name: str, tag_line: str, region: Region, num_masteries: int, num_matches: int) -> Optional[str]:
        """Fetches a full player profile in one call: ranked flex/solo stats, top
        champion masteries, and recent ranked match results, formatted as
        human-readable text. Prefer this over calling search_summoner,
        get_top_champion_masteries, and get_ranked_matches separately when the
        user wants a general overview of a player.

        Args:
            summoner_name: Riot ID game name (the part before the #).
            tag_line: Riot ID tagline (the part after the #).
            region: Riot region the account is registered in, e.g. 'NA', 'EUW1', 'KR'.
            num_masteries: How many top champion masteries to include.
            num_matches: How many recent ranked matches to include.
        """
        if isinstance(region, str):
            region = Region(region)
        try:
            ranked_stats = self.search_summoner(region, summoner_name, tag_line)
            if ranked_stats is None or ranked_stats == []:
                ranked_stats = [None, None]

            champion_masteries = self.get_top_champion_masteries(region, summoner_name, tag_line, count=num_masteries)
            if champion_masteries is None:
                champion_masteries = []

            recent_matches = self.get_ranked_matches(summoner_name, tag_line, region=region, count=num_matches)
            if recent_matches is None:
                recent_matches = []

            return self.summoner_formatter(summoner_name, tag_line, region, [ranked_stats[0], ranked_stats[1], champion_masteries, recent_matches])
        except Exception as e:
            self.error_traceback()
            return f"An error occurred: {str(e)}. Check logs for variable states."

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
