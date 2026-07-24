"""Direct Riot API access: summoner lookup, ranked stats, champion mastery,
live games, and match history.

RiotClientMixin is mixed into ScuttleBotService, which owns the shared HTTP
session state (self.headers, self.lol_url, self.db, self.champion_mapping,
self.error_traceback) these methods read -- it isn't meant to be used
standalone.
"""

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional
from urllib.parse import parse_qs, urlparse

import requests

from scuttle_bot.utilities.schemas import Region, get_account_routing_url, get_match_routing_url
from scuttle_bot.data.collector import Collector
from scuttle_bot.utilities.role_inference import infer_roles

# Smite is the only summoner spell reserved for one role in 5v5 ranked, so it's
# a reliable jungler tell. Spectator-v5 doesn't expose lane/role for the other
# four picks -- Match-v5's teamPosition is a post-game inference Riot computes
# from timeline data, not something available on a live game.
SMITE_SPELL_ID = 11

# requests.get() has no default timeout -- a stalled connection blocks
# forever with no exception ever raised, which would freeze the bot's whole
# event loop indefinitely. Every call in this module gets one explicitly.
REQUEST_TIMEOUT = 30

# Match-v5's teamPosition uses Riot's internal lane names -- normalized here
# to match the top/jungle/mid/adc/support vocabulary used elsewhere in this
# codebase (e.g. get_active_game, PlayerDraftEntry).
_TEAM_POSITION_TO_ROLE = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "mid",
    "BOTTOM": "adc",
    "UTILITY": "support",
}

# Replay download URLs are S3 paths like
# ".../lol-prod-us-west-2-match-history-replay/na1_5607321601/0.replay?...",
# with the match ID embedded lowercase in the path -- this pulls it out and
# get_replay_urls() upper-cases it to match Riot's normal matchId casing
# (e.g. "NA1_5607321601") used everywhere else in this codebase.
_REPLAY_URL_MATCH_ID_RE = re.compile(r"/([A-Za-z0-9]+_\d+)/\d+\.replay")


class RiotClientMixin:
    def get_puuid(self, summoner_name: str, tag_line: str, region: Region = Region.NA) -> Optional[str]:
        if isinstance(region, str):
            region = Region(region)
        try:
            riot_url = get_account_routing_url(region)
            response = requests.get(f"{riot_url}/riot/account/v1/accounts/by-riot-id/{summoner_name}/{tag_line}", headers=self.headers, timeout=REQUEST_TIMEOUT)
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
        """Looks up a player's current ranked flex and ranked solo/duo stats
        (tier, rank, LP, wins, losses) by Riot ID. Returns None if the player
        can't be found or has no ranked data.

        Args:
            region: Riot region the account is registered in, e.g. 'NA', 'EUW1', 'KR'.
            summoner_name: Riot ID game name (the part before the #).
            tag_line: Riot ID tagline (the part after the #).
        """
        if isinstance(region, str):
            region = Region(region)
        try:
            puuid = self.get_puuid(summoner_name, tag_line, region=region)
            if puuid is None:
                return None

            url = self.lol_url.format(region=region.value)
            response = requests.get(f"{url}/lol/league/v4/entries/by-puuid/{puuid}", headers=self.headers, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                response = response.json()
                return response
            else:
                raise Exception(response.status_code)
        except Exception as e:
            self.error_traceback()
            return None

    def get_top_champion_masteries(self, region: Region, summoner_name: str, tag_line: str, count: int = 5) -> Optional[list]:
        """Looks up a player's highest-mastery champions by Riot ID, most points
        first. Returns None if the player can't be found.

        Args:
            region: Riot region the account is registered in, e.g. 'NA', 'EUW1', 'KR'.
            summoner_name: Riot ID game name (the part before the #).
            tag_line: Riot ID tagline (the part after the #).
            count: How many champions to return, highest mastery first.
        """
        if isinstance(region, str):
            region = Region(region)
        try:
            puuid = self.get_puuid(summoner_name, tag_line, region=region)
            if puuid is None:
                return None

            url = self.lol_url.format(region=region.value)
            response = requests.get(f"{url}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/top?count={count}", headers=self.headers, timeout=REQUEST_TIMEOUT)
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

    def get_active_game(self, summoner_name: str, tag_line: str, region: Region) -> Optional[dict]:
        """Spectator-v5 lookup for a live in-progress game. Returns None if the
        player isn't currently in one. The API doesn't expose lane/role, so
        roles are inferred: the jungler is identified via Smite, and the other
        4 per team are assigned by matching historical pick-role frequency
        (see build_champion_roles.py) -- best-effort, not ground truth."""
        if isinstance(region, str):
            region = Region(region)
        try:
            puuid = self.get_puuid(summoner_name, tag_line, region=region)
            if puuid is None:
                return None

            game = Collector(region).collect_active_game(puuid)
            if game is None:
                return None

            raw_participants = game.get("participants", [])

            team_champion_ids = {"blue": [], "red": []}
            known_roles = {"blue": {}, "red": {}}
            for p in raw_participants:
                team = "blue" if p.get("teamId") == 100 else "red"
                champ_id = p.get("championId")
                team_champion_ids[team].append(champ_id)
                if SMITE_SPELL_ID in (p.get("spell1Id"), p.get("spell2Id")):
                    known_roles[team][champ_id] = "jungle"

            role_by_champ = {
                team: infer_roles(team_champion_ids[team], known=known_roles[team])
                for team in ("blue", "red")
            }

            participants = []
            for p in raw_participants:
                team = "blue" if p.get("teamId") == 100 else "red"
                champ_id = p.get("championId")
                participants.append({
                    "puuid": p.get("puuid"),
                    "riot_id": p.get("riotId"),
                    "champion": self.champion_mapping.get(champ_id, champ_id),
                    "team": team,
                    "role": role_by_champ[team].get(champ_id),
                })

            bans = [
                {
                    "champion": self.champion_mapping.get(b.get("championId"), b.get("championId")),
                    "team": "blue" if b.get("teamId") == 100 else "red",
                }
                for b in game.get("bannedChampions", [])
            ]

            return {
                "game_id": game.get("gameId"),
                "game_mode": game.get("gameMode"),
                "game_length_seconds": game.get("gameLength"),
                "participants": participants,
                "bans": bans,
            }
        except Exception as e:
            self.error_traceback()
            return None

    def _fetch_match_ids(
        self,
        puuid: str,
        region: Region,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        count: int = 5,
    ) -> list[str]:
        """Raw match-v5 matchlist lookup: the most recent `count` ranked match
        IDs for a puuid, most recent first. start_time/end_time (unix
        seconds) are optional -- omitted entirely means "most recent, no date
        restriction" rather than defaulting to a window, so callers with
        different default policies (get_ranked_matches's 7-day default vs.
        the analyzer's "just the last N games") can layer their own defaults
        on top. Internal helper -- not exposed as an LLM tool.
        """
        match_url = get_match_routing_url(region)
        url = f"{match_url}/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count={count}&type=ranked"
        if start_time is not None:
            url += f"&startTime={int(start_time)}"
        if end_time is not None:
            url += f"&endTime={int(end_time)}"
        response = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            raise Exception(response.status_code)
        return response.json()

    def _get_cached_match_detail(self, match_id: str, summoner_name: str, region: Region) -> Optional[dict]:
        """Raw match-v5 match detail, from the local cache if present, else
        fetched from Riot and cached. Shared by get_match_stats and the
        performance-trend analyzer so both draw from (and populate) the same
        cache instead of double-fetching. Internal helper -- not exposed as
        an LLM tool.
        """
        if self.db.exists_match(match_id):
            return self.db.retrieve_match(match_id)
        match_url = get_match_routing_url(region)
        match = requests.get(f"{match_url}/lol/match/v5/matches/{match_id}", headers=self.headers, timeout=REQUEST_TIMEOUT).json()
        self.db.store_match(match_id=match_id, summoner_name=summoner_name, data=json.dumps(match))
        return match

    def get_match_timeline(self, match_id: str, region: Region = Region.NA) -> Optional[dict]:
        """Raw match-v5 timeline for a match: ~60s frames with each
        participant's gold/xp/CS at that point, plus timestamped events
        (kills, objective takes, item purchases). Cached locally like match
        details, since a finished match's timeline never changes. This is an
        internal building block for analyze_performance_trend and
        find_notable_moments -- a full timeline is large and not useful on
        its own, so it isn't exposed as its own LLM tool.
        """
        if isinstance(region, str):
            region = Region(region)
        try:
            if self.db.exists_match_timeline(match_id):
                return self.db.retrieve_match_timeline(match_id)

            match_url = get_match_routing_url(region)
            response = requests.get(f"{match_url}/lol/match/v5/matches/{match_id}/timeline", headers=self.headers, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                raise Exception(response.status_code)

            timeline = response.json()
            self.db.store_match_timeline(match_id=match_id, data=json.dumps(timeline))
            return timeline
        except Exception as e:
            self.error_traceback()
            return None

    def get_replay_urls(self, puuid: str, region: Region = Region.NA) -> dict:
        """Fetches pre-signed download URLs for a player's most recent
        available match replays (.rofl files, playable via "Watch" in the
        League client). Riot only keeps a replay available for a limited
        time after the match is played, so not every recent match has one --
        and unlike match/timeline data, these URLs are never cached, since
        each one is a temporary signed link that expires (typically about an
        hour after it's issued).

        Returns a dict keyed by match ID (e.g. 'NA1_5607321601'), each value
        {"url": <download url>, "expires_at": <ISO timestamp, or None if it
        couldn't be determined>}. Internal helper used by
        find_notable_moments -- not exposed as its own LLM tool, since a bare
        list of signed URLs with no context isn't useful on its own.
        """
        if isinstance(region, str):
            region = Region(region)
        try:
            match_url = get_match_routing_url(region)
            response = requests.get(f"{match_url}/lol/match/v5/matches/by-puuid/{puuid}/replays", headers=self.headers, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                raise Exception(response.status_code)

            replays = {}
            for url in response.json().get("matchFileURLs", []):
                match = _REPLAY_URL_MATCH_ID_RE.search(url)
                if match is None:
                    continue
                match_id = match.group(1).upper()
                replays[match_id] = {"url": url, "expires_at": self._parse_replay_expiry(url)}
            return replays
        except Exception as e:
            self.error_traceback()
            return {}

    def _parse_replay_expiry(self, url: str) -> Optional[str]:
        """Best-effort read of the S3 pre-signed URL's own X-Amz-Date/
        X-Amz-Expires query params to compute when it actually expires,
        rather than hardcoding an assumed TTL that could drift from reality
        if Riot changes it. Returns None (not an error) if the URL doesn't
        have the expected params -- callers treat that as "unknown expiry",
        not a failure."""
        try:
            query = parse_qs(urlparse(url).query)
            issued_raw = query.get("X-Amz-Date", [None])[0]
            expires_raw = query.get("X-Amz-Expires", [None])[0]
            if not issued_raw or not expires_raw:
                return None
            issued = datetime.strptime(issued_raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            return (issued + timedelta(seconds=int(expires_raw))).isoformat()
        except Exception:
            return None

    def get_ranked_matches(
        self,
        summoner_name: str,
        tag_line: str,
        region: Region = Region.NA,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        count: int = 5,
        stats_level: Literal["personal", "advanced"] = "personal",
    ) -> Optional[list]:
        """Looks up a player's recent ranked match results by Riot ID.
        Defaults to the last 7 days if no time window is given.

        Args:
            summoner_name: Riot ID game name (the part before the #).
            tag_line: Riot ID tagline (the part after the #).
            region: Riot region the account is registered in, e.g. 'NA', 'EUW1', 'KR'.
            start_time: Unix timestamp (seconds) to search from. Defaults to 7 days before end_time.
            end_time: Unix timestamp (seconds) to search until. Defaults to now.
            count: Maximum number of matches to return.
            stats_level: 'personal' (default) returns only this player's champion, KDA, and win/loss per match. 'advanced' additionally returns every other participant in each match (their Riot ID, champion, team, role, KDA, and win/loss) -- use 'advanced' when the user wants to know how other players in the game performed, not just this player.
        """
        if isinstance(region, str):
            region = Region(region)
        from datetime import datetime, timedelta
        if end_time is None:
            end_time = int(datetime.now().timestamp())
        if start_time is None:
            start_time = end_time - timedelta(days=7).total_seconds()

        try:
            puuid = self.get_puuid(summoner_name, tag_line, region=region)
            if puuid is None:
                return None

            match_ids = self._fetch_match_ids(puuid, region=region, start_time=start_time, end_time=end_time, count=count)
            stats = []
            for match_id in match_ids:
                stats.append(self.get_match_stats(match_id, puuid, summoner_name=summoner_name, region=region, stats_level=stats_level))
            return stats
        except Exception as e:
            self.error_traceback()
            return None

    def get_match_stats(
        self,
        match_id: str,
        puuid: str,
        summoner_name: str,
        region: Region = Region.NA,
        stats_level: Literal["personal", "advanced"] = "personal",
    ) -> Optional[dict]:
        """Looks up one player's stats for one specific, already-known match,
        by match ID and puuid -- not a player's recent match history. Prefer
        get_ranked_matches for "how has this player been doing lately"
        questions; reach for this instead only when you already have a
        specific match_id in hand (e.g. from a prior get_ranked_matches or
        find_notable_moments result) and want to re-inspect just that one
        match, or look up a different participant's stats within it. Returns
        None if the match or the player isn't found in it.

        Args:
            match_id: A Riot match ID, e.g. 'NA1_5607321601' -- typically taken from a prior get_ranked_matches or find_notable_moments result, not guessed.
            puuid: The target player's Riot puuid (not a summoner_name/tag_line). If you only have a Riot ID, use get_ranked_matches or get_complete_summoner_info instead -- they resolve the puuid internally.
            summoner_name: Riot ID game name of the player this lookup is for (the part before the #) -- used for local caching, not for finding the match.
            region: Riot region the match was played in, e.g. 'NA', 'EUW1', 'KR'.
            stats_level: 'personal' (default) returns only this player's champion, KDA, and win/loss. 'advanced' additionally returns every other participant in the match (their Riot ID, champion, team, role, KDA, and win/loss).
        """
        if isinstance(region, str):
            region = Region(region)
        try:
            match = self._get_cached_match_detail(match_id, summoner_name=summoner_name, region=region)
            if match is None:
                return None

            match_info = match["info"]
            personal_stats = None
            for participant in match_info["participants"]:
                if participant["puuid"] == puuid:
                    personal_stats = {
                        "champion": participant["championName"],
                        "kills": participant["kills"],
                        "deaths": participant["deaths"],
                        "assists": participant["assists"],
                        "win": participant["win"],
                    }
                    break

            if personal_stats is None:
                return None

            if stats_level == "personal":
                return personal_stats

            other_participants = [
                {
                    "summoner_name": p.get("riotIdGameName"),
                    "tag_line": p.get("riotIdTagline"),
                    "champion": p.get("championName"),
                    "team": "blue" if p.get("teamId") == 100 else "red",
                    "role": _TEAM_POSITION_TO_ROLE.get(p.get("teamPosition")),
                    "kills": p.get("kills"),
                    "deaths": p.get("deaths"),
                    "assists": p.get("assists"),
                    "win": p.get("win"),
                }
                for p in match_info["participants"]
                if p["puuid"] != puuid
            ]
            return {**personal_stats, "participants": other_participants}
        except Exception as e:
            self.error_traceback()
            return None

    def format_recent_matches(self, matches):
        formatted = []
        for match in matches:
            result = f"Champion: {match['champion']}, K/D/A: {match['kills']}/{match['deaths']}/{match['assists']}, Win: {'Yes' if match['win'] else 'No'}"
            formatted.append(result)
        return "\n".join(formatted)
