"""Personal-coach style analysis on top of match timelines: comparing a
player's checkpoint stats against their lane opponent across recent games,
and flagging notable in-game moments (e.g. a sudden gold spike) with a
best-effort explanation drawn from nearby match events.

MatchAnalyzerMixin is mixed into ScuttleBotService alongside RiotClientMixin,
whose get_puuid / _fetch_match_ids / _get_cached_match_detail /
get_match_timeline it calls via self. All the heavy lifting (fetching,
caching, aggregating) happens here in Python -- these methods return small,
precomputed summaries rather than raw timeline data, since a raw timeline is
far too large and numeric for an LLM to reliably reason over directly.
"""

from typing import Literal, Optional

from scuttle_bot.utilities.schemas import Region

# Match-v5 timeline field each supported metric reads from a participant's
# frame snapshot. "cs" isn't a single field -- it's the sum of lane and
# jungle minions, so it's handled separately in _extract_metric.
_METRIC_FIELDS = {
    "gold": "totalGold",
    "xp": "xp",
}


class MatchAnalyzerMixin:
    def analyze_performance_trend(
        self,
        summoner_name: str,
        tag_line: str,
        region: Region,
        metric: Literal["gold", "xp", "cs"],
        checkpoint_minutes: int = 15,
        num_matches: int = 10,
    ) -> dict:
        """Compares a player's <metric> at a fixed point in the game against
        their lane opponent, averaged over their last num_matches ranked
        matches. Use this to answer coaching questions like "am I behind on
        gold at 15 minutes" -- it computes the numeric comparison itself, so
        don't try to estimate or recompute it from raw match data yourself.
        Only matches where both this player and a same-role opponent are
        identifiable count towards num_matches; others are silently skipped,
        so matches_used in the result may be lower than num_matches.

        Args:
            summoner_name: Riot ID game name (the part before the #).
            tag_line: Riot ID tagline (the part after the #).
            region: Riot region the account is registered in, e.g. 'NA', 'EUW1', 'KR'.
            metric: Which stat to compare -- 'gold' (total gold), 'xp' (experience), or 'cs' (lane + jungle minion kills).
            checkpoint_minutes: The in-game minute to compare at, e.g. 15.
            num_matches: How many of the player's most recent ranked matches to average over.
        """
        try:
            if isinstance(region, str):
                region = Region(region)

            puuid = self.get_puuid(summoner_name, tag_line, region=region)
            if puuid is None:
                return {"error": f"Could not resolve {summoner_name}#{tag_line}."}

            # Over-fetch match IDs since matches without a resolvable lane
            # opponent (e.g. no teamPosition data) get skipped below.
            match_ids = self._fetch_match_ids(puuid, region=region, count=num_matches * 2)
            if not match_ids:
                return {"error": "No recent ranked matches found."}

            target_ms = checkpoint_minutes * 60 * 1000
            player_values = []
            opponent_values = []

            for match_id in match_ids:
                if len(player_values) >= num_matches:
                    break

                match_detail = self._get_cached_match_detail(match_id, summoner_name=summoner_name, region=region)
                if match_detail is None:
                    continue

                participants = match_detail["info"]["participants"]
                me = next((p for p in participants if p["puuid"] == puuid), None)
                if me is None or not me.get("teamPosition"):
                    continue
                opponent = next(
                    (p for p in participants if p.get("teamPosition") == me["teamPosition"] and p["teamId"] != me["teamId"]),
                    None,
                )
                if opponent is None:
                    continue

                timeline = self.get_match_timeline(match_id, region=region)
                if timeline is None:
                    continue

                frame = self._nearest_frame(timeline, target_ms)
                if frame is None:
                    continue

                participant_frames = frame["participantFrames"]
                my_frame = participant_frames.get(str(me["participantId"]))
                opponent_frame = participant_frames.get(str(opponent["participantId"]))
                if my_frame is None or opponent_frame is None:
                    continue

                player_values.append(self._extract_metric(my_frame, metric))
                opponent_values.append(self._extract_metric(opponent_frame, metric))

            matches_used = len(player_values)
            if matches_used == 0:
                return {"error": "Not enough data: none of the recent matches had a resolvable lane opponent with timeline data at that checkpoint."}

            player_avg = sum(player_values) / matches_used
            opponent_avg = sum(opponent_values) / matches_used
            delta_pct = ((player_avg - opponent_avg) / opponent_avg * 100) if opponent_avg else None

            return {
                "metric": metric,
                "checkpoint_minutes": checkpoint_minutes,
                "matches_used": matches_used,
                "player_avg": round(player_avg, 1),
                "lane_opponent_avg": round(opponent_avg, 1),
                "delta_pct_vs_opponent": round(delta_pct, 1) if delta_pct is not None else None,
            }
        except Exception as e:
            self.error_traceback()
            return {"error": str(e)}

    def find_notable_moments(
        self,
        summoner_name: str,
        tag_line: str,
        region: Region,
        num_matches: int = 3,
        min_gold_jump: int = 1500,
        max_moments: int = 5,
    ) -> list[dict]:
        """Scans a player's most recent ranked matches for sudden gold jumps
        between consecutive timeline frames (~60s apart), and explains each
        one using nearby match events -- a jump almost always means a kill,
        an objective take, or a big minion wave. Use this to answer "what
        were the big moments in my recent games" or to find a timestamp
        worth pointing the user to in a replay. Returns at most max_moments
        entries, biggest gold swing first -- don't ask for more than a few
        unless the user specifically wants an exhaustive list, since each
        entry carries a long signed replay URL. If no jump in any scanned
        match clears min_gold_jump, falls back to returning just the single
        biggest gold swing found (flagged with below_threshold: true) instead
        of an empty list, so there's still something to point to.

        Each moment includes a replay_url when Riot still has that match's
        replay available -- pass it straight to the user along with
        timestamp_seconds (the point in the replay to skip to), but always
        mention replay_expires_at too, since these download links are
        temporary and go dead after about an hour.

        Args:
            summoner_name: Riot ID game name (the part before the #).
            tag_line: Riot ID tagline (the part after the #).
            region: Riot region the account is registered in, e.g. 'NA', 'EUW1', 'KR'.
            num_matches: How many of the player's most recent ranked matches to scan.
            min_gold_jump: Minimum gold gained between two consecutive frames to count as "notable".
            max_moments: Maximum number of moments to return, biggest gold swing first.
        """
        try:
            if isinstance(region, str):
                region = Region(region)

            puuid = self.get_puuid(summoner_name, tag_line, region=region)
            if puuid is None:
                return [{"error": f"Could not resolve {summoner_name}#{tag_line}."}]

            match_ids = self._fetch_match_ids(puuid, region=region, count=num_matches)
            if not match_ids:
                return []

            # Fetched once up front for all matches -- these expire quickly,
            # so there's no point caching them like match/timeline data.
            replay_urls = self.get_replay_urls(puuid, region=region)

            moments = []
            biggest_moment = None
            for match_id in match_ids:
                match_detail = self._get_cached_match_detail(match_id, summoner_name=summoner_name, region=region)
                if match_detail is None:
                    continue

                me = next((p for p in match_detail["info"]["participants"] if p["puuid"] == puuid), None)
                if me is None:
                    continue
                participant_id = me["participantId"]

                timeline = self.get_match_timeline(match_id, region=region)
                if timeline is None:
                    continue

                replay = replay_urls.get(match_id)

                prev_gold = None
                for frame in timeline["info"]["frames"]:
                    participant_frame = frame["participantFrames"].get(str(participant_id))
                    if participant_frame is None:
                        continue

                    gold = participant_frame.get("totalGold")
                    if prev_gold is not None and gold is not None:
                        gold_gained = gold - prev_gold
                        moment = {
                            "match_id": match_id,
                            "timestamp_seconds": frame["timestamp"] // 1000,
                            "gold_gained": gold_gained,
                            "likely_cause": self._explain_jump(frame, participant_id),
                            "replay_url": replay["url"] if replay else None,
                            "replay_expires_at": replay["expires_at"] if replay else None,
                        }

                        if gold_gained >= min_gold_jump:
                            moments.append(moment)
                        if biggest_moment is None or gold_gained > biggest_moment["gold_gained"]:
                            biggest_moment = moment
                    prev_gold = gold

            if moments:
                moments.sort(key=lambda m: m["gold_gained"], reverse=True)
                return moments[:max_moments]
            if biggest_moment is not None:
                return [{**biggest_moment, "below_threshold": True}]
            return []
        except Exception as e:
            self.error_traceback()
            return [{"error": str(e)}]

    def _nearest_frame(self, timeline: dict, target_ms: int) -> Optional[dict]:
        frames = timeline.get("info", {}).get("frames") or []
        if not frames:
            return None
        return min(frames, key=lambda f: abs(f["timestamp"] - target_ms))

    def _extract_metric(self, participant_frame: dict, metric: str) -> float:
        if metric == "cs":
            return participant_frame.get("minionsKilled", 0) + participant_frame.get("jungleMinionsKilled", 0)
        return participant_frame.get(_METRIC_FIELDS[metric], 0)

    def _explain_jump(self, frame: dict, participant_id: int) -> str:
        causes = []
        for event in frame.get("events", []):
            involved = {event.get("killerId"), event.get("participantId"), *(event.get("assistingParticipantIds") or [])}
            if participant_id not in involved:
                continue

            event_type = event.get("type")
            if event_type == "CHAMPION_KILL":
                causes.append("champion kill")
            elif event_type == "ELITE_MONSTER_KILL":
                causes.append(f"{(event.get('monsterType') or 'epic monster').lower()} kill")
            elif event_type == "BUILDING_KILL":
                causes.append(f"{(event.get('buildingType') or 'building').lower()} destroyed")

        return ", ".join(causes) if causes else "no kill/objective event found -- likely a big minion wave or passive gold"
