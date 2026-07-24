from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Literal

import os
from dotenv import load_dotenv
import logging
from typing import Optional

from src.scuttle_bot.infra.db_client import DatabaseClient
from src.scuttle_bot.service.service import ScuttleBotService
from scuttle_bot.utilities.schemas import Region
from scuttle_bot.llm.system_prompts import build_system_prompt, FORCE_FINAL_ANSWER_PROMPT
from src.scuttle_bot.data.collector import Collector
from src.scuttle_bot.ml.predictor import WinPredictor


class PlayerDraftEntry(BaseModel):
    team: Literal["blue", "red"] = Field(description="Which side this player is on")
    role: Literal["top", "jungle", "mid", "adc", "support"] = Field(description="Lane/role this player is playing")
    champion: str = Field(description="Champion this player is picking, e.g. 'Ahri' or 'Kai'Sa'")
    summoner_name: str = Field(description="Riot ID game name (the part before the #), used to look up this player's live rank and champion mastery")
    tag_line: str = Field(description="Riot ID tagline (the part after the #)")


class LLMService:
    def __init__(self, db: DatabaseClient):
        self.db = db
        self.service = ScuttleBotService(db=self.db)
        self.predictor = WinPredictor()
        self.last_tool_calls = []
        self.tools = [
            self.service.get_complete_summoner_info,
            self.service.search_summoner,
            self.service.get_top_champion_masteries,
            self.service.get_ranked_matches,
            self.service.get_match_stats,
            self.service.get_active_game,
            self.predict_win_probability,
            self.service.register_user,
            self.service.get_registered_user,
            self.service.unregister_user,
            self.service.list_available_personalities,
            self.service.select_personality,
            self.service.set_custom_personality,
            self.service.remove_personality,
            self.service.analyze_performance_trend,
            self.service.find_notable_moments,
        ]

        load_dotenv()
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                        google_api_key=os.getenv("GEMINI_API_KEY"),
                                        temperature=0.7,
                                        max_tokens=None,
                                        timeout=None,
                                        max_retries=2,).bind_tools(self.tools)

    def _resolve_region(self, region: str) -> Region:
        try:
            return Region[region.upper()]
        except KeyError:
            try:
                return Region(region.lower())
            except ValueError:
                return Region.NA

    def predict_win_probability(self, players: list[PlayerDraftEntry], region: str = "NA", patch_version: str = "15.1") -> str:
        """
        Predicts the win probability for a 5v5 League of Legends ranked
        solo/duo draft, using each player's champion pick plus their live
        ranked tier/rank and champion-specific mastery.

        `players` must contain exactly 10 entries: one blue-side and one
        red-side player for each of the 5 roles (top, jungle, mid, adc,
        support). Look up each player by their Riot ID (summoner_name#tag_line)
        -- do not guess or make up rank or mastery values.

        Always pass champion names through exactly as given, even ones you
        don't personally recognize -- the game's champion roster is live and
        may include champions released after your training data. This tool
        validates each name against the current roster itself and will
        return a clear error if one is genuinely invalid, so do not refuse
        or ask the user to double-check a champion name yourself.
        """
        try:
            if len(players) != 10:
                return f"Error: expected exactly 10 players (5 per side), got {len(players)}."

            region_enum = self._resolve_region(region)
            collector = Collector(region_enum)

            seen_slots = set()
            player_inputs = {}
            unresolved = []

            for entry in players:
                if isinstance(entry, dict):
                    entry = PlayerDraftEntry(**entry)

                slot = f"{entry.team}_{entry.role}"
                if slot in seen_slots:
                    return f"Error: duplicate entry for {slot} -- each of the 10 team/role slots must appear exactly once."
                seen_slots.add(slot)

                if self.predictor.resolve_champion_id(entry.champion) is None:
                    return f"Error: unknown champion {entry.champion!r} for {entry.summoner_name}#{entry.tag_line}."

                tier = rank = None
                wins = losses = 0
                champion_points = champion_level = None

                puuid = self.service.get_puuid(entry.summoner_name, entry.tag_line, region=region_enum)
                if puuid is None:
                    unresolved.append(f"{entry.summoner_name}#{entry.tag_line}")
                else:
                    ranked_entries = collector.collect_ranked_stats(puuid) or []
                    for ranked_entry in ranked_entries:
                        if ranked_entry.get("queueType") == "RANKED_SOLO_5x5":
                            tier = ranked_entry.get("tier")
                            rank = ranked_entry.get("rank")
                            wins = ranked_entry.get("wins", 0)
                            losses = ranked_entry.get("losses", 0)
                            break

                    champ_id = self.predictor.resolve_champion_id(entry.champion)
                    mastery = collector.collect_champion_mastery(puuid, champ_id) or {}
                    champion_points = mastery.get("championPoints")
                    champion_level = mastery.get("championLevel")

                player_inputs[slot] = {
                    "champion": entry.champion,
                    "tier": tier,
                    "rank": rank,
                    "wins": wins,
                    "losses": losses,
                    "champion_points": champion_points,
                    "champion_level": champion_level,
                }

            missing_slots = [
                f"{team}_{role}"
                for team in ("blue", "red")
                for role in ("top", "jungle", "mid", "adc", "support")
                if f"{team}_{role}" not in player_inputs
            ]
            if missing_slots:
                return f"Error: missing players for slots: {missing_slots}."

            blue_win_probability = self.predictor.predict(player_inputs, patch_version=patch_version)

            result = (
                f"Blue side win probability: {blue_win_probability:.1%}. "
                f"Red side win probability: {1 - blue_win_probability:.1%}. "
                f"(Model: RandomForest, draft + player stats, ~61% historical test accuracy.)"
            )
            if unresolved:
                result += f" Note: could not find live stats for {', '.join(unresolved)}; used default/average values for them instead."
            return result
        except Exception as e:
            self.service.error_traceback()
            return f"Error: {str(e)}"

    MAX_TOOL_ITERATIONS = 5
    DEFAULT_HISTORY_LIMIT = 5

    def generate_response(self, user_input, discord_id: Optional[str] = None, history_limit: int = DEFAULT_HISTORY_LIMIT) -> str:
        personality = self.db.retrieve_personality_setting(discord_id) if discord_id else None
        history = self.db.retrieve_recent_interactions(discord_id, limit=history_limit) if discord_id else []
        try:
            messages: list[BaseMessage] = [SystemMessage(content=build_system_prompt(personality=personality, discord_id=discord_id))]
            for turn in history:
                messages.append(HumanMessage(content=turn["query"]))
                messages.append(AIMessage(content=turn["response"]))

            messages.append(HumanMessage(content=user_input))

            tool_calls_log = []
            response: AIMessage = self.llm.invoke(messages) # type: ignore
            messages.append(response)

            iterations = 0
            while response.tool_calls and iterations < self.MAX_TOOL_ITERATIONS:
                iterations += 1
                print(f"Tools used (round {iterations}): {response.tool_calls}")

                for tool in response.tool_calls:
                    tool_name = tool["name"]
                    tool_args = tool["args"]
                    call_id = tool["id"]

                    tool_func = None
                    for t in self.tools:
                        actual_name = getattr(t, "name", getattr(t, "__name__", None))
                        if actual_name == tool_name:
                            tool_func = t
                            break

                    if tool_func:
                        if hasattr(tool_func, "invoke"):
                            observation = tool_func.invoke(tool_args)
                        else:
                            observation = tool_func(**tool_args)
                    else:
                        observation = f"Error: Tool {tool_name} not found."

                    print(f"Tool {tool_name} returned observation: {observation}")
                    tool_calls_log.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "observation": str(observation),
                    })
                    messages.append(ToolMessage(
                        content=str(observation),
                        tool_call_id=call_id
                    ))

                response: AIMessage = self.llm.invoke(messages) # type: ignore
                messages.append(response)

            if response.tool_calls:
                # Hit MAX_TOOL_ITERATIONS while still requesting tools -- force a
                # wrap-up answer from whatever's been gathered so far instead of
                # looping indefinitely.
                messages.append(HumanMessage(content=FORCE_FINAL_ANSWER_PROMPT))
                response: AIMessage = self.llm.invoke(messages) # type: ignore

            import datetime
            import json

            os.makedirs("src/scuttle_bot/logs", exist_ok=True)
            with open("src/scuttle_bot/logs/llm_logs.txt", "a") as log_file:
                log_file.write(f"{datetime.datetime.now()} - User Input: {user_input}\n")
                if tool_calls_log:
                    log_file.write(f"{datetime.datetime.now()} - Tool Calls:\n")
                    for call in tool_calls_log:
                        log_file.write(f"    Tool: {call['tool']}\n")
                        log_file.write(f"    Args: {json.dumps(call['args'], indent=2, default=str)}\n")
                        log_file.write(f"    Observation: {call['observation']}\n\n")
                else:
                    log_file.write(f"{datetime.datetime.now()} - Tool Calls: none\n")
                log_file.write(f"{datetime.datetime.now()} - Response Metadata: {response.additional_kwargs}\n")
                log_file.write(f"{datetime.datetime.now()} - Final Response: {response.content}\n\n")

            text_response = str(response.content)

            # Exposed for callers (tests, debugging) that need to inspect which
            # tools were actually used without scraping the log file.
            self.last_tool_calls = tool_calls_log

            if discord_id:
                self.db.store_interaction(user_input=user_input, response=text_response, user_id=discord_id)
            return text_response

        except Exception as e:
            logging.error(f"Error generating LLM response: {e}")
            return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    try:
        load_dotenv()
        db_path = os.getenv("DB_PATH", "src/scuttle_bot/cache/scuttle_bot.db")
        db_client = DatabaseClient(db_path)
        llm_service = LLMService(db=db_client)
        user_query = input("Enter your query: ") or "What's Sorrrymakerrr#DOINB top 10 champion masteries?"
        print(llm_service.generate_response(user_query))
    except Exception as e:
        print(f"An error occurred: {e}")