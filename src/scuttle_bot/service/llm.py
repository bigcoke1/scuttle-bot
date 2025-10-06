from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

from src.scuttle_bot.infra.db_client import DatabaseClient
from src.scuttle_bot.service.service import ScuttleBotService

class LLMService:
    def __init__(self, db: DatabaseClient):
        self.db = db
        self.service = ScuttleBotService(db=self.db)
        self.tools = [
            Tool(
                name="FetchSummonerInfo",
                description="Fetch detailed information about a League of Legends summoner. Requires region, game name, and tag line. Returns comprehensive summoner info including ranked stats, champion masteries, and recent matches.",
                func=self.service.get_complete_summoner_info
            ),
            Tool(
                name="FetchMatchData",
                description="Fetch match data using match ID. Requires match ID as input. Returns detailed match information.",
                func=self.service.get_ranked_stats
            ),
            Tool(
                name="FetchRecentMatches",
                description="Fetch recent matches for a summoner. Requires game name, and tag line. Optional parameters are count, start time and end time. Returns a list of recent match ids.",
                func=self.service.get_ranked_matches
            ),
            Tool(
                name="FetchChampionMasteries",
                description="Fetch top champion masteries for a summoner. Requires region, game name, and tag line. Returns a list of champion masteries.",
                func=self.service.get_top_champion_masteries
            )
        ]

        load_dotenv()
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                        temperature=0,
                                        max_tokens=None,
                                        timeout=None,
                                        max_retries=2,)
        self.llm.bind_tools(self.tools)

    def generate_response(self, user_input):
        # Example prompt template
        prompt = PromptTemplate(
            input_variables=["user_input"],
            template="You are a helpful assistant. Answer the following question: {user_input}"
        )
        formatted_prompt = prompt.format(user_input=user_input)
        
        # Here you would integrate with an actual LLM (e.g., OpenAI, Hugging Face)
        # For demonstration, we'll return a mock response
        response = self.llm.invoke(input=formatted_prompt)
        if response:
            response = str(response.content)
            self.db.store_interaction(user_input, response)
        
        return response