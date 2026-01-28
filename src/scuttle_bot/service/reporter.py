import schedule
import time
from src.scuttle_bot.service.llm import LLMService

class Reporter:

    def __init__(self, db_client, llm_service: LLMService):
        self.db_client = db_client
        self.llm_service = llm_service
        
    def generate_report(self):
        users = self.db_client.get_all_registered_users()
        reports = []
        for user in users:
            time.sleep(10) 
            report = self.llm_service.generate_response(f"Retrieve matches played by {user['game_name']}#{user['game_tag']} in the last 24 hours and summarize their performance.")
            reports.append({"user": user['discord_id'], "report": report})
        return reports
        