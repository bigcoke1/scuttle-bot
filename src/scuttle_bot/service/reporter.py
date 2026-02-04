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
            report = self.llm_service.generate_response(f"Retrieve matches played by {user['summoner_name']}#{user['tag_line']}, puuid: {user['puuid']} in the last 24 hours and summarize their performance. \n"
                                                        f"Start time: {int(time.time()) - 86400}, max match count: 20")
            reports.append({"user": user['discord_id'], "report": report})
            time.sleep(10)
        return reports
        