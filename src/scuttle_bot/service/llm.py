from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage, AIMessage

import os
from dotenv import load_dotenv
import logging
from typing import Optional

from src.scuttle_bot.infra.db_client import DatabaseClient
from src.scuttle_bot.service.service import ScuttleBotService

class LLMService:
    def __init__(self, db: DatabaseClient):
        self.db = db
        self.service = ScuttleBotService(db=self.db)
        self.tools = [
            self.service.get_complete_summoner_info,
            self.service.search_summoner,
            self.service.get_top_champion_masteries,
            self.service.get_ranked_matches,
        ]

        load_dotenv()
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                        temperature=0.7,
                                        max_tokens=None,
                                        timeout=None,
                                        max_retries=2,).bind_tools(self.tools)

    def generate_response(self, user_input, username: Optional[str] = None) -> str:
        personality = self.db.retrieve_personality_setting(username) if username else None
        try:
            messages: list[BaseMessage] = [
                HumanMessage(content=f"""You are a League of Legends expert. Answer the following question: {user_input}. 
                            If region is not specified, assume NA. 
                            If a personality has been set for you, use it in your response.
                            Personality: {personality if personality else 'No specific personality set'}"""),
                HumanMessage(content="""You have access to the following tools to help you gather information:
                            Once you have the data from the tool, summarize it for the user in a friendly way. Do not ask for the same information twice""")
            ]
            
            response: AIMessage = self.llm.invoke(messages) # type: ignore
            messages.append(response)
            # print(f"Initial LLM response: {response}")
            tools_used = response.tool_calls
            print(f"Tools used: {tools_used}")

            if tools_used is not None and len(tools_used) > 0:
                for tool in tools_used:
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
                    messages.append(ToolMessage(
                        content=str(observation),
                        tool_call_id=call_id
                    ))
                    # print(f"Updated messages: {messages}")

                messages.append(HumanMessage(content="Based on the tool outputs, provide a final answer to the user. DO NOT CALL ANY MORE TOOLS."))
                response: AIMessage = self.llm.invoke(messages) # type: ignore
                # print(f"Final LLM response after tool usage: {response}")
            
            import datetime
            
            with open("src/scuttle_bot/logs/llm_logs.txt", "a") as log_file:
                log_file.write(f"{datetime.datetime.now()} - User Input: {user_input}\n")
                log_file.write(f"{datetime.datetime.now()} - Response Metadata: {response.additional_kwargs}\n")
                log_file.write(f"{datetime.datetime.now()} - Final Response: {response.content}\n\n")

            text_response = str(response.content)

            if username:
                self.db.store_interaction(user_input=user_input, response=text_response, user_id=username)
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
        user_query = "What's Sorrrymakerrr#DOINB top 10 champion masteries?"
        print(llm_service.generate_response(user_query))
    except Exception as e:
        print(f"An error occurred: {e}")