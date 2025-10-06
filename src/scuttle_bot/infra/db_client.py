import sqlite3
import json

from typing import Optional

class DatabaseClient:
    def __init__(self, db_path: str):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()

        self.create_table(
            "interactions",
            "id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, user_input TEXT, response TEXT"
        )

        self.create_table(
            "matches",
            "id varchar(100) PRIMARY KEY, data TEXT"
        )

    def create_table(self, table_name: str, schema: str):
        with self.connection:
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
            self.cursor.execute(query)

    def execute_query(self, query: str, params: tuple = ()):
        with self.connection:
            self.cursor.execute(query, params)
        
        return self.cursor.fetchall()
    
    def store_interaction(self, user_input: str, response: str):
        self.execute_query(
            "INSERT INTO interactions (user_input, response) VALUES (?, ?)",
            (user_input, response)
        )
    
    def store_match(self, match_id: str, data: str):
        self.execute_query(
            "INSERT OR IGNORE INTO matches (id, data) VALUES (?, ?)",
            (match_id, data)
        )

    def retrieve_match(self, match_id: str):
        result = self.execute_query(
            "SELECT data FROM matches WHERE id = ?",
            (match_id,)
        )[0]
        result = json.loads(result) if result else None
        return result
    
    def retrieve_all_matches(self, match_ids: list):
        placeholders = ','.join('?' for _ in match_ids)
        query = f"SELECT id, data FROM matches WHERE id IN ({placeholders})"
        results = self.execute_query(query, tuple(match_ids))
        return {match_id: json.loads(data) for match_id, data in results}
    
    def retrieve_all_interactions(self, username: Optional[str] = None):
        if username:
            results = self.execute_query("SELECT user_input, response FROM interactions WHERE user = ?", (username,))
        else:
            results = self.execute_query("SELECT user_input, response FROM interactions")
        return [{"user_input": user_input, "response": response} for user_input, response in results]

    def close(self):
        self.connection.close()

if __name__ == "__main__":
    db = DatabaseClient("../cache/scuttle_bot.db")
    db.create_table("matches", "id varchar(100) PRIMARY KEY, data TEXT")
    from src.scuttle_bot.service.service import ScuttleBotService
    service = ScuttleBotService(db)
