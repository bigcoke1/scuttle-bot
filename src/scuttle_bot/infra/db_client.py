import os
import sqlite3
import json
from pathlib import Path

from typing import Optional

class DatabaseClient:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection = None
        self._cursor = None
        self._initialize_db()
    
    @property
    def connection(self):
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
        return self._connection
    
    @property
    def cursor(self):
        if self._cursor is None:
            self._cursor = self.connection.cursor()
        return self._cursor
    
    def _initialize_db(self):
        """Initialize database from schema if it doesn't exist"""
        db_exists = os.path.exists(self.db_path)
        
        if not db_exists:
            # Create directory if needed
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Load and execute schema
            schema_path = Path(__file__).parent.parent / "cache" / "schema.sql"
            with open(schema_path, 'r') as f:
                schema = f.read()
            
            conn = sqlite3.connect(self.db_path)
            conn.executescript(schema)
            conn.commit()
            conn.close()

    def execute_query(self, query: str, params: tuple = ()):
        with self.connection:
            self.cursor.execute(query, params)
        
        return self.cursor.fetchall()
    
    def store_interaction(self, user_input: str, response: str, user_id: Optional[str] = None):
        self.execute_query(
            "INSERT INTO interactions (user_id, query, response) VALUES (?, ?, ?)",
            (user_id, user_input, response)
        )
    
    def store_match(self, match_id: str, summoner_name: str, data: str):
        self.execute_query(
            "INSERT OR IGNORE INTO matches (match_id, summoner_name, data) VALUES (?, ?, ?)",
            (match_id, summoner_name, data)
        )

    def retrieve_match(self, match_id: str):
        result = self.execute_query(
            "SELECT data FROM matches WHERE match_id = ?",
            (match_id,)
        )
        if result:
            return json.loads(result[0][0])
        return None
    
    def exists_match(self, match_id: str) -> bool:
        result = self.execute_query(
            "SELECT 1 FROM matches WHERE match_id = ?",
            (match_id,)
        )
        return len(result) > 0
    
    def retrieve_all_matches(self, match_ids: list):
        placeholders = ','.join('?' for _ in match_ids)
        query = f"SELECT match_id, data FROM matches WHERE match_id IN ({placeholders})"
        results = self.execute_query(query, tuple(match_ids))
        return {match_id: json.loads(data) for match_id, data in results}
    
    def retrieve_all_interactions(self, user_id: Optional[str] = None):
        if user_id:
            results = self.execute_query("SELECT query, response FROM interactions WHERE user_id = ?", (user_id,))
        else:
            results = self.execute_query("SELECT query, response FROM interactions")
        return [{"query": query, "response": response} for query, response in results]
    
    def store_personality_setting(self, user_id: str, personality: str):
        self.execute_query(
            "INSERT OR REPLACE INTO user_preferences (user_id, personality) VALUES (?, ?)",
            (user_id, personality)
        )

    def retrieve_personality_setting(self, user_id: str) -> Optional[str]:
        result = self.execute_query(
            "SELECT personality FROM user_preferences WHERE user_id = ?",
            (user_id,)
        )
        return result[0][0] if result else None

    def close(self):
        self.connection.close()

if __name__ == "__main__":
    db = DatabaseClient(os.getenv("DB_PATH", "src/scuttle_bot/cache/scuttle_bot.db"))
    db.retrieve_match("NA1_5424853876")
