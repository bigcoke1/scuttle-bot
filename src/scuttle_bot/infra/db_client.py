import os
import sqlite3
import json
from pathlib import Path

from typing import Optional

class DatabaseClient:
    def __init__(self, db_path: str, sql_script_path: Optional[str] = None):
        self.db_path = db_path
        self.sql_script_path = sql_script_path
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
        """Create the DB directory if needed, then run the schema. Every
        statement in schema.sql is CREATE TABLE IF NOT EXISTS, so running it
        against an existing DB is a no-op for tables that already exist and
        adds any new ones -- this is what lets new tables (e.g.
        match_timelines) show up in already-deployed databases without a
        separate migration step."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.run_script()

    def run_script(self):
        if self.sql_script_path:
            schema_path = Path(self.sql_script_path)
        else:
            schema_path = Path("src/scuttle_bot/infra/schema.sql")
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
    
    def store_match_timeline(self, match_id: str, data: str):
        self.execute_query(
            "INSERT OR IGNORE INTO match_timelines (match_id, data) VALUES (?, ?)",
            (match_id, data)
        )

    def retrieve_match_timeline(self, match_id: str):
        result = self.execute_query(
            "SELECT data FROM match_timelines WHERE match_id = ?",
            (match_id,)
        )
        if result:
            return json.loads(result[0][0])
        return None

    def exists_match_timeline(self, match_id: str) -> bool:
        result = self.execute_query(
            "SELECT 1 FROM match_timelines WHERE match_id = ?",
            (match_id,)
        )
        return len(result) > 0

    def retrieve_all_matches(self, match_ids: list):
        placeholders = ','.join('?' for _ in match_ids)
        query = f"SELECT match_id, data FROM matches WHERE match_id IN ({placeholders})"
        results = self.execute_query(query, tuple(match_ids))
        return {match_id: json.loads(data) for match_id, data in results}
    
    def retrieve_recent_interactions(self, user_id: str, limit: int = 5) -> list:
        """Most recent interactions for this user, oldest first (ready to
        replay as conversation history)."""
        results = self.execute_query(
            "SELECT query, response FROM interactions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        )
        return [{"query": query, "response": response} for query, response in reversed(results)]

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

    def delete_personality_setting(self, user_id: str) -> bool:
        existing = self.execute_query(
            "SELECT 1 FROM user_preferences WHERE user_id = ?",
            (user_id,)
        )
        if not existing:
            return False  # Nothing to remove

        self.execute_query(
            "DELETE FROM user_preferences WHERE user_id = ?",
            (user_id,)
        )
        return True


    def get_all_registered_users(self):
        results = self.execute_query("SELECT * FROM registered_users")
        return [{"discord_id": row[0], "summoner_name": row[1], "tag_line": row[2], "game_region": row[3], "puuid": row[4]} for row in results]
    
    def register_user(self, discord_id: str, summoner_name: str, tag_line: str, region: str, puuid: str) -> bool:
        existing = self.execute_query(
            "SELECT 1 FROM registered_users WHERE discord_id = ?",
            (discord_id,)
        )
        if existing:
            return False  # User already registered
        
        self.execute_query(
            "INSERT INTO registered_users (discord_id, summoner_name, tag_line, game_region, puuid) VALUES (?, ?, ?, ?, ?)",
            (discord_id, summoner_name, tag_line, region, puuid)
        )
        return True
    
    def unregister_user(self, discord_id: str) -> bool:
        existing = self.execute_query(
            "SELECT 1 FROM registered_users WHERE discord_id = ?",
            (discord_id,)
        )
        if not existing:
            return False  # Nothing to unregister

        self.execute_query(
            "DELETE FROM registered_users WHERE discord_id = ?",
            (discord_id,)
        )
        return True

    def get_registered_user(self, discord_id: str) -> Optional[dict]:
        result = self.execute_query(
            "SELECT summoner_name, tag_line, game_region, puuid FROM registered_users WHERE discord_id = ?",
            (discord_id,)
        )
        if result:
            summoner_name, tag_line, game_region, puuid = result[0]
            return {
                "discord_id": discord_id,
                "summoner_name": summoner_name,
                "tag_line": tag_line,
                "game_region": game_region,
                "puuid": puuid
            }
        return None

    def close(self):
        self.connection.close()

if __name__ == "__main__":
    db = DatabaseClient(os.getenv("DB_PATH", "src/scuttle_bot/cache/scuttle_bot.db"))
    db.run_script()
