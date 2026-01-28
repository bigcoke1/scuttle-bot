CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    summoner_name TEXT NOT NULL,
    data TEXT NOT NULL,
    cached_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_preferences (
    user_id TEXT PRIMARY KEY,
    personality TEXT DEFAULT 'friendly',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS registered_users (
    discord_id TEXT PRIMARY KEY,
    summoner_name TEXT NOT NULL,
    game_tag TEXT NOT NULL,
    game_region TEXT NOT NULL,
    registered_at DATETIME DEFAULT CURRENT_TIMESTAMP
);