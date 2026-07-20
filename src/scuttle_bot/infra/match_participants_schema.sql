CREATE TABLE IF NOT EXISTS match_participants (
    match_id TEXT NOT NULL,
    puuid TEXT NOT NULL,

    team TEXT NOT NULL,
    role TEXT NOT NULL,
    champion_id INTEGER NOT NULL,

    tier TEXT,
    rank TEXT,
    league_points INTEGER,
    wins INTEGER,
    losses INTEGER,
    win_rate REAL,

    champion_points INTEGER,
    champion_level INTEGER,
    champion_last_play_time INTEGER,

    PRIMARY KEY (match_id, puuid),
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);
