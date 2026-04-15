CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    
    patch_version TEXT NOT NULL,
    
    average_tier INTEGER,
    
    blue_win INTEGER NOT NULL,
    
    blue_top TEXT,
    blue_jungle TEXT,
    blue_mid TEXT,
    blue_adc TEXT,
    blue_support TEXT,
    
    red_top TEXT,
    red_jungle TEXT,
    red_mid TEXT,
    red_adc TEXT,
    red_support TEXT,
    
    blue_bans TEXT,
    red_bans TEXT,
    
    game_duration INTEGER,
    queue_id INTEGER,
    
    cached_at DATETIME DEFAULT CURRENT_TIMESTAMP
);