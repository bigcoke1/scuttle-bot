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
    
    blue_ban_0 TEXT,
    blue_ban_1 TEXT,
    blue_ban_2 TEXT,
    blue_ban_3 TEXT,
    blue_ban_4 TEXT,
    red_ban_0 TEXT,
    red_ban_1 TEXT,
    red_ban_2 TEXT,
    red_ban_3 TEXT,
    red_ban_4 TEXT,
    
    game_duration INTEGER,
    queue_id INTEGER,
    
    cached_at DATETIME DEFAULT CURRENT_TIMESTAMP
);