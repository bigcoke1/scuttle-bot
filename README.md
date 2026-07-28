A Discord bot that fetches League of Legends summoner data, analyzes it with Google's Gemini LLM, and predicts match win probability using models trained on real high-elo match data.

## Features

- **Summoner Lookup**: Fetch detailed information about League of Legends summoners including ranked stats, champion masteries, and recent match history
- **Live Game Lookup**: Detect a player's in-progress ranked game (via Riot's Spectator-v5 API) and infer each pick's role, so a game can be looked up without typing out the draft by hand
- **Win Probability Prediction**: Predict a draft's outcome from champion picks and each player's live rank/mastery, served by a model trained on real high-elo matches. Handles streamer-mode players (identity hidden but champion visible) by keeping the real pick and defaulting only their stats
- **Match Analysis / Personal Coach**: Compare a player's gold/XP/CS at a fixed in-game minute against their lane opponent across recent games, and surface notable moments (sudden gold swings, with a likely cause) plus a replay download link and the timestamp to jump to
- **AI-Powered Analysis**: Uses Google Gemini with tool-calling to gather data (summoner stats, live games, win predictions, match analysis) across multiple steps and summarize it conversationally
- **Conversation Memory**: Remembers a limited number of each Discord user's recent messages, so follow-up questions ("what's their win probability?") don't require repeating context
- **Customizable Personality**: Users can pick a predefined bot personality or describe a custom one in their own words, and clear it later
- **Daily Reports**: Registered users receive a daily DM summarizing their recent ranked performance
- **Match Caching**: Stores match details and timelines locally to reduce API calls and improve response times
- **Interaction Logging**: Records user interactions, the tool calls the LLM made to answer them, and the final responses for analysis and debugging

## Prerequisites

- Python 3.8+
- Discord bot token
- Riot API key
- Gemini API key
- (Optional) AWS credentials, for the Riot key in Secrets Manager and S3 backups of the databases and model weights

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd scuttle-bot
```

2. Install dependencies:
```bash
rye sync
```

3. Create a `.env` file in the root directory with your credentials:
```env
DISCORD_TOKEN=your_discord_bot_token
RIOT_API_KEY=your_riot_api_key
GEMINI_API_KEY=your_gemini_api_key
DB_PATH=src/scuttle_bot/cache/scuttle_bot.db
```

## Usage

### Running the Bot

```bash
rye run python -m src.scuttle_bot.service.bot
```

### Discord Commands

- `$hello` - Greet the bot
- `$help` - Show available commands
- `$stats <summoner_name>#<tag_line> <region>` - Fetch ranked stats for a summoner
- `$register <summoner_name>#<tag_line> <region>` - Register for daily match performance reports
- `$chat <message>` - Chat with the bot. Looks up summoners, detects live games, predicts win probability, analyzes recent performance, and manages your personality/registration -- e.g. `$chat is Faker#KR1 in a game right now, and if so who's favored to win?`. Remembers recent messages, so natural follow-ups work without repeating context. (In a DM you can skip the `$chat` prefix and just type.)
- `$personality` - Select a predefined personality for the bot
- `$goodbye` - Shut down the bot (admin only)

## Machine Learning

Win probability is predicted by models trained on real matches collected from high-elo (Master+) NA solo/duo queue games, stored in `src/scuttle_bot/cache/ml_dataset.db`.

### Feature set

Several feature sets were evaluated historically -- draft only, draft + average lobby tier, draft + individual player stats, and draft + player stats + bans. **Draft + individual player stats** (rank, win rate, champion mastery; no bans) was the clear winner across every model type, so it is the only feature set trained now, referred to in the code as **variant C**. All model types share the same `FeatureEncoder`.

### Model types and selection

Three model families are trained on variant C: logistic regression, random forest, and a neural network. For each, a **greedy (coordinate-descent) hyperparameter search** (`src/scuttle_bot/ml/greedy_search.py`) tunes the hyperparameters -- optimizing one at a time against 3-fold mean accuracy, which costs the sum of the candidate counts rather than the full grid's product -- and the winning config is then refit across 5 random splits for a mean +/- stdev estimate (see each `models/C/cv_summary.json`).

After tuning, **logistic regression (~0.647 mean accuracy)** edged out the random forest (~0.628) and the neural network (~0.633) on the identical feature set, and is the model the bot currently serves -- see `src/scuttle_bot/ml/predictor.py`.

### Training

```bash
rye run python -m scuttle_bot.ml.logistic.train
rye run python -m scuttle_bot.ml.rf.train
rye run python -m scuttle_bot.ml.nn.train
```

Each runs the greedy search, refits the best config, and writes trained models, configs, confusion-matrix plots, and a `cv_summary.json` under that model type's `models/C/` and `plots/C/` directories.

### Model storage

The served model (logistic regression) and all small artifacts (encoders, scalers, configs, summaries) are tracked in git, so the bot runs from a clean checkout. The heavier auxiliary weights (random forest `.pkl` files at ~72 MB each, neural-network `.pt` files) are gitignored and stored in S3 instead. Back up / restore them with:

```bash
AWS_PROFILE=scuttle-bot python -m scuttle_bot.ml.model_store backup    # upload all model artifacts to S3
AWS_PROFILE=scuttle-bot python -m scuttle_bot.ml.model_store restore   # download them back locally
```

### Live-game role inference

Riot's Spectator-v5 API (used for live-game lookups) doesn't expose each pick's lane/role. The jungler is identified reliably via the Smite summoner spell; the other 4 roles per team are inferred from historical pick-role frequency (`src/scuttle_bot/utilities/champion_roles.json`, built from the same match dataset by `src/scuttle_bot/data/build_champion_roles.py`) using the Hungarian algorithm to jointly assign all 4 remaining roles without collisions. This is a best-effort heuristic, not ground truth.

## Project Structure

```
src/scuttle_bot/
├── service/
│   ├── bot.py                 # Main Discord bot implementation
│   ├── service.py             # ScuttleBotService facade, composing the mixins below
│   ├── riot_client.py         # Riot API access (summoner, ranked, mastery, live game, timelines)
│   ├── summoner_profile.py    # Composes + formats a full player profile
│   ├── registration.py        # Discord-user <-> LoL-account registration for daily reports
│   ├── personalities.py       # Canonical list of predefined bot personalities
│   ├── personality_service.py # Personality management (list/select/custom/remove)
│   └── reporter.py            # Daily performance report generation
├── llm/
│   ├── llm.py                 # LLM service: tool-calling loop, win prediction, history, logging
│   └── system_prompts.py      # Bot identity + behavioral/tool-use prompt text
├── analyzer/
│   └── match_analyzer.py      # Timeline-based coaching: checkpoint comparisons, notable moments
├── data/
│   ├── collector.py           # Riot API client for bulk data collection
│   ├── processor.py           # Raw Riot payloads -> training rows
│   ├── dataset.py             # Collects and stores training data in ml_dataset.db
│   ├── run_collection.py      # Collection entrypoint (+ S3 db backup)
│   └── build_champion_roles.py# Derives champion_roles.json from collected matches
├── ml/
│   ├── feature_encoder.py     # Shared preprocessing (encoding, scaling, imputation)
│   ├── predictor.py           # Serves win-probability predictions using the production model
│   ├── greedy_search.py       # Greedy coordinate-descent hyperparameter search
│   ├── model_store.py         # Back up / restore model weights to/from S3
│   ├── logistic/              # Logistic regression: model, training, artifacts (served model)
│   ├── rf/                    # Random forest: model, training, artifacts
│   └── nn/                    # Neural network: model, training, artifacts
├── utilities/
│   ├── schemas.py             # Region/queue enums and routing helpers
│   ├── role_inference.py      # Infers per-pick roles for a live game
│   ├── utilities.py           # Champion ID/name mapping helpers
│   ├── bot_utilities.py       # Discord UI (personality picker) + long-message chunking
│   └── champion_roles.json    # Historical pick-role frequencies for role inference
├── infra/
│   ├── db_client.py           # SQLite client (interactions, personalities, registrations, caches)
│   ├── aws_client.py          # AWS: Riot key from Secrets Manager, S3 db backup/restore
│   ├── schema.sql             # Bot-state + cache schema (incl. match + timeline caches)
│   ├── ml_schema.sql          # Training-dataset schema
│   └── match_participants_schema.sql
├── cache/                     # (gitignored) scuttle_bot.db bot state, ml_dataset.db training data
├── logs/                      # (gitignored) LLM interaction logs
└── __init__.py
```

## Key Components

- **ScuttleBotService**: Facade over Riot API access, summoner profiles, registration, and personality management, composed from focused mixins
- **LLMService**: Manages Gemini tool-calling, chaining multiple tool calls per request, win prediction, conversation history, and response logging
- **MatchAnalyzerMixin**: Timeline-based analysis -- checkpoint stat comparisons vs. lane opponent, and notable-moment detection with replay links
- **WinPredictor**: Loads the production model + encoder and turns a draft + player stats into a win probability
- **FeatureEncoder**: Shared preprocessing for all ML model types -- categorical encoding, scaling, and missing-value imputation
- **DatabaseClient**: SQLite database management for caching, personalities, conversation history, and registrations

## API Dependencies

- Discord.py
- Riot Games API
- Google Generative AI (Gemini)
- LangChain
- scikit-learn / PyTorch (model training and inference)
- pandas / scipy
- boto3 (AWS Secrets Manager + S3)
- python-dotenv

## License

MIT License
