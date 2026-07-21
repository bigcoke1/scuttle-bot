A Discord bot that fetches League of Legends summoner data, analyzes it with Google's Gemini LLM, and predicts match win probability using models trained on real high-elo match data.

## Features

- **Summoner Lookup**: Fetch detailed information about League of Legends summoners including ranked stats, champion masteries, and recent match history
- **Live Game Lookup**: Detect a player's in-progress ranked game (via Riot's Spectator-v5 API) and infer each pick's role, so a game can be looked up without typing out the draft by hand
- **Win Probability Prediction**: Predict a draft's outcome from champion picks and each player's live rank/mastery, served by a RandomForest model trained on ~4,000 real high-elo matches
- **AI-Powered Analysis**: Uses Google Gemini with tool-calling to gather data (summoner stats, live games, win predictions) across multiple steps and summarize it conversationally
- **Conversation Memory**: Remembers a limited number of each Discord user's recent messages, so follow-up questions ("what's their win probability?") don't require repeating context
- **Customizable Personality**: Users can select different bot personalities to customize interactions
- **Match Caching**: Stores match data locally to reduce API calls and improve response times
- **Interaction Logging**: Records user interactions, the tool calls the LLM made to answer them, and the final responses for analysis and debugging

## Prerequisites

- Python 3.8+
- Discord bot token
- Riot API key
- Gemini API key

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
- `$chat <message>` - Chat with the bot. Can look up summoners, detect a player's live game, and predict its win probability -- e.g. `$chat is Faker#KR1 in a game right now, and if so who's favored to win?`. Remembers your recent messages, so natural follow-ups work without repeating context.
- `$personality` - Select a personality for the bot
- `$goodbye` - Shut down the bot (admin only)

## Machine Learning

Win probability is predicted by models trained on real matches collected from high-elo (Master+) NA solo/duo queue games, stored in `src/scuttle_bot/cache/ml_dataset.db`.

### Feature variants

Every model is trained in 4 variants of increasing feature complexity, sharing the same `FeatureEncoder`:

| Variant | Features |
|---|---|
| A | Draft (picks) only |
| B | Draft + average lobby tier |
| C | Draft + individual player stats (rank, win rate, champion mastery) |
| D | Draft + player stats + bans |

### Model types

Three model families are trained on all 4 variants: logistic regression, random forest, and a neural network (whose architecture depth/width scales with each variant's feature complexity). Each variant is trained 5 times with different random splits and the results averaged, so results in each `models/<variant>/cv_summary.json` reflect a mean +/- stdev rather than a single lucky split.

The bot currently serves win-probability predictions with the best-performing combination found this way: **RandomForest, variant C** (draft + player stats, no bans) -- see `src/scuttle_bot/ml/predictor.py`.

### Training

```bash
rye run python -m scuttle_bot.ml.logistic.train
rye run python -m scuttle_bot.ml.rf.train
rye run python -m scuttle_bot.ml.nn.train
```

Each writes trained models, configs, confusion-matrix plots, and a `cv_summary.json` per variant under that model type's `models/` and `plots/` directories.

### Live-game role inference

Riot's Spectator-v5 API (used for live-game lookups) doesn't expose each pick's lane/role. The jungler is identified reliably via the Smite summoner spell; the other 4 roles per team are inferred from historical pick-role frequency (`src/scuttle_bot/service/champion_roles.json`, built from the same match dataset by `src/scuttle_bot/data/build_champion_roles.py`) using the Hungarian algorithm to jointly assign all 4 remaining roles without collisions. This is a best-effort heuristic, not ground truth.

## Project Structure

```
src/scuttle_bot/
├── service/
│   ├── bot.py               # Main Discord bot implementation
│   ├── bot_utilities.py     # Discord UI components (personality picker)
│   ├── llm.py               # LLM service: tool-calling loop, conversation history, logging
│   ├── service.py           # League of Legends API integration (summoner, live game lookups)
│   ├── role_inference.py    # Infers per-pick roles for a live game
│   ├── reporter.py          # Daily performance report generation
│   ├── utilities.py         # Champion ID/name mapping helpers
│   └── schemas.py           # Data models and enums
├── data/
│   ├── collector.py         # Riot API client (match history, ranked stats, mastery, live games)
│   ├── processor.py         # Raw Riot payloads -> training rows
│   ├── dataset.py           # Collects and stores training data in ml_dataset.db
│   └── build_champion_roles.py  # Derives champion_roles.json from collected matches
├── ml/
│   ├── feature_encoder.py   # Shared preprocessing (encoding, scaling, missing-value imputation)
│   ├── predictor.py         # Serves win-probability predictions using the production model
│   ├── logistic/            # Logistic regression: model, training, artifacts
│   ├── rf/                  # Random forest: model, training, artifacts
│   └── nn/                  # Neural network: model, training, artifacts
├── infra/
│   └── db_client.py         # SQLite database client (interactions, personalities, registrations)
├── cache/
│   ├── scuttle_bot.db       # Bot state (interactions, personalities, registered users)
│   └── ml_dataset.db        # Collected match data used to train the ML models
├── logs/                    # LLM interaction logs (user input, tool calls, final response)
└── __init__.py
```

## Key Components

- **ScuttleBotService**: Handles League of Legends API interactions, including summoner lookups and live-game detection
- **LLMService**: Manages Gemini tool-calling, chaining multiple tool calls per request, conversation history, and response logging
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
- python-dotenv

## License

MIT License
