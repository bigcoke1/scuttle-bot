A Discord bot that fetches League of Legends summoner data and analyzes it with Google's Gemini LLM to provide intelligent insights and performance analysis.

## Features

- **Summoner Lookup**: Fetch detailed information about League of Legends summoners including ranked stats, champion masteries, and recent match history
- **AI-Powered Analysis**: Uses Google Gemini to provide intelligent summaries and insights about player performance
- **Customizable Personality**: Users can select different bot personalities to customize interactions
- **Match Caching**: Stores match data locally to reduce API calls and improve response times
- **Interaction Logging**: Records all user interactions and bot responses for analysis and debugging

## Prerequisites

- Python 3.8+
- Discord bot token
- Riot API key
- Google API key (for Gemini LLM)

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
RIOT_KEY=your_riot_api_key
GOOGLE_API_KEY=your_google_api_key
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
- `$chat <message>` - Chat with the bot and ask questions about League of Legends
- `$personality` - Select a personality for the bot
- `$goodbye` - Shut down the bot (admin only)

## Project Structure

```
src/scuttle_bot/
├── service/
│   ├── bot.py              # Main Discord bot implementation
│   ├── bot_utilities.py    # Discord UI components
│   ├── llm.py              # LLM service for AI responses
│   ├── service.py          # League of Legends API integration
│   └── schemas.py          # Data models and enums
├── infra/
│   └── db_client.py        # SQLite database client
├── cache/                  # Match data cache
├── logs/                   # LLM interaction logs
└── __init__.py
```

## Key Components

- **ScuttleBotService**: Handles League of Legends API interactions and data fetching
- **LLMService**: Manages Gemini API calls and response formatting
- **DatabaseClient**: SQLite database management for caching and persistence

## API Dependencies

- Discord.py
- Riot Games API
- Google Generative AI (Gemini)
- LangChain
- python-dotenv

## License

MIT License