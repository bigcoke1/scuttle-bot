import discord

from src.scuttle_bot.service.service import ScuttleBotService
from src.scuttle_bot.service.llm import LLMService
from src.scuttle_bot.infra.db_client import DatabaseClient

class ScuttleBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.db = DatabaseClient("../cache/scuttle_bot.db")
        self.service = ScuttleBotService(db=self.db)
        self.llm_service = LLMService(db=self.db)

    async def on_ready(self):
        print(f'Logged in as {self.user}')

    async def on_message(self, message):
        try:
            if message.author == self.user:
                return

            if message.content.startswith('$hello'):
                await message.channel.send('Hello!')

            if message.content.startswith('$help'):
                help_message = (
                    "Available commands:\n"
                    "$hello - Greet the bot\n"
                    "$help - Show this help message\n"
                    "$stats <summoner_name>#<tag_line> <region> - Fetch ranked stats for a summoner\n"
                    "$goodbye - Say goodbye to the bot"
                )
                await message.channel.send(help_message)

            if message.content.startswith('$stats'):
                summoner = message.content.split(' ')[1].split("#", maxsplit=1)
                game_name, tag_line = summoner[0], summoner[1]
                region = message.content.split(' ')[2]
                stats = self.service.search_summoner(region=region, game_name=game_name, tag_line=tag_line)
                if stats:
                    await message.channel.send(f"User stats for {game_name}#{tag_line}: {stats}")
                else:
                    await message.channel.send(f"Could not fetch stats for {game_name}#{tag_line}.")

            if message.content.startswith('$chat'):
                user_input = message.content[len('$chat '):]
                response = self.llm_service.generate_response(user_input)
                await message.channel.send(response)

            if message.content.startswith('$goodbye'):
                await message.channel.send('Goodbye!')
                self.db.close()
                await self.close()
        except Exception as e:
            await message.channel.send(f"An error occurred: {e}")

if __name__ == '__main__':
    import os
    from dotenv import load_dotenv

    load_dotenv()
    DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
    if DISCORD_TOKEN is None:
        raise ValueError("No DISCORD_TOKEN found in environment variables")
    
    intents = discord.Intents.default()
    intents.message_content = True
    client = ScuttleBot(intents=intents)
    client.run(DISCORD_TOKEN)