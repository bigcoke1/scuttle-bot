import discord

from scuttle_bot.service.schemas import Region
from src.scuttle_bot.service.service import ScuttleBotService
from src.scuttle_bot.service.llm import LLMService
from src.scuttle_bot.infra.db_client import DatabaseClient
from src.scuttle_bot.service.bot_utilities import PersonalityView

class ScuttleBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = DatabaseClient(os.getenv('DB_PATH', 'src/scuttle_bot/cache/scuttle_bot.db'))
        self.service = ScuttleBotService(db=self.db)
        self.llm_service = LLMService(db=self.db)

    async def on_ready(self):
        print(f'Logged in as {self.user}')

    async def on_message(self, message: discord.Message):
        try:
            if message.author == self.user:
                return
            
            content: str = message.content.lower()

            if content.startswith('$hello'):
                await message.channel.send('Hello!')

            if content.startswith('$help'):
                help_message = (
                    "I am Scuttle Bot! Here are my commands:\n"
                    "$hello - Greet the bot\n"
                    "$help - Show this help message\n\n"
                    "$personality - Set your personality for the bot\n"
                    "$stats <summoner_name>#<tag_line> <region> - Fetch ranked stats for a summoner\n"
                    "$chat <message> - Chat with the bot\n"
                )
                await message.channel.send(help_message)

            if content.startswith('$stats'):
                summoner = message.content.split(' ')[1].split("#", maxsplit=1)
                summoner_name, tag_line = summoner[0], summoner[1]
                region = message.content.split(' ')[2]
                stats = self.service.search_summoner(region=Region(region), summoner_name=summoner_name, tag_line=tag_line)
                if stats:
                    await message.channel.send(f"User stats for {summoner_name}#{tag_line}: {stats}")
                else:
                    await message.channel.send(f"Could not fetch stats for {summoner_name}#{tag_line}.")

            if content.startswith('$chat'):
                user_input = content[len('$chat '):]
                response = self.llm_service.generate_response(user_input, username=message.author.name)
                await message.channel.send(response)
            
            if content.startswith('$personality'):
                view = PersonalityView(user=message.author.name, db_client=self.db)
                await message.channel.send("Select a personality for the bot:", view=view)

            if content.startswith('$goodbye') and message.author.name == "sorrrymakerrr":
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