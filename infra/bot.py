import discord
from service.service import ScuttleBotService

class MyClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.service = ScuttleBotService()

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
                    "$opgg <summoner_name> <region> - Fetch OPGG stats for a summoner\n"
                    "$goodbye - Say goodbye to the bot"
                )
                await message.channel.send(help_message)

            if message.content.startswith('$opgg'):
                summoner_name = message.content.split(' ')[1]
                region = message.content.split(' ')[2]
                stats = self.service.get_user_stats(summoner_name, region)
                if stats:
                    await message.channel.send(f"User stats for {summoner_name}: {stats}")
                else:
                    await message.channel.send(f"Could not fetch stats for {summoner_name}.")

            if message.content.startswith('$goodbye'):
                await message.channel.send('Goodbye!')
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
    client = MyClient(intents=intents)
    client.run(DISCORD_TOKEN)