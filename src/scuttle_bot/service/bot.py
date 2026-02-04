import logging
import sys
import discord
import os
import schedule

from scuttle_bot.service.schemas import Region
from src.scuttle_bot.service.service import ScuttleBotService
from src.scuttle_bot.service.llm import LLMService
from src.scuttle_bot.infra.db_client import DatabaseClient
from src.scuttle_bot.service.bot_utilities import PersonalityView
from src.scuttle_bot.service.reporter import Reporter

class ScuttleBot(discord.Client):

    REPORTING_TIME = "10:30"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = DatabaseClient(os.getenv('DB_PATH', 'src/scuttle_bot/cache/scuttle_bot.db'))
        self.service = ScuttleBotService(db=self.db)
        self.llm_service = LLMService(db=self.db)
        self.reporter = Reporter(db_client=self.db, llm_service=self.llm_service)
        schedule.every().day.at(self.REPORTING_TIME).do(self.report_daily)

        self.testing = kwargs.get('testing', False)

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
                    "$register <summoner_name>#<tag_line> <region> - Register for daily match performance reports\n"
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
            
            if content.startswith('$register'):
                summoner = message.content.split(' ')[1].split("#", maxsplit=1)
                summoner_name, tag_line = summoner[0], summoner[1]
                region = message.content.split(' ')[2]
                registration_success = self.service.register_user(
                    discord_id=str(message.author.id),
                    summoner_name=summoner_name,
                    tag_line=tag_line,
                    region=Region(region)
                )
                if registration_success:
                    await message.channel.send(f"Successfully registered {summoner_name}#{tag_line} for daily reports.")
                else:
                    await message.channel.send(f"Registration failed for {summoner_name}#{tag_line}. You may already be registered.")

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

            if content.startswith('$start_tests') and message.author.name == "sorrrymakerrr":
                self.testing = True
                await message.channel.send('Testing mode activated.')

            if self.testing and message.author.name == "sorrrymakerrr":
                if content.startswith('$test_chat'):
                    await message.author.send(f"Hello! This is a test message from Scuttle Bot.")
                    user = await self.fetch_user(584234181014323205)
                    if user:
                        await user.send(f"Hello! This is a test message from Scuttle Bot.")

                if content.startswith('$test_report'):
                    reports = self.reporter.generate_report()
                    for report in reports:
                        user_id = report['user']
                        report_content = report['report']
                        user = await self.fetch_user(int(user_id))
                        if user:
                            try:
                                await user.send(f"Test Report:\n{report_content}")
                                logging.info(f"Sent test report to user {user_id}")
                            except Exception as e:
                                logging.error(f"Failed to send test report to user {user_id}: {e}")
                if content.startswith('$stop_tests'):
                    self.testing = False
                    await message.channel.send('Testing mode deactivated.')

                if content.startswith('$reload'):
                    await message.channel.send('Reloading bot...')
                    await self.reload()

        except Exception as e:
            await message.channel.send(f"An error occurred...Please try again later.")
            logging.error(f"Error processing message: {e}")

    async def report_daily(self):
        logging.info("Starting daily report generation...")
        reports = self.reporter.generate_report()
        for report in reports:
            user_id = report['user']
            report_content = report['report']
            user = await self.fetch_user(int(user_id))
            if user:
                try:
                    await user.send(f"Daily Report:\n{report_content}")
                    logging.info(f"Sent daily report to user {user_id}")
                except Exception as e:
                    logging.error(f"Failed to send report to user {user_id}: {e}")

    async def reload(self):
        await self.close()
        os.execv(
            sys.executable,
            [sys.executable, "-m", "src.scuttle_bot.service.bot", *sys.argv[1:]]
        )

def main():
    from dotenv import load_dotenv

    load_dotenv()
    DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
    if DISCORD_TOKEN is None:
        raise ValueError("No DISCORD_TOKEN found in environment variables")
    
    intents = discord.Intents.default()
    intents.message_content = True
    client = ScuttleBot(intents=intents, testing=False)
    client.run(DISCORD_TOKEN)

if __name__ == '__main__':
    main()