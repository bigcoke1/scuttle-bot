import asyncio
import logging
import sys
import discord
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

from discord.ext import tasks

from scuttle_bot.utilities.schemas import Region
from src.scuttle_bot.service.service import ScuttleBotService
from scuttle_bot.llm.llm import LLMService
from src.scuttle_bot.infra.db_client import DatabaseClient
from scuttle_bot.utilities.bot_utilities import PersonalityView, send_long_message
from src.scuttle_bot.service.reporter import Reporter

# Daily report time, overridable via env for different deployments. Parsed into
# a timezone-aware datetime.time up front because discord.ext.tasks.loop needs
# the schedule at class-definition time; a naive time would be treated as UTC.
REPORTING_TIME = os.getenv("REPORT_TIME", "10:30")
REPORT_TIMEZONE = os.getenv("REPORT_TIMEZONE", "America/Los_Angeles")


def _parse_daily_time(hhmm: str, tz: str) -> dtime:
    hour, minute = (int(part) for part in hhmm.split(":"))
    return dtime(hour=hour, minute=minute, tzinfo=ZoneInfo(tz))


_REPORT_TIME = _parse_daily_time(REPORTING_TIME, REPORT_TIMEZONE)


class ScuttleBot(discord.Client):

    KNOWN_PREFIXES = (
        '$hello', '$help', '$stats', '$register', '$chat', '$personality', '$goodbye',
        '$start_tests', '$test_chat', '$test_report', '$stop_tests', '$reload',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = DatabaseClient(os.getenv('DB_PATH', 'src/scuttle_bot/cache/scuttle_bot.db'))
        self.service = ScuttleBotService(db=self.db)
        self.llm_service = LLMService(db=self.db)
        self.reporter = Reporter(db_client=self.db, llm_service=self.llm_service)

        self.testing = kwargs.get('testing', False)

    async def on_ready(self):
        print(f'Logged in as {self.user}')
        # Start the daily-report loop once the gateway is up. Guarded because
        # on_ready fires again on every reconnect.
        if not self.daily_report_task.is_running():
            self.daily_report_task.start()

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
                    "$chat <message> - Chat with the bot. Can look up summoners, detect a player's live game, and predict its win probability -- e.g. $chat is Faker#KR1 in a game right now, and if so who's favored to win?. Remembers your recent messages, so natural follow-ups work without repeating context.\n"
                    "(In a DM, you can skip the $chat prefix and just type your message directly.)\n"
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
                response = self.llm_service.generate_response(user_input, discord_id=str(message.author.id))
                await send_long_message(message.channel, response)

            if content.startswith('$personality'):
                view = PersonalityView(discord_id=str(message.author.id), db_client=self.db)
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
                    # Offloaded like the daily report -- see report_daily().
                    reports = await asyncio.to_thread(self.reporter.generate_report)
                    for report in reports:
                        user_id = report['user']
                        report_content = report['report']
                        user = await self.fetch_user(int(user_id))
                        if user:
                            try:
                                await send_long_message(user, f"Test Report:\n{report_content}")
                                logging.info(f"Sent test report to user {user_id}")
                            except Exception as e:
                                logging.error(f"Failed to send test report to user {user_id}: {e}")
                if content.startswith('$stop_tests'):
                    self.testing = False
                    await message.channel.send('Testing mode deactivated.')

                if content.startswith('$reload'):
                    await message.channel.send('Reloading bot...')
                    await self.reload()

            is_dm = isinstance(message.channel, discord.DMChannel)
            if is_dm and not content.startswith(self.KNOWN_PREFIXES):
                # In DMs, a plain message is chat -- no $chat prefix needed.
                response = self.llm_service.generate_response(message.content, discord_id=str(message.author.id))
                await send_long_message(message.channel, response)

        except Exception as e:
            await message.channel.send(f"An error occurred...Please try again later.")
            logging.error(f"Error processing message: {e}")

    @tasks.loop(time=_REPORT_TIME)
    async def daily_report_task(self):
        await self.report_daily()

    @daily_report_task.before_loop
    async def before_daily_report(self):
        # Don't run the first scheduled report until the client is fully ready
        # (fetch_user / DMs need an established gateway connection).
        await self.wait_until_ready()

    async def report_daily(self):
        logging.info("Starting daily report generation...")
        # generate_report() is synchronous and slow (an LLM call plus a 10s
        # sleep per registered user), so run it off the event loop -- otherwise
        # it would block the gateway and freeze all message handling for minutes.
        reports = await asyncio.to_thread(self.reporter.generate_report)
        for report in reports:
            user_id = report['user']
            report_content = report['report']
            user = await self.fetch_user(int(user_id))
            if user:
                try:
                    await send_long_message(user, f"Daily Report:\n{report_content}")
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
    from scuttle_bot.infra.aws_client import (
        get_secret,
        DISCORD_TOKEN_SECRET_NAME,
        GEMINI_API_KEY_SECRET_NAME,
    )

    load_dotenv()

    # Secrets Manager is the source of truth in production (the EC2 instance
    # role can read it); locally it's unreachable so the .env value wins. The
    # Gemini key is pushed back into the environment because LLMService reads
    # it via os.getenv when it's constructed inside ScuttleBot below.
    DISCORD_TOKEN = get_secret(DISCORD_TOKEN_SECRET_NAME) or os.getenv('DISCORD_TOKEN')
    gemini_api_key = get_secret(GEMINI_API_KEY_SECRET_NAME) or os.getenv('GEMINI_API_KEY')
    if gemini_api_key:
        os.environ['GEMINI_API_KEY'] = gemini_api_key

    if DISCORD_TOKEN is None:
        raise ValueError("No Discord token found in Secrets Manager or environment variables")

    intents = discord.Intents.default()
    intents.message_content = True
    client = ScuttleBot(intents=intents, testing=False)
    client.run(DISCORD_TOKEN)

if __name__ == '__main__':
    main()