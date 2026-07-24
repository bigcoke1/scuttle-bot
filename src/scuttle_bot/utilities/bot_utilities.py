import discord
import os

from src.scuttle_bot.infra.db_client import DatabaseClient
from src.scuttle_bot.service.personalities import PREDEFINED_PERSONALITIES

# Discord's message content limit is 2000 characters for everyone; some
# boosted servers allow more, but this codebase has no way to know a given
# server's boost level, so it stays under the universal floor rather than
# risking a 400 on servers without the higher limit.
DISCORD_MESSAGE_LIMIT = 1900


async def send_long_message(destination, text: str, limit: int = DISCORD_MESSAGE_LIMIT):
    """Sends `text` to `destination` (anything with an async .send(content),
    e.g. a channel or a user), splitting across multiple messages if it
    exceeds Discord's per-message length limit. LLM responses can run long
    on their own, and tool results embedding things like signed replay URLs
    (a few hundred characters each) make it easy to blow past the limit
    without the response looking obviously huge -- this is the backstop so
    that overflow degrades to multiple messages instead of a crash.
    Splits on line boundaries where possible; a single line longer than the
    limit on its own still gets hard-split as a last resort.
    """
    if len(text) <= limit:
        await destination.send(text)
        return

    chunk = ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if chunk:
                await destination.send(chunk)
                chunk = ""
            await destination.send(line[:limit])
            line = line[limit:]

        if len(chunk) + len(line) > limit:
            await destination.send(chunk)
            chunk = ""
        chunk += line

    if chunk:
        await destination.send(chunk)

class PersonalitySelect(discord.ui.Select):
    def __init__(self, discord_id:str, db_client:DatabaseClient):
        self.db_client = db_client
        self.discord_id = discord_id

        options = [
            discord.SelectOption(label=p.name, description=p.description)
            for p in PREDEFINED_PERSONALITIES
        ]
        super().__init__(placeholder="Choose a personality...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        selected_personality = self.values[0]

        self.db_client.store_personality_setting(user_id=self.discord_id, personality=selected_personality)
        await interaction.response.edit_message(
            content=f"Personality: **{selected_personality}** has been saved!", 
            view=None
        )

class PersonalityView(discord.ui.View):
    def __init__(self, discord_id:str, db_client:DatabaseClient):
        super().__init__()
        self.discord_id = discord_id
        self.add_item(PersonalitySelect(discord_id=discord_id, db_client=db_client))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        # If the person clicking isn't the author, send them a private error
        if str(interaction.user.id) != self.discord_id:
            await interaction.response.send_message("This menu isn't for you!", ephemeral=True)
            return False
        return True