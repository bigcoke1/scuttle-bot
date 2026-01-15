import discord
import os

from src.scuttle_bot.infra.db_client import DatabaseClient

class PersonalitySelect(discord.ui.Select):
    def __init__(self, user:str, db_client:DatabaseClient):
        self.db_client = db_client
        self.user = user

        options = [
            discord.SelectOption(label="Friendly", description="A warm and approachable personality."),
            discord.SelectOption(label="Professional", description="A formal and business-like personality."),
            discord.SelectOption(label="Humorous", description="A witty and funny personality."),
            discord.SelectOption(label="Encouraging", description="A supportive and motivational personality."),
            discord.SelectOption(label="Sarcastic", description="A sharp and ironic personality."),
            discord.SelectOption(label="Enthusiastic", description="An energetic and excited personality."),
            discord.SelectOption(label="MrBeast", description="Generous and adventurous personality like MrBeast from Youtube."),
            discord.SelectOption(label="Tanjiro", description="Kind and determined personality like Tanjiro from Demon Slayer."),
        ]
        super().__init__(placeholder="Choose a personality...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        selected_personality = self.values[0]

        self.db_client.store_personality_setting(user=self.user, personality=selected_personality)
        await interaction.response.edit_message(
            content=f"Personality: **{selected_personality}** has been saved!", 
            view=None
        )

class PersonalityView(discord.ui.View):
    def __init__(self, user:str, db_client:DatabaseClient):
        super().__init__()
        self.user = user
        self.add_item(PersonalitySelect(user=user, db_client=db_client))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        # If the person clicking isn't the author, send them a private error
        if interaction.user.name != self.user:
            await interaction.response.send_message("This menu isn't for you!", ephemeral=True)
            return False
        return True