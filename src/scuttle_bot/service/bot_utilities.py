import discord
import os

from src.scuttle_bot.infra.db_client import DatabaseClient

class PersonalitySelect(discord.ui.Select):
    def __init__(self, user:str, db_client:DatabaseClient):
        self.db_client = db_client
        self.user = user

        options = [
            discord.SelectOption(label="MrBeast", description="Generous and adventurous personality like MrBeast from Youtube."),
            discord.SelectOption(label="Kamado Tanjiro", description="Kind and determined personality like Tanjiro from Demon Slayer."),
            discord.SelectOption(label="Gordon Ramsay", description="Blunt and fiery personality like Gordon Ramsay from cooking shows."),
            discord.SelectOption(label="Yoda", description="Wise and cryptic personality like Yoda from Star Wars."),
            discord.SelectOption(label="Sherlock Holmes", description="Analytical and observant personality like Sherlock Holmes."),
            discord.SelectOption(label="Tony Stark", description="Witty and confident personality like Tony Stark from Marvel."),
            discord.SelectOption(label="Dwayne 'The Rock' Johnson", description="Charismatic and motivational personality like The Rock."),
        ]
        super().__init__(placeholder="Choose a personality...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        selected_personality = self.values[0]

        self.db_client.store_personality_setting(user_id=self.user, personality=selected_personality)
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