"""Discord user <-> League of Legends account registration, for the daily
match performance reports (see reporter.Reporter).

RegistrationMixin is mixed into ScuttleBotService, which owns self.db and
provides get_puuid (from RiotClientMixin) that these methods call via self.
"""

from typing import Optional

from scuttle_bot.utilities.schemas import Region


class RegistrationMixin:
    def register_user(self, discord_id: str, summoner_name: str, tag_line: str, region: Region) -> bool:
        """Registers a Discord user's League of Legends account so they receive
        automated daily match performance reports. Fails if the Riot ID can't
        be resolved or the Discord user is already registered.

        Args:
            discord_id: The Discord user's numeric ID.
            summoner_name: Riot ID game name (the part before the #).
            tag_line: Riot ID tagline (the part after the #).
            region: Riot region the account is registered in, e.g. 'NA', 'EUW1', 'KR'.
        """
        try:
            if isinstance(region, str):
                region = Region(region)
            puuid = self.get_puuid(summoner_name, tag_line, region=region)
            if puuid is None:
                return False

            self.db.register_user(
                discord_id=discord_id,
                summoner_name=summoner_name,
                tag_line=tag_line,
                region=region.value,
                puuid=puuid
            )
            return True
        except Exception as e:
            self.error_traceback()
            return False

    def get_registered_user(self, discord_id: str) -> Optional[dict]:
        """Looks up the League of Legends account a Discord user has registered
        for daily reports, if any. Returns None if they aren't registered.

        Args:
            discord_id: The Discord user's numeric ID.
        """
        try:
            user = self.db.get_registered_user(discord_id)
            return user
        except Exception as e:
            self.error_traceback()
            return None

    def unregister_user(self, discord_id: str) -> bool:
        """Removes a Discord user's League of Legends account registration,
        stopping their daily match performance reports. Returns True if a
        registration was found and removed, False if the user wasn't
        registered in the first place.

        Args:
            discord_id: The Discord user's numeric ID.
        """
        try:
            return self.db.unregister_user(discord_id)
        except Exception as e:
            self.error_traceback()
            return False
