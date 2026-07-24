"""Per-user bot personality management, backed by the predefined list in
personalities.py.

PersonalityMixin is mixed into ScuttleBotService, which owns self.db that
these methods read/write via self.
"""

from scuttle_bot.service.personalities import PREDEFINED_PERSONALITIES, find_personality


class PersonalityMixin:
    def list_available_personalities(self) -> list[dict]:
        """Lists the predefined personalities the bot can adopt when responding
        to a user, each with a short description of its tone. Use this before
        calling select_personality so you can offer the user real options, or
        to answer a user asking what personalities are available.
        """
        return [{"name": p.name, "description": p.description} for p in PREDEFINED_PERSONALITIES]

    def select_personality(self, discord_id: str, personality_name: str) -> str:
        """Sets a Discord user's response personality to one of the predefined
        personalities from list_available_personalities. Matching is
        case-insensitive. Fails if personality_name doesn't match a predefined
        personality -- call list_available_personalities first if unsure, or
        use set_custom_personality instead for a personality that isn't on
        the predefined list.

        Args:
            discord_id: The Discord user's numeric ID.
            personality_name: Name of a predefined personality, e.g. 'Yoda'.
        """
        try:
            match = find_personality(personality_name)
            if match is None:
                available = ", ".join(p.name for p in PREDEFINED_PERSONALITIES)
                return f"Error: {personality_name!r} is not a predefined personality. Available: {available}."
            self.db.store_personality_setting(user_id=discord_id, personality=match.name)
            return f"Personality set to {match.name}."
        except Exception as e:
            self.error_traceback()
            return f"Error: {str(e)}"

    def set_custom_personality(self, discord_id: str, personality_description: str) -> bool:
        """Sets a custom, free-form response personality for a Discord user,
        instead of one of the predefined options. Use this when the user
        describes a personality in their own words rather than naming one of
        list_available_personalities' options.

        Args:
            discord_id: The Discord user's numeric ID.
            personality_description: Free-form description of the personality to adopt, e.g. "a pirate captain who's always in a hurry".
        """
        try:
            self.db.store_personality_setting(user_id=discord_id, personality=personality_description)
            return True
        except Exception as e:
            self.error_traceback()
            return False

    def remove_personality(self, discord_id: str) -> bool:
        """Clears a Discord user's personality setting, reverting the bot to
        its default voice for that user. Returns True if a setting was found
        and removed, False if none was set.

        Args:
            discord_id: The Discord user's numeric ID.
        """
        try:
            return self.db.delete_personality_setting(discord_id)
        except Exception as e:
            self.error_traceback()
            return False
