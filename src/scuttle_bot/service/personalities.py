"""Canonical list of predefined bot personalities.

Single source of truth shared by the Discord personality dropdown
(bot_utilities.PersonalitySelect) and the LLM's personality tools
(ScuttleBotService.list_available_personalities / select_personality), so the
two surfaces never drift apart on what "available" means.
"""

from typing import NamedTuple, Optional


class Personality(NamedTuple):
    name: str
    description: str


PREDEFINED_PERSONALITIES: list[Personality] = [
    Personality("MrBeast", "Generous and adventurous personality like MrBeast from Youtube."),
    Personality("Kamado Tanjiro", "Kind and determined personality like Tanjiro from Demon Slayer."),
    Personality("Gordon Ramsay", "Blunt and fiery personality like Gordon Ramsay from cooking shows."),
    Personality("Yoda", "Wise and cryptic personality like Yoda from Star Wars."),
    Personality("Sherlock Holmes", "Analytical and observant personality like Sherlock Holmes."),
    Personality("Tony Stark", "Witty and confident personality like Tony Stark from Marvel."),
    Personality("Dwayne 'The Rock' Johnson", "Charismatic and motivational personality like The Rock."),
]


def find_personality(name: str) -> Optional[Personality]:
    """Case-insensitive lookup of a predefined personality by name."""
    normalized = name.strip().lower()
    for personality in PREDEFINED_PERSONALITIES:
        if personality.name.lower() == normalized:
            return personality
    return None
