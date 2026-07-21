from enum import Enum

class Region(Enum):
    NA = "na1"
    BR1 = "br1"
    LA1 = "la1"
    LA2 = "la2"
    KR = "kr"
    JP1 = "jp1"
    EUW1 = "euw1"
    EUN1 = "eun1"
    TR1 = "tr1"
    TW2 = "tw2"
    RU = "ru"
    OC1 = "oc1"
    PH2 = "ph2"
    SG2 = "sg2"
    TH2 = "th2"
    VN2 = "vn2"

class Queue(Enum):
    RANKED_SOLO_5x5 = "RANKED_SOLO_5x5"
    RANKED_FLEX_SR = "RANKED_FLEX_SR"
    RANKED_FLEX_TT = "RANKED_FLEX_TT"

# account-v1 only has americas/asia/europe clusters (no "sea") -- OC1/PH2/SG2/
# TH2/TW2/VN2 fall back to asia for it, per Riot's own routing guidance.
_ACCOUNT_ROUTING = {
    Region.NA: "americas", Region.BR1: "americas", Region.LA1: "americas", Region.LA2: "americas",
    Region.KR: "asia", Region.JP1: "asia",
    Region.EUW1: "europe", Region.EUN1: "europe", Region.TR1: "europe", Region.RU: "europe",
    Region.OC1: "asia", Region.PH2: "asia", Region.SG2: "asia", Region.TH2: "asia", Region.TW2: "asia", Region.VN2: "asia",
}

# match-v5 (and other v5 match/tournament APIs) additionally has a dedicated
# "sea" cluster for OC1/PH2/SG2/TH2/TW2/VN2.
_MATCH_ROUTING = {
    **_ACCOUNT_ROUTING,
    Region.OC1: "sea", Region.PH2: "sea", Region.SG2: "sea", Region.TH2: "sea", Region.TW2: "sea", Region.VN2: "sea",
}


def get_account_routing_url(region: Region) -> str:
    """Continental base URL for account-v1 calls (e.g. puuid lookup)."""
    return f"https://{_ACCOUNT_ROUTING.get(region, 'americas')}.api.riotgames.com"


def get_match_routing_url(region: Region) -> str:
    """Continental base URL for match-v5 calls."""
    return f"https://{_MATCH_ROUTING.get(region, 'americas')}.api.riotgames.com"