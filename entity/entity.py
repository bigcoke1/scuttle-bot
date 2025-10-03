from enum import Enum

class Riot_Regions(Enum):
    AMERICAS = "americas"
    ASIA = "asia"
    EUROPE = "europe"
    SEA = "sea"

    def __str__(self):
        return self.value

class Lol_Regions(Enum):
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

    def __str__(self):
        return self.value