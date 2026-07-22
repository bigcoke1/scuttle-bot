import requests
import sys
import logging
import re

# requests.get() has no default timeout -- a stalled connection blocks
# forever with no exception ever raised.
REQUEST_TIMEOUT = 30

def get_champion_mapping(version = None):
    try:
        if version is None:
            version = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=REQUEST_TIMEOUT).json()[0]
        url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
        data = requests.get(url, timeout=REQUEST_TIMEOUT).json()["data"]

        mapping = {int(info["key"]): info["name"] for info in data.values()}
        return mapping
    except Exception as e:
        return {}
    
def get_champ_to_idx():
    champion_mapping = get_champion_mapping()
    champ_to_idx = {
        re.sub(r"[^A-Za-z0-9]", "", champ).lower(): idx
        for idx, champ in enumerate(sorted(champion_mapping.values()))
    }
    champ_to_idx["Unknown"] = -1  # Add mapping for unknown champions
    return champ_to_idx

def get_id_to_idx(version = None):
    champion_mapping = get_champion_mapping(version)
    id_to_idx = {
        champ_id: idx
        for idx, champ_id in enumerate(sorted(champion_mapping.keys(), key=lambda cid: champion_mapping[cid]))  # Sort by champion name to ensure consistent ordering
    }
    return id_to_idx
    
def error_traceback():
    # 1. Capture the traceback
    exc_type, exc_value, exc_traceback = sys.exc_info()
    
    # 2. Get the "frame" where the error actually happened
    # We go to the end of the traceback to find the deepest function (your tool)
    tb = exc_traceback
    if tb is not None:
        while tb.tb_next:
            tb = tb.tb_next
        frame = tb.tb_frame
    else:
        return
    
    # 3. Log the error and the variables in that function
    logging.error(f"CRASH: {exc_value}")
    logging.error(f"Error occurred in: {frame.f_code.co_name}")
    
    # 4. Filter and log only lists to find the culprit
    logging.error("--- Local Variable States ---")
    for var_name, var_value in frame.f_locals.items():
        if isinstance(var_value, list):
            logging.error(f"LIST FOUND: {var_name} (Length: {len(var_value)}) -> {var_value}")
        elif isinstance(var_value, (str, int, dict)):
            logging.error(f"VAR: {var_name} = {var_value}")