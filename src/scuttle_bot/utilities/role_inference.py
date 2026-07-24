import json
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

ROLES = ["top", "jungle", "mid", "adc", "support"]
ROLE_FREQUENCIES_PATH = "src/scuttle_bot/utilities/champion_roles.json"

_role_frequencies_cache = None


def _load_role_frequencies() -> dict:
    global _role_frequencies_cache
    if _role_frequencies_cache is None:
        with open(ROLE_FREQUENCIES_PATH) as f:
            raw = json.load(f)
        _role_frequencies_cache = {int(champ_id): counts for champ_id, counts in raw.items()}
    return _role_frequencies_cache


def infer_roles(champion_ids: list[int], known: Optional[dict[int, str]] = None) -> dict[int, str]:
    """
    Assigns each champion_id one of the 5 roles for a single team, using
    historical pick-role frequency (see build_champion_roles.py) and the
    Hungarian algorithm to solve for the joint assignment that best matches
    every champion's role tendencies at once. This matters because picking
    each champion's single most-common role independently can collide --
    e.g. two flex picks both defaulting to "mid" and leaving "top" empty.

    `known` pins any roles already determined by other means (e.g. jungler
    identified via Smite); those champions/roles are excluded from the solve.
    """
    known = dict(known or {})
    frequencies = _load_role_frequencies()

    remaining_champs = [c for c in champion_ids if c not in known]
    remaining_roles = [r for r in ROLES if r not in known.values()]

    assignment = known
    if not remaining_champs or not remaining_roles:
        return assignment

    cost = np.zeros((len(remaining_champs), len(remaining_roles)))
    for i, champ_id in enumerate(remaining_champs):
        champ_freq = frequencies.get(champ_id, {})
        total = sum(champ_freq.values()) or 1
        for j, role in enumerate(remaining_roles):
            cost[i, j] = -(champ_freq.get(role, 0) / total)

    row_idx, col_idx = linear_sum_assignment(cost)
    for i, j in zip(row_idx, col_idx):
        assignment[remaining_champs[i]] = remaining_roles[j]

    return assignment
