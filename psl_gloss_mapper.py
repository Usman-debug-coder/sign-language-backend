import json
from typing import Dict, Tuple, Set


def load_psl_gloss_index(json_path: str) -> Dict[str, object]:
    """
    Load PSL dictionary items and build a lookup index.

    Expects a JSON array where each item has keys:
      - "text": gloss word (string)
      - "category_name": category label (string)

    Returns a dict with:
      - by_lower: Dict[str, Tuple[str, str]] mapping lowercased gloss -> (CANONICAL_UPPER, category)
      - gloss_set_lower: Set[str] of all lowercased gloss words
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_lower: Dict[str, Tuple[str, str]] = {}
    gloss_set_lower: Set[str] = set()

    for item in data:
        # Defensive reads in case of imperfect data
        text = (item.get("text") or "").strip()
        category = (item.get("category_name") or "").strip()
        if not text:
            continue

        lower = text.lower()
        canonical_upper = text.upper()
        by_lower[lower] = (canonical_upper, category)
        gloss_set_lower.add(lower)

    return {
        "by_lower": by_lower,
        "gloss_set_lower": gloss_set_lower,
    }


__all__ = ["load_psl_gloss_index"]


