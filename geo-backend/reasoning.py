def generate_reasoning(contributing_clues, narrowed_countries):
    if not narrowed_countries:
        return None
    top_country = narrowed_countries[0]
    clue_phrases = []  # list of (confidence, phrase) pairs

    for label, category, conf, pool in contributing_clues:
        if top_country not in pool:
            continue
        clue_phrases.append((conf, _humanize_clue(label, category)))

    if not clue_phrases:
        return f"{top_country} was the top guess, based mainly on the trained classifier's prediction."

    clue_phrases.sort(key=lambda x: x[0], reverse=True)
    top_clues = [phrase for _, phrase in clue_phrases[:3]]

    if len(top_clues) == 1:
        clue_text = top_clues[0]
        verb = "points"
    elif len(top_clues) == 2:
        clue_text = f"{top_clues[0]} and {top_clues[1]}"
        verb = "point"
    else:
        clue_text = f"{', '.join(top_clues[:-1])}, and {top_clues[-1]}"
        verb = "point"

    return f"{clue_text} {verb} most strongly toward {top_country}."


def _humanize_clue(label: str, category: str) -> str:
    """Turn a raw CLIP label into a natural-sounding phrase"""
    lead_insert = {
        "vegetation": "the",
        "architecture": "the",
        "road_marking": "",
        "road_marking_combo": "",
        "signage_style": "",
        "boundary_style": "",
        "terrain": "the",
    }
    lead = lead_insert.get(category, "the")
    return f"{lead} {label}".strip()