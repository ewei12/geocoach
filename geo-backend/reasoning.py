def generate_reasoning(contributing_clues, narrowed_countries):
    if not narrowed_countries:
        return None
    top_country = narrowed_countries[0]
    clue_phrases = []  # list of (confidence, phrase) pairs

    # Only get clues whose country pool actually has the winner
    for label, category, conf, pool in contributing_clues:
        if top_country not in pool:
            continue
        readable_category = category.replace("_", " ")
        clue_phrases.append((conf, f"{label} ({readable_category})"))

    if not clue_phrases:
        return f"{top_country} was the top guess, based mainly on the trained classifier's prediction."

    clue_phrases.sort(key=lambda x: x[0], reverse=True)
    top_clues = [phrase for _, phrase in clue_phrases[:3]]

    if len(top_clues) == 1:
        clue_text = top_clues[0]
    elif len(top_clues) == 2:
        clue_text = f"{top_clues[0]} and {top_clues[1]}"
    else:
        clue_text = f"{', '.join(top_clues[:-1])}, and {top_clues[-1]}"

    return f"{clue_text} point most strongly toward {top_country}."