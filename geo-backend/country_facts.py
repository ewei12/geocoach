# country_facts.py
import re
import pycountry

# Only countries actually present in the training set (osv5m_filtered).
TRAINED_COUNTRY_CODES = {
    "SE", "BO", "BR", "GB", "US", "RU", "HU", "FR", "CA", "TH", "ES", "UZ",
    "MA", "BD", "NP", "MX", "KW", "LV", "DE", "IT", "PK", "ZM", "AR", "GR",
    "RW", "NZ", "DO", "JP", "FI", "SK", "PL", "PH", "VN", "SG", "NI", "TW",
    "BY", "IR", "TZ", "IS", "AL", "AU", "CO", "MY", "HK", "PY", "NO", "HN",
    "QA", "RO", "NL", "BG", "DK", "ZA", "ID", "LT", "CH", "LK", "IN", "IE",
    "PT", "NG", "CL", "BN", "MN", "OM", "BH", "DZ", "RS", "EE", "HR", "TR",
    "BA", "UG", "MD", "MM", "CY", "CZ", "AT", "SI", "BE", "EG", "IL", "GH",
    "PS", "GT", "TN", "TL", "RE", "EC", "MU", "SN", "UY", "JO", "AE", "ET",
    "GE", "KR", "PE", "SA", "MZ", "CN", "MK", "PA", "LS", "TM", "LU", "CR",
    "SL", "AZ", "ML", "LA", "KG", "XK", "KZ", "CD", "SV", "KE", "MR"
}

# Common/short names to use instead of pycountry's formal ISO names.
_COMMON_NAMES = {
    "RU": "Russia",
    "KR": "South Korea",
    "BO": "Bolivia",
    "MD": "Moldova",
    "US": "United States",
    "GB": "United Kingdom",
    "VN": "Vietnam",
    "SY": "Syria",
    "TW": "Taiwan",
    "LA": "Laos",
    "IR": "Iran",
    "TZ": "Tanzania",
    "VE": "Venezuela",
    "CZ": "Czechia",
    "BN": "Brunei",
    "VN": "Vietnam",
}

# Kosovo (XK) isn't in pycountry's standard ISO list
_MANUAL_NAMES = {"XK": "Kosovo"}


def code_to_name(code: str) -> str:
    if code in _MANUAL_NAMES:
        return _MANUAL_NAMES[code]
    if code in _COMMON_NAMES:
        return _COMMON_NAMES[code]
    country = pycountry.countries.get(alpha_2=code)
    return country.name if country else code


# code -> full name, name -> code, only for the training set
CODE_TO_NAME = {c: code_to_name(c) for c in TRAINED_COUNTRY_CODES}
NAME_TO_CODE = {v: k for k, v in CODE_TO_NAME.items()}
TRAINED_NAMES = set(CODE_TO_NAME.values())


def _only_trained(countries: list) -> list:
    """Keep only countries that are actually in the training set."""
    return [c for c in countries if c in TRAINED_NAMES]


# Sign background color (restricted to trained countries) -----------------------------
SIGN_BACKGROUND_COLOR = {
    "blue": _only_trained([
        "France", "Italy", "Spain", "Portugal", "Netherlands", "Belgium",
        "Switzerland", "United Arab Emirates", "Saudi Arabia", "Egypt",
        "Turkey", "Greece", "Israel",
    ]),
    "yellow": _only_trained([
        "Germany", "Poland", "Czechia", "Slovakia", "Hungary", "Austria",
    ]),
    "white": _only_trained([
        "United States", "United Kingdom", "Australia", "New Zealand",
        "South Africa", "Japan", "Ireland", "India", "Brazil", "Argentina",
    ]),
    "green": _only_trained([
        "United States", "Sweden", "Finland", "Norway", "Denmark",
        "Korea, Republic of", "China",
    ]),
}

# Road center-line color ----------------------------------------------------------------
ROAD_LINE_COLOR = {
    "yellow_center": _only_trained([
        "United States", "Canada", "Philippines", "Korea, Republic of", "Mexico",
    ]),
    "white_center": _only_trained([
        "Germany", "France", "Austria", "United Kingdom", "Spain", "Italy",
        "Australia", "New Zealand", "Japan",
        "Czechia", "Slovakia", "Poland", "Hungary", "Croatia", "Slovenia",
        "Romania", "Bosnia and Herzegovina", "Serbia", "Bulgaria",
        "Netherlands", "Belgium", "Switzerland", "Portugal", "Ireland",
        "Denmark", "Sweden", "Finland", "Norway", "Greece", "Turkey",
        "Estonia", "Latvia", "Lithuania", "Albania", "North Macedonia",
    ]),
}

ROAD_LINE_COMBO = {
    "white_center_yellow_edge": _only_trained(["South Africa"]),
}

# Architecture style ---------------------------------------------------------------------
ARCHITECTURE_STYLE = {
    "German or Austrian half-timbered houses": _only_trained(
        ["Germany", "Austria", "Switzerland"]),
    "French rural stone farmhouses": _only_trained(
        ["France", "Belgium"]),
    "Alpine wooden chalets with sloped roofs": _only_trained(
        ["Switzerland", "Austria", "France", "Italy"]),
    "British brick terraced houses": _only_trained(
        ["United Kingdom", "Ireland"]),
    "North American suburban houses with front lawns and driveways": _only_trained(
        ["United States", "Canada"]),
    "colorful Latin American concrete buildings": _only_trained([
        "Mexico", "Brazil", "Colombia", "Peru", "Argentina", "Chile",
        "Ecuador", "Bolivia", "Guatemala", "Honduras",
        "Costa Rica", "Panama", "El Salvador", "Nicaragua", "Dominican Republic",
        "Paraguay", "Uruguay",
    ]),
    "Eastern European Soviet-era apartment blocks": _only_trained([
        "Russian Federation", "Belarus", "Poland", "Romania", "Bulgaria",
        "Serbia", "Bosnia and Herzegovina", "North Macedonia", "Georgia",
        "Azerbaijan", "Kazakhstan", "Uzbekistan", "Moldova",
        "Albania", "Latvia", "Lithuania", "Estonia",
    ]),
}

# Vegetation ------------------------------------------------------------------------------
VEGETATION_REGION = {
    "Mediterranean shrubland with olive and cypress trees": _only_trained([
        "Italy", "Spain", "Greece", "Portugal", "Turkey", "Morocco",
        "Israel", "Croatia", "Cyprus", "Tunisia", "Algeria",
    ]),
    "dense temperate deciduous forest": _only_trained([
        "Germany", "France", "United Kingdom", "Poland", "Czechia",
        "Slovakia", "Hungary", "Austria", "Belgium", "Netherlands",
        "United States", "Canada", "Japan", "South Korea",
    ]),
    "coniferous alpine forest": _only_trained([
        "Switzerland", "Austria", "Norway", "Sweden", "Finland",
        "Slovenia", "Canada",
    ]),
    "dry Australian eucalyptus bushland": _only_trained(["Australia"]),
    "tropical palm and banana vegetation": _only_trained([
        "Thailand", "Philippines", "Indonesia", "Malaysia", "Vietnam",
        "Brazil", "Colombia", "Ecuador", "Costa Rica", "Sri Lanka",
        "India", "Bangladesh", "Nepal", "Nicaragua", "Honduras",
        "Sierra Leone", "Ghana", "Senegal", "Nigeria", 
    ]),
    "African savanna grassland with acacia trees": _only_trained([
        "Rwanda", "South Africa", "Zambia", "Uganda", "Ethiopia",
        "Mozambique", "Ghana", "Senegal", "Nigeria", "Mali", "Lesotho", "Kenya"
    ]),
    "cerrado with scattered trees and red soil": _only_trained([
        "Brazil", "Paraguay", "Bolivia, Plurinational State of",
    ]),
    "sparse scrubby vegetation in arid climate": _only_trained([
        "Saudi Arabia", "United Arab Emirates", "Qatar", "Kuwait", "Bahrain",
        "Oman", "Jordan", "Egypt", "Algeria", "Morocco", "Chile", "Mongolia",
        "Kazakhstan", "Uzbekistan", "Turkmenistan", "Argentina",
        "South Africa", "Lesotho", "Australia", "India"
    ]),
    "flat farmland with hedgerows": _only_trained([
        "France", "United Kingdom", "Germany", "Netherlands", "Belgium",
        "Poland", "Hungary", "Ireland", "Denmark", "Czechia", "Slovakia",
        "Austria",
    ]),
    "dense hillside jungle or monsoon forest, broadleaf canopy without visible palms": _only_trained([
        "Sri Lanka", "Nepal", "India", "Rwanda", "Vietnam", "Philippines",
        "Indonesia", "Malaysia", "Thailand", "Bangladesh", "Mozambique",
        "Uganda", "Tanzania", "Ethiopia", "Costa Rica", "Honduras",
        "Nicaragua", "Colombia", "Ecuador", "Mauritius",
    ]),
}

# Boundaries ----------------------------------------------------------------------
BOUNDARY_STYLE = {
    "high concrete or brick wall topped with razor wire or electric fencing": _only_trained([
        "South Africa", "Zambia", "Uganda", "Ghana", "Nigeria", "Mozambique",
        "Kenya",  # (only if trained)
    ]),
    "plain wire or post-and-rail rural fencing": _only_trained([
        "Australia", "New Zealand", "United States", "Canada",
    ]),
}


def _normalize(country: str) -> str:
    """Strip any stray parenthetical qualifiers so lists stay consistent."""
    return re.sub(r"\s*\(.*?\)", "", country).strip()


def _add_votes(scores: dict, countries: list, weight: float, normalize_by_pool=False):
    if not countries:
        return
    per_country_weight = weight / len(countries) if normalize_by_pool else weight
    for c in countries:
        name = _normalize(c)
        scores[name] = scores.get(name, 0) + per_country_weight


def narrow_candidates(clip_results, road_markings=None, model_guesses=None, top_n=8):
    scores = {}
    contributing_clues = []  # each entry: (label, category, confidence, pool)

    MIN_ABS_CONFIDENCE = 0.5

    if model_guesses:
        for g in model_guesses:
            conf = g["confidence"]
            weight = conf * 10 + (conf ** 4) * 25
            scores[g["country"]] = scores.get(g["country"], 0) + weight

    if clip_results.get("signage_style") and len(clip_results["signage_style"]) >= 2:
        top_pred, second_pred = clip_results["signage_style"][0], clip_results["signage_style"][1]
        top_sign, sign_conf = top_pred["label"], top_pred["confidence"]
        margin = sign_conf - second_pred["confidence"]
        if margin >= 0.15 and sign_conf >= MIN_ABS_CONFIDENCE:
            for color in ("blue", "yellow", "white", "green"):
                if f"{color} background" in top_sign:
                    pool = SIGN_BACKGROUND_COLOR.get(color, [])
                    _add_votes(scores, pool, weight=2 * sign_conf * 10)
                    contributing_clues.append((top_sign, "signage_style", sign_conf, pool))
                    break

    used_real_road_signal = False
    if road_markings and road_markings.get("any_markings_detected"):
        yellow = road_markings.get("yellow_line")
        white = road_markings.get("white_line")
        yellow_conf = yellow.get("confidence", 0) if yellow else 0
        white_conf = white.get("confidence", 0) if white else 0

        if yellow_conf >= 0.25 and white_conf >= 0.25:
            combo_conf = (yellow_conf + white_conf) / 2
            pool = ROAD_LINE_COMBO["white_center_yellow_edge"]
            _add_votes(scores, pool, weight=4 * combo_conf * 10, normalize_by_pool=True)
            contributing_clues.append(("white + yellow road lines together (sensor)", "road_marking_combo", combo_conf, pool))
            used_real_road_signal = True

        if not used_real_road_signal and (yellow_conf >= 0.4 or white_conf >= 0.4):
            if yellow_conf >= white_conf:
                pool = ROAD_LINE_COLOR["yellow_center"]
                _add_votes(scores, pool, weight=3 * yellow_conf * 10, normalize_by_pool=True)
                contributing_clues.append(("yellow center line (sensor)", "road_marking", yellow_conf, pool))
            else:
                pool = ROAD_LINE_COLOR["white_center"]
                _add_votes(scores, pool, weight=3 * white_conf * 10, normalize_by_pool=True)
                contributing_clues.append(("white center line (sensor)", "road_marking", white_conf, pool))
            used_real_road_signal = True

    if not used_real_road_signal and clip_results.get("road_infrastructure"):
        top_road = clip_results["road_infrastructure"][0]["label"]
        road_conf = clip_results["road_infrastructure"][0]["confidence"]
        if road_conf >= MIN_ABS_CONFIDENCE:
            if "yellow center line" in top_road:
                pool = ROAD_LINE_COLOR["yellow_center"]
                _add_votes(scores, pool, weight=1.5)
                contributing_clues.append((top_road, "road_infrastructure", road_conf, pool))
            elif "white dashed center line" in top_road:
                pool = ROAD_LINE_COLOR["white_center"]
                _add_votes(scores, pool, weight=1.5)
                contributing_clues.append((top_road, "road_infrastructure", road_conf, pool))

    if clip_results.get("architecture") and len(clip_results["architecture"]) >= 2:
        preds = clip_results["architecture"]
        top_pred, second_pred = preds[0], preds[1]
        margin = top_pred["confidence"] - second_pred["confidence"]
        if margin >= 0.15 and top_pred["confidence"] >= MIN_ABS_CONFIDENCE:
            pool = ARCHITECTURE_STYLE.get(top_pred["label"])
            if pool:
                _add_votes(scores, pool, weight=1.5 * top_pred["confidence"] * 10, normalize_by_pool=True)
                contributing_clues.append((top_pred["label"], "architecture", top_pred["confidence"], pool))

    if clip_results.get("boundary_style") and len(clip_results["boundary_style"]) >= 2:
        preds = clip_results["boundary_style"]
        top_pred, second_pred = preds[0], preds[1]
        margin = top_pred["confidence"] - second_pred["confidence"]
        if margin >= 0.15 and top_pred["confidence"] >= MIN_ABS_CONFIDENCE:
            pool = BOUNDARY_STYLE.get(top_pred["label"])
            if pool:
                _add_votes(scores, pool, weight=1.5 * top_pred["confidence"] * 10, normalize_by_pool=True)
                contributing_clues.append((top_pred["label"], "boundary_style", top_pred["confidence"], pool))

    if clip_results.get("vegetation") and len(clip_results["vegetation"]) >= 2:
        preds = clip_results["vegetation"]
        top_pred, second_pred = preds[0], preds[1]
        margin = top_pred["confidence"] - second_pred["confidence"]

        if margin >= 0.15 and top_pred["confidence"] >= MIN_ABS_CONFIDENCE:
            pool = VEGETATION_REGION.get(top_pred["label"])
            if pool:
                _add_votes(scores, pool, weight=1 * top_pred["confidence"] * 10, normalize_by_pool=True)
                contributing_clues.append((top_pred["label"], "vegetation", top_pred["confidence"], pool))
        else:
            for pred in preds[:3]:
                if pred["confidence"] < top_pred["confidence"] - 0.2:
                    continue
                pool = VEGETATION_REGION.get(pred["label"])
                if pool:
                    _add_votes(scores, pool, weight=0.6 * pred["confidence"] * 10)
                    contributing_clues.append((pred["label"], "vegetation", pred["confidence"], pool))

    if not scores:
        return [], []

    print("[DEBUG] final scores:", sorted(scores.items(), key=lambda x: -x[1]), flush=True)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [country for country, _ in ranked[:top_n]], contributing_clues