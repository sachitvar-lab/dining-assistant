"""
Backend for the Calorie-Constrained Dining Assistant.

Pipeline:
  1. parse_query()  - LLM extracts structured parameters from natural language
                      (falls back to a keyword parser if no API key is set)
  2. recommend()    - deterministic pandas filtering against the menu dataset
  3. verify()       - programmatic check that every recommendation satisfies
                      every stated constraint (nothing is taken on trust)
"""

import json
import os
import re

import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "menus.csv")

KNOWN_CUISINES = ["italian", "indian", "japanese", "thai", "mediterranean", "american", "asian", "mexican"]
KNOWN_EXCLUSIONS = ["soy", "seafood", "gluten", "dairy", "nuts", "egg", "sesame"]

# Cuisines people commonly ask for; used to detect requests the dataset
# cannot serve so they fail loudly instead of being silently dropped.
RECOGNIZABLE_CUISINES = KNOWN_CUISINES + [
    "chinese", "korean", "vietnamese", "french", "greek", "turkish",
    "lebanese", "ethiopian", "spanish", "caribbean",
]

# Common food words that imply a cuisine in the dataset.
CUISINE_ALIASES = {"pizza": "italian", "apizza": "italian", "sushi": "japanese",
                   "ramen": "asian", "pho": "asian", "curry": "indian",
                   "falafel": "mediterranean", "shawarma": "mediterranean"}


# ---------------------------------------------------------------- data layer
def load_menus() -> pd.DataFrame:
    """Load the menu dataset and coerce numeric columns defensively.

    Coercion happens here, at the boundary, so type errors can never
    reach the recommendation logic (lesson learned from v1).
    """
    df = pd.read_csv(DATA_PATH)
    df["Calories"] = pd.to_numeric(df["Calories"], errors="coerce")
    df["Price"] = pd.to_numeric(
        df["Price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )
    df = df.dropna(subset=["Calories", "Price"]).copy()
    df["Calories"] = df["Calories"].astype(int)
    df["TagList"] = (
        df["Tags"].fillna("").str.lower().str.split(";").apply(lambda tags: [t.strip() for t in tags if t.strip()])
    )
    return df


# ------------------------------------------------------------- parsing layer
def _fallback_parse(query: str) -> dict:
    """Keyword-based parser used when no OpenAI key is configured.

    Less flexible than the LLM, but keeps the app fully functional
    for anyone running it without credentials.
    """
    q = query.lower()
    calorie_match = re.search(r"(\d{3,4})\s*(?:k?cal|calories?)?", q)
    max_calories = int(calorie_match.group(1)) if calorie_match else None

    requested = next((c for c in RECOGNIZABLE_CUISINES if c in q), None)
    if not requested:
        requested = next((cuisine for word, cuisine in CUISINE_ALIASES.items() if word in q), None)
    cuisine = requested if requested in KNOWN_CUISINES else None
    unsupported_cuisine = requested if requested and requested not in KNOWN_CUISINES else None
    exclusions = [e for e in KNOWN_EXCLUSIONS if re.search(rf"no\s+{e}|{e}[- ]free|without\s+{e}", q)]
    vegetarian = bool(re.search(r"vegetarian|veggie|plant[- ]based|vegan", q))

    return {
        "cuisine": cuisine,
        "unsupported_cuisine": unsupported_cuisine,
        "max_calories": max_calories,
        "exclusions": exclusions,
        "vegetarian": vegetarian,
    }


def parse_query(query, api_key=None):
    """Extract structured parameters from a natural-language query.

    Uses an LLM when a key is available; otherwise falls back to keywords.
    The LLM output is validated field-by-field before use - a malformed
    or hallucinated response degrades to the fallback parser rather than
    crashing or silently passing bad data downstream.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _fallback_parse(query)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract dining constraints from the user's request. "
                        "Respond ONLY with JSON: "
                        '{"cuisine": str|null (the cuisine requested, lowercase, '
                        'e.g. japanese, indian, mexican), '
                        '"max_calories": int|null, '
                        '"exclusions": [str] (subset of: soy, seafood, gluten, dairy, nuts, egg, sesame), '
                        '"vegetarian": bool}'
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
        )
        raw = json.loads(response.choices[0].message.content)

        # Validate every field; never trust model output blindly.
        raw_cuisine = raw.get("cuisine")
        raw_cuisine = raw_cuisine.lower() if isinstance(raw_cuisine, str) else None
        cuisine = raw_cuisine if raw_cuisine in KNOWN_CUISINES else None
        unsupported_cuisine = raw_cuisine if raw_cuisine and raw_cuisine not in KNOWN_CUISINES else None
        max_calories = raw.get("max_calories")
        max_calories = int(max_calories) if isinstance(max_calories, (int, float)) and max_calories > 0 else None
        exclusions = [e.lower() for e in raw.get("exclusions", []) if isinstance(e, str) and e.lower() in KNOWN_EXCLUSIONS]
        vegetarian = bool(raw.get("vegetarian", False))

        return {
            "cuisine": cuisine,
            "unsupported_cuisine": unsupported_cuisine,
            "max_calories": max_calories,
            "exclusions": exclusions,
            "vegetarian": vegetarian,
        }
    except Exception:
        return _fallback_parse(query)


# -------------------------------------------------------- recommendation layer
def recommend(df: pd.DataFrame, params: dict, top_n: int = 5) -> pd.DataFrame:
    """Filter the menu deterministically against the parsed constraints.

    Every constraint is applied as a pandas filter over real data -
    no generated calorie counts, no approximate matches.
    """
    result = df.copy()

    if params.get("cuisine"):
        result = result[result["Cuisine"].str.lower() == params["cuisine"]]

    if params.get("max_calories"):
        result = result[result["Calories"] <= params["max_calories"]]

    for exclusion in params.get("exclusions", []):
        result = result[~result["TagList"].apply(lambda tags: exclusion in tags)]

    if params.get("vegetarian"):
        result = result[result["TagList"].apply(lambda tags: "vegetarian" in tags or "vegan" in tags)]

    # Rank: closest to the calorie budget first (a 730-cal meal beats a
    # 230-cal snack when the budget is 750), then by price.
    if params.get("max_calories"):
        result = result.sort_values(["Calories", "Price"], ascending=[False, True])
    else:
        result = result.sort_values("Price")

    return result.head(top_n)


def verify(row: pd.Series, params: dict) -> dict:
    """Programmatically re-check each constraint for a recommendation.

    Returns a dict of constraint -> bool used to render the verification
    badges in the UI. This is the explicit 'don't take it on trust' step.
    """
    checks = {}
    if params.get("max_calories"):
        checks[f"≤ {params['max_calories']} cal"] = row["Calories"] <= params["max_calories"]
    if params.get("cuisine"):
        checks[params["cuisine"].title()] = row["Cuisine"].lower() == params["cuisine"]
    for exclusion in params.get("exclusions", []):
        checks[f"no {exclusion}"] = exclusion not in row["TagList"]
    if params.get("vegetarian"):
        checks["vegetarian"] = "vegetarian" in row["TagList"] or "vegan" in row["TagList"]
    return checks


def has_constraints(params):
    """True if the user expressed at least one actionable constraint."""
    return bool(
        params.get("cuisine")
        or params.get("max_calories")
        or params.get("exclusions")
        or params.get("vegetarian")
    )
