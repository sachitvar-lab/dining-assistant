# Calorie-Constrained Dining Assistant

Natural-language restaurant recommendations that **provably** satisfy calorie
budgets and dietary restrictions. Ask for "a Japanese meal under 750 calories,
no soy" and get exact menu items from real published nutrition data — never
generated or approximated numbers.

**Architecture** (unstructured input → structured extraction → grounded verification):

1. **Parse** — an LLM (gpt-4o-mini) extracts the query into structured JSON:
   cuisine, calorie ceiling, exclusions, vegetarian flag. Output is validated
   field-by-field; malformed responses degrade to a keyword parser rather than
   passing bad data downstream.
2. **Retrieve** — deterministic pandas filtering over the menu dataset. The
   LLM never invents a calorie count; it only ever selects from real data.
3. **Verify** — every recommendation is programmatically re-checked against
   every stated constraint before display (the badges in the UI).

## Run locally

    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...        # optional; keyword parser works without it
    streamlit run app.py

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. share.streamlit.io → New app → select repo, main file `app.py`.
3. App Settings → Secrets → add: `OPENAI_API_KEY = "sk-..."`

## Data

`data/menus.csv` — 45 items across 15 New Haven restaurants. Local
restaurants don't publish nutrition data — the information gap that motivated
this project — so calorie values are good-faith fixed estimates and the app
discloses this. Constraint satisfaction (calorie ceiling, allergen exclusions,
cuisine) is exact and programmatically verified against the dataset. Swap in
any CSV with
columns `Restaurant, Item, Cuisine, Calories, Price, Tags` (Tags
semicolon-separated: soy, seafood, gluten, dairy, nuts, egg, sesame,
vegetarian, vegan, gluten-free).

## Lineage

v2 of a Yale SOM team project (with Parmjot Gill and Shruti Shambhavi).
Rebuilt solo: defensive type coercion at the data boundary, validated LLM
output, graceful no-key fallback, pinned dependencies so the deployment
doesn't rot.
