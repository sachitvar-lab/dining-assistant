"""Calorie-Constrained Dining Assistant - Streamlit UI."""

import streamlit as st

from backend import KNOWN_CUISINES, has_constraints, load_menus, parse_query, recommend, verify

st.set_page_config(page_title="Calorie-Constrained Dining Assistant", page_icon="🍽️")

st.markdown("""
<style>
    /* hide default chrome */
    #MainMenu, footer {visibility: hidden;}
    .block-container {max-width: 760px; padding-top: 2.5rem;}

    /* header */
    .hero-title {font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em;
                 margin-bottom: 0.2rem; color: #1F2430;}
    .hero-sub {color: #6B7280; font-size: 1.02rem; line-height: 1.55; margin-bottom: 0.4rem;}

    /* example buttons */
    .stButton > button {border-radius: 999px; border: 1px solid #E3DDD2;
                        background: #FFFFFF; font-weight: 600; padding: 0.45rem 0.9rem;}
    .stButton > button:hover {border-color: #B33A3A; color: #B33A3A;}

    /* result cards */
    .result-card {background: #FFFFFF; border: 1px solid #ECE6DC; border-radius: 14px;
                  padding: 1.05rem 1.2rem; margin-bottom: 0.8rem;
                  box-shadow: 0 1px 3px rgba(31,36,48,0.05);}
    .item-name {font-weight: 700; font-size: 1.08rem; color: #1F2430;}
    .item-meta {color: #8A8F99; font-size: 0.86rem; margin: 0.1rem 0 0.55rem;}
    .cal-big {font-weight: 800; font-size: 1.5rem; color: #B33A3A; text-align: right;}
    .cal-label {color: #8A8F99; font-size: 0.78rem; text-align: right;}
    .price {color: #6B7280; font-size: 0.9rem; text-align: right; margin-top: 0.15rem;}
    .badge {display: inline-block; background: #EAF5EC; color: #1E7A37;
            border-radius: 999px; padding: 0.16rem 0.6rem; font-size: 0.78rem;
            font-weight: 600; margin: 0 0.3rem 0.3rem 0;}
    .budget-note {color: #8A8F99; font-size: 0.8rem; margin-top: 0.45rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🍽️ Calorie-Constrained Dining Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Tell me what you\'re craving, your calorie budget, and any '
    'dietary restrictions — I\'ll suggest orders from New Haven restaurants that fit. '
    'Constraint matching is exact and verified against the dataset; local restaurants '
    'don\'t publish nutrition data, so calorie values are transparent, fixed estimates — '
    'the assistant never invents a number at answer time.</div>',
    unsafe_allow_html=True,
)

# Resolve the API key: environment first, then Streamlit secrets -
# but only ask st.secrets if a secrets file actually exists, otherwise
# Streamlit renders a "No secrets found" error box.
import os

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    _secret_paths = [
        os.path.expanduser("~/.streamlit/secrets.toml"),
        os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml"),
    ]
    if any(os.path.exists(p) for p in _secret_paths):
        try:
            api_key = st.secrets.get("OPENAI_API_KEY") or None
        except Exception:
            api_key = None

df = load_menus()

# --- example queries -------------------------------------------------------
st.caption("Try one:")
col1, col2, col3 = st.columns(3)
examples = {
    "🍕 Pizza under 700 cal": "pizza under 700 calories",
    "🇮🇳 Indian < 650, no dairy": "Indian under 650 calories, no dairy",
    "🍜 Asian < 600, no seafood": "Asian under 600 calories, no seafood",
}
if "query" not in st.session_state:
    st.session_state.query = ""
for col, (label, q) in zip([col1, col2, col3], examples.items()):
    if col.button(label, use_container_width=True):
        st.session_state.query = q

query = st.text_input(
    "What are you looking for?",
    value=st.session_state.query,
    placeholder="e.g. a high-protein lunch under 500 calories, no nuts",
)

# --- results ---------------------------------------------------------------
if query:
    with st.spinner("Parsing your constraints and searching menus..."):
        params = parse_query(query, api_key=api_key)
        recs = recommend(df, params) if has_constraints(params) else None

    parsed_bits = []
    if params["cuisine"]:
        parsed_bits.append(f"cuisine: **{params['cuisine'].title()}**")
    if params["max_calories"]:
        parsed_bits.append(f"budget: **{params['max_calories']} cal**")
    if params["exclusions"]:
        parsed_bits.append("excluding: **" + ", ".join(params["exclusions"]) + "**")
    if params["vegetarian"]:
        parsed_bits.append("**vegetarian**")
    if parsed_bits:
        st.caption("Understood as — " + " · ".join(parsed_bits))
    if not api_key:
        st.caption("⚙️ Running on the keyword parser (no API key configured). Add an OpenAI key in secrets for full natural-language parsing.")

    if params.get("unsupported_cuisine"):
        st.warning(
            f"I don't have **{params['unsupported_cuisine'].title()}** restaurants in the "
            "current dataset, and I won't substitute something else and call it a match. "
            "Available cuisines: " + ", ".join(c.title() for c in KNOWN_CUISINES) + "."
        )
    elif recs is None:
        st.info(
            "I couldn't find a constraint in that — tell me a cuisine, a calorie "
            "budget, or a restriction (e.g. 'Mexican under 600 calories, no dairy')."
        )
    elif recs.empty:
        st.warning(
            "No menu items satisfy every constraint. That's by design — "
            "this assistant returns exact matches or nothing, never approximations. "
            "Try relaxing the calorie budget or an exclusion."
        )
    else:
        st.subheader(f"{len(recs)} verified match{'es' if len(recs) > 1 else ''}")
        for _, row in recs.iterrows():
            checks = verify(row, params)
            badges = "".join(f'<span class="badge">✓ {label}</span>' for label, ok in checks.items() if ok)
            budget_note = ""
            if params.get("max_calories"):
                remaining = params["max_calories"] - row["Calories"]
                budget_note = f'<div class="budget-note">{remaining} cal under your {params["max_calories"]}-cal budget</div>'
            st.markdown(f"""
<div class="result-card">
  <div style="display:flex; justify-content:space-between; gap:1rem;">
    <div style="flex:1;">
      <div class="item-name">{row['Item']}</div>
      <div class="item-meta">{row['Restaurant']} · {row['Cuisine']}</div>
      <div>{badges}</div>
      {budget_note}
    </div>
    <div style="min-width:90px;">
      <div class="cal-big">{row['Calories']}</div>
      <div class="cal-label">calories (est.)</div>
      <div class="price">${row['Price']:.2f}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

st.divider()
st.caption(
    "Built by Sachit Varma · Yale SOM '26 · "
    "[GitHub](https://github.com/sachitvar-lab) · "
    "New Haven restaurants · Calorie values are good-faith fixed estimates — "
    "local venues don't publish nutrition data, which is the gap this project explores."
)
