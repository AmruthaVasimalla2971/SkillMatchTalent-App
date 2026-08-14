"""
SkillMatch Talent — Streamlit deployment app
Doing Business with AI — Group 09

Loads the trained pipeline (skillmatch_best_model.pkl) and the bias-audit
summary (bias_audit_summary.csv), both produced by DBAI.ipynb. Takes a
candidate profile as input, returns a predicted employability score, and
always shows the model-level bias-audit disclosure alongside it.

IMPORTANT: sex, race, and disability status are never used as inputs to
the score. They were excluded from training and are only used, on a
held-out test set, to audit the model afterward. The "bias flag" below
describes the model in general — it is not recomputed per prediction.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "skillmatch_best_model.pkl"
BIAS_SUMMARY_PATH = "bias_audit_summary.csv"

st.set_page_config(page_title="SkillMatch Talent", page_icon="🎯", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_bias_summary():
    if os.path.exists(BIAS_SUMMARY_PATH):
        return pd.read_csv(BIAS_SUMMARY_PATH)
    # Fallback if the CSV wasn't uploaded to the Space — replace it with
    # the real bias_audit_summary.csv from Phase 3 for live figures.
    return pd.DataFrame(
        {
            "Attribute": ["SEX", "RAC1P", "DIS"],
            "Four-fifths ratio": [0.901, 0.599, 0.450],
            "Flagged": [False, True, True],
        }
    )


model = load_model()
bias_summary = load_bias_summary()

st.title("🎯 SkillMatch Talent")
st.caption(
    "Employability-scoring demo on US Census ACS PUMS data — "
    "Doing Business with AI, Group 09"
)
st.markdown(
    "Enter a candidate profile below. The model predicts the likelihood of "
    "**current employment** from demographic and human-capital features. "
    "Sex, race, and disability status are **not** used as inputs — they're "
    "only used afterward, on a held-out test set, to audit the model for bias. "
    "That audit is shown after every prediction."
)

STATE_OPTIONS = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa", "20": "Kansas",
    "21": "Kentucky", "22": "Louisiana", "23": "Maine", "24": "Maryland",
    "25": "Massachusetts", "26": "Michigan", "27": "Minnesota", "28": "Mississippi",
    "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada",
    "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York",
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma",
    "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island", "45": "South Carolina",
    "46": "South Dakota", "47": "Tennessee", "48": "Texas", "49": "Utah",
    "50": "Vermont", "51": "Virginia", "53": "Washington", "54": "West Virginia",
    "55": "Wisconsin", "56": "Wyoming",
}

# Simplified to the 8 buckets used in bucket_schl(); expand to the full
# 24-level SCHL codebook if you want finer-grained input.
SCHL_OPTIONS = {
    "Less than high school": 10,
    "High school diploma": 16,
    "GED or alternative credential": 17,
    "Some college, no degree": 19,
    "Associate's degree": 20,
    "Bachelor's degree": 21,
    "Master's degree": 22,
    "Professional degree beyond bachelor's": 23,
    "Doctorate degree": 24,
}

CIT_OPTIONS = {
    "Born in the US": 1,
    "Born in Puerto Rico / US Island Areas": 2,
    "Born abroad to US-citizen parents": 3,
    "Naturalized US citizen": 4,
    "Not a US citizen": 5,
}

MAR_OPTIONS = {
    "Married": 1, "Widowed": 2, "Divorced": 3, "Separated": 4, "Never married": 5,
}

MIL_OPTIONS = {
    "On active duty now": 1,
    "On active duty in the past, not now": 2,
    "Reserves / National Guard only": 3,
    "Never served": 4,
}

ENG_OPTIONS = {
    "Very well": 1, "Well": 2, "Not well": 3, "Not at all": 4,
}

with st.form("profile_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=16, max_value=96, value=30)
        schl_choice = st.selectbox("Educational attainment", list(SCHL_OPTIONS.keys()), index=5)
        wkwn = st.slider("Weeks worked in the past 12 months", 0, 52, 40)
        state_choice = st.selectbox("State", list(STATE_OPTIONS.values()), index=4)
    with col2:
        nativity_choice = st.radio("Nativity", ["Native", "Foreign born"])
        cit_choice = st.selectbox("Citizenship status", list(CIT_OPTIONS.keys()), index=0)
        mar_choice = st.selectbox("Marital status", list(MAR_OPTIONS.keys()), index=4)
        mil_choice = st.selectbox("Military service", list(MIL_OPTIONS.keys()), index=3)

    lanx_choice = st.radio("Speaks a language other than English at home?", ["No, English only", "Yes"])
    eng_choice = None
    if lanx_choice == "Yes":
        eng_choice = st.selectbox("Self-reported English-speaking ability", list(ENG_OPTIONS.keys()), index=0)

    submitted = st.form_submit_button("Get employability score")

if submitted:
    state_code = [k for k, v in STATE_OPTIONS.items() if v == state_choice][0]
    row = pd.DataFrame(
        [
            {
                "AGEP": age,
                "WKWN": wkwn,
                "SCHL": SCHL_OPTIONS[schl_choice],
                "LANX": 1 if lanx_choice == "Yes" else 2,
                "ENG": ENG_OPTIONS[eng_choice] if eng_choice else np.nan,
                "CIT": CIT_OPTIONS[cit_choice],
                "NATIVITY": 1 if nativity_choice == "Native" else 2,
                "MIL": MIL_OPTIONS[mil_choice],
                "MAR": MAR_OPTIONS[mar_choice],
                "STATE": int(state_code),
            }
        ]
    )

    proba = model.predict_proba(row)[0, 1]
    pred = int(proba >= 0.5)

    st.divider()
    st.subheader("Result")
    st.metric("Predicted employability score", f"{proba:.1%}")
    st.write("**Predicted status:**", "✅ Likely employed" if pred == 1 else "⚠️ Likely not employed")
    st.progress(min(max(proba, 0.0), 1.0))

    st.divider()
    st.subheader("Bias audit disclosure")
    st.caption(
        "Demographic parity was measured on a held-out test set across sex, race, and "
        "disability status using the four-fifths rule (a positive-prediction-rate ratio "
        "below 0.80 across groups is flagged as potential disparate impact). This "
        "describes the model overall, not this specific prediction."
    )
    for _, r in bias_summary.iterrows():
        flag_icon = "🚩" if r["Flagged"] else "✅"
        status = "flagged" if r["Flagged"] else "not flagged"
        st.write(f"{flag_icon} **{r['Attribute']}** — four-fifths ratio: {r['Four-fifths ratio']:.3f} ({status})")

    if bias_summary["Flagged"].any():
        st.warning(
            "One or more protected attributes show a four-fifths ratio below 0.80 in our "
            "audit. See the project report for the full breakdown and discussion."
        )
