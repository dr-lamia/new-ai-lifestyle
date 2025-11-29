# streamlit_app_behaviors.py — Dental AI Coach (Behaviours + SES + Explainability)
# with RandomForest + XGBoost + Blend + Final Insights

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import SES utilities
import ses_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

# Try XGBoost (optional)
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

# ============================ CONFIG =============================
st.set_page_config(page_title="AI tool for personalized awareness and education", page_icon="🦷", layout="wide")

DATA_PATH = "data/no_recommendation_dental_dataset_cleaned_keep_including_wisdom.csv"
TARGET_COL = "elham_s_index_including_wisdom"
ID_COLS = ["id"]

# Behaviour feature list (column names present in your dataset)
BEHAVIOR_COLS = [
    "tooth_brushing_frequency", "time_of_tooth_brushing", "interdental_cleaning", "mouth_rinse",
    "snacks_frequency", "snack_content", "sugar", "sticky_food", "carbonated_beverages",
    "type_of_diet", "hydration", "salivary_ph", "salivary_consistency", "buffering_capacity",
    "mutans_load_in_saliva", "lactobacilli_load_in_saliva"
]

# ======== ELHAM (subset) FIELDS YOU HAVE NOW ========
ELHAM_FIELDS_PRESENT = [
    "missing_0_including_wisdom_", "decayed_1", "filled_2",
    "hypoplasia_3", "hypocalcification_4", "fluorosis_5",
    "erosion_6", "abrasion_7", "attrition_8", "abfraction_9",
    "sealant_a", "fractured_h",
    "crown_pontic", "crown_abutment", "crown_implant",
    "veneer_f"
]

def compute_elham_from_inputs(values_dict: dict):
    per_item = {}
    total = 0.0
    for k in ELHAM_FIELDS_PRESENT:
        v = float(values_dict.get(k, 0) or 0)
        per_item[k] = v
        total += v
    return total, per_item

# ====================== BEHAVIOUR NORMALIZERS =====================
def _title(v): return str(v).strip().title()

def norm_yes_no(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"yes", "y", "true", "1"}: return "Yes"
    if s in {"no", "n", "false", "0"}: return "No"
    return "Unknown"

def norm_brushing_freq(v: str) -> str:
    s = str(v).strip().lower()
    if any(k in s for k in ["twice", "two", "2/day", "2 per", " 2 "]): return "2/day"
    if any(k in s for k in ["once", "one", "1/day", "1 per", " 1 "]): return "1/day"
    if "irreg" in s or "sometimes" in s: return "Irregular"
    return _title(v)

def norm_snack_freq(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"never", "none", "no", "0", "0/day"}: return "0/day"
    if any(k in s for k in ["1-2", "1–2", "sometimes", "occasional", "few"]): return "1–2/day"
    if any(k in s for k in [">2", "3+", "many", "often", "frequent", "a lot"]): return "3+/day"
    return _title(v)

def norm_risk_freq(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"none", "no", "never", "0"}: return "None"
    if any(k in s for k in ["occas", "some", "rare"]): return "Occasional"
    if any(k in s for k in ["freq", "often", "daily", "many"]): return "Frequent"
    return _title(v)

def norm_saliva_level(v: str) -> str:
    s = str(v).strip().lower()
    if "low" in s or "acid" in s: return "Low"
    if "high" in s: return "High"
    if "mod" in s: return "Moderate"
    return "Normal" if "normal" in s else _title(v)

def norm_mutans_lacto(v: str) -> str:
    s = str(v).strip().lower()
    if "more" in s or ">" in s or "10^5" in s or "10)5" in s: return "High"
    if "less" in s or "<" in s: return "Low"
    return "Normal" if "normal" in s else _title(v)

NORMALIZERS = {
    "tooth_brushing_frequency": norm_brushing_freq,
    "time_of_tooth_brushing": _title,
    "interdental_cleaning": norm_yes_no,
    "mouth_rinse": norm_yes_no,
    "snacks_frequency": norm_snack_freq,
    "snack_content": _title,
    "sugar": norm_risk_freq,
    "sticky_food": norm_yes_no,
    "carbonated_beverages": norm_risk_freq,
    "type_of_diet": _title,
    "hydration": _title,
    "salivary_ph": norm_saliva_level,
    "salivary_consistency": norm_saliva_level,
    "buffering_capacity": norm_saliva_level,
    "mutans_load_in_saliva": norm_mutans_lacto,
    "lactobacilli_load_in_saliva": norm_mutans_lacto,
}

def normalize_cats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c, f in NORMALIZERS.items():
        if c in df.columns:
            df[c] = df[c].astype(str).map(f)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df


# ========================= RISK TIERS =============================
def build_risk_bins(df, target_col=TARGET_COL):
    y = df[target_col].dropna().values
    q1, q2 = np.quantile(y, [0.34, 0.67])
    return (float(q1), float(q2))

def index_tier(y_hat, bins):
    low_u, mod_u = bins
    if y_hat < low_u: return "low"
    if y_hat < mod_u: return "moderate"
    return "high"

def tier_plan(tier):
    if tier == "high":
        return dict(
            recall="1–3 months",
            toothpaste="High-fluoride (2800–5000 ppm) twice daily",
            rinse="0.05% NaF daily (or 0.2% weekly) + consider CHX as indicated",
            varnish="Professional fluoride varnish every 3 months",
            diet_focus="Strong sugar reduction + avoid acidic drinks",
        )
    if tier == "moderate":
        return dict(
            recall="3–6 months",
            toothpaste="1450 ppm fluoride twice daily",
            rinse="0.05% NaF daily",
            varnish="Fluoride varnish every 6 months",
            diet_focus="Reduce between-meal snacks, especially sticky sweets",
        )
    return dict(
        recall="6–12 months",
        toothpaste="1450 ppm fluoride twice daily",
        rinse="Optional fluoride rinse if enamel defects or ortho",
        varnish="Varnish at routine intervals if indicated",
        diet_focus="Maintain current habits; keep sweets with meals",
    )

def treatment_plan_from_elham(counts: dict, tier: str):
    n = lambda k: int(counts.get(k, 0) or 0)
    out = []
    plan = tier_plan(tier)
    out += [
        f"**Overall plan for {tier.title()} risk:**",
        f"- Recall: **{plan['recall']}**",
        f"- Toothpaste: **{plan['toothpaste']}**",
        f"- Mouthrinse: **{plan['rinse']}**",
        f"- Varnish: **{plan['varnish']}**",
        f"- Diet focus: **{plan['diet_focus']}**",
        "—"
    ]
    if n("decayed_1") > 0:
        out += [f"• Caries on **{n('decayed_1')}** tooth/teeth → **restore** (GIC/composite); pulpal diagnosis if deep.", "  - Add fluoride varnish and sugar frequency counseling."]
    if n("filled_2") > 0:
        out += [f"• **{n('filled_2')}** restoration(s) → check margins; repair/polish if needed."]
    if n("hypoplasia_3") > 0 or n("hypocalcification_4") > 0:
        out += [f"• Enamel defects (hypoplasia: {n('hypoplasia_3')}, hypocalcification: {n('hypocalcification_4')}) →", "  - **Sealants / resin infiltration**, fluoride varnish for sensitivity."]
    if n("fluorosis_5") > 0:
        out += [f"• **Fluorosis** ({n('fluorosis_5')}) → microabrasion ± external bleaching."]
    if n("erosion_6") > 0:
        out += [f"• **Erosion** ({n('erosion_6')}) → acid control, straw use, rinse water; high-fluoride paste; restore if dentin exposed."]
    if n("abrasion_7") > 0:
        out += [f"• **Abrasion** ({n('abrasion_7')}) → brushing technique coaching; soft brush; desensitizing paste; restore if needed."]
    if n("attrition_8") > 0:
        out += [f"• **Attrition** ({n('attrition_8')}) → assess parafunction; **night guard** consideration; restore worn facets if indicated."]
    if n("abfraction_9") > 0:
        out += [f"• **Abfraction** ({n('abfraction_9')}) → manage occlusal load; restore if sensitive or deep."]
    if n("fractured_h") > 0:
        out += [f"• **Fracture** ({n('fractured_h')}) → immediate protection; definitive onlay/crown after assessment."]
    if n("sealant_a") > 0:
        out += [f"• **Sealants** present ({n('sealant_a')}) → check retention; re-seal where partial loss."]
    if n("missing_0_including_wisdom_") > 0:
        out += [f"• **Missing teeth** (incl. wisdom): {n('missing_0_including_wisdom_')} → discuss replacement needs."]
    if n("crown_pontic") > 0 or n("crown_abutment") > 0:
        out += [f"• **Bridge** (pontic {n('crown_pontic')}, abutment {n('crown_abutment')}) → super-floss/interdental brushes; margin review."]
    if n("crown_implant") > 0:
        out += [f"• **Implant crowns** ({n('crown_implant')}) → implant maintenance; peri-implant screening."]
    if n("veneer_f") > 0:
        out += [f"• **Veneers** ({n('veneer_f')}) → hygiene at margins; composite repair if chipping."]
    out += ["—", "• Reinforce personalized diet & hygiene education (see behaviour section).", f"• Recall per tier: **{plan['recall']}**."]
    return out

# ==================== PREPROCESS / TRAINING =======================
def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Dataset not found at '{path}'. Put the CSV in a 'data' folder with this exact name.")
        st.stop()
    df = pd.read_csv(path)
    return normalize_cats(df)

def split_feature_types(df: pd.DataFrame):
    cat_cols = [c for c in BEHAVIOR_COLS if c in df.columns]
    num_cols = df.select_dtypes(exclude="object").columns.tolist()
    num_cols = [c for c in num_cols if c not in ID_COLS + [TARGET_COL]]
    if TARGET_COL not in df.columns:
        st.error(f"Target column '{TARGET_COL}' not found.")
        st.stop()
    return num_cols, cat_cols

def build_rf_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num_cols)] +
        ([("cat", make_ohe(), cat_cols)] if len(cat_cols) > 0 else []),
        remainder="drop", verbose_feature_names_out=True
    )
    reg = RandomForestRegressor(n_estimators=450, random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("reg", reg)])

def build_xgb_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num_cols)] +
        ([("cat", make_ohe(), cat_cols)] if len(cat_cols) > 0 else []),
        remainder="drop", verbose_feature_names_out=True
    )
    reg = XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=5,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist"
    )
    return Pipeline([("pre", pre), ("reg", reg)])

@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame, cat_cols_override=None, drop_num_cols=None):
    num_cols, beh_cat_cols = split_feature_types(df)
    if drop_num_cols:
        num_cols = [c for c in num_cols if c not in set(drop_num_cols)]
    cat_cols = (cat_cols_override[:] if cat_cols_override else beh_cat_cols[:])

    X = df[num_cols + cat_cols].copy()
    y = df[TARGET_COL].astype(float).values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_pipe = build_rf_pipeline(num_cols, cat_cols)
    rf_pipe.fit(X_tr, y_tr)
    y_rf = rf_pipe.predict(X_te)
    metrics_rf = {"R2": float(r2_score(y_te, y_rf)), "MAE": float(mean_absolute_error(y_te, y_rf))}

    if XGB_OK:
        xgb_pipe = build_xgb_pipeline(num_cols, cat_cols)
        xgb_pipe.fit(X_tr, y_tr)
        y_xgb = xgb_pipe.predict(X_te)
        metrics_xgb = {"R2": float(r2_score(y_te, y_xgb)), "MAE": float(mean_absolute_error(y_te, y_xgb))}
        y_blend = 0.5 * (y_rf + y_xgb)
        metrics_blend = {"R2": float(r2_score(y_te, y_blend)), "MAE": float(mean_absolute_error(y_te, y_blend))}
    else:
        xgb_pipe = None
        metrics_xgb = None
        metrics_blend = None

    feat_names = rf_pipe.named_steps["pre"].get_feature_names_out().tolist()
    risk_bins = build_risk_bins(df, TARGET_COL)

    metrics_all = {"RandomForest": metrics_rf}
    if XGB_OK:
        metrics_all["XGBoost"] = metrics_xgb
        metrics_all["Blend (avg RF+XGB)"] = metrics_blend

    return rf_pipe, xgb_pipe, metrics_all, num_cols, cat_cols, feat_names, risk_bins

# ========================= SHAP GROUPING ==================
def build_group_map(feature_names, num_cols, cat_cols):
    group_map = {}
    for i, name in enumerate(feature_names):
        if name.startswith("num__"):
            orig = name[len("num__"):]
        elif name.startswith("cat__"):
            orig = None
            for c in cat_cols:
                prefix = f"cat__{c}_"
                if name.startswith(prefix):
                    orig = c
                    break
            if orig is None: orig = name
        else:
            orig = name
        group_map.setdefault(orig, []).append(i)
    return group_map

def group_shap_by_original(feature_names, shap_vals, num_cols, cat_cols):
    group_map = build_group_map(feature_names, num_cols, cat_cols)
    arr = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    row = arr[0] if getattr(arr, "ndim", 1) == 2 else arr
    grouped = {orig: float(np.sum(row[idxs])) for orig, idxs in group_map.items()}
    return sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)

def plot_bar(items, title):
    if not items:
        st.info("Nothing to show.")
        return
    fig, ax = plt.subplots()
    ax.bar([k for k, _ in items], [v for _, v in items])
    ax.set_xticklabels([k for k, _ in items], rotation=45, ha="right")
    ax.set_ylabel("Grouped value")
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)

# =================== DETAILED ADVICE (ALL BEHAVIOURS) =============
def detailed_behavior_recommendations(all_behaviors: dict, tier: str):
    plan = tier_plan(tier)
    recs = [
        f"**Overall plan for {tier.title()} risk:**",
        f"- Recall: **{plan['recall']}**",
        f"- Toothpaste: **{plan['toothpaste']}**",
        f"- Mouthrinse: **{plan['rinse']}**",
        f"- Varnish: **{plan['varnish']}**",
        f"- Diet focus: **{plan['diet_focus']}**",
    ]
    
    # Behavior logic
    for name, val in all_behaviors.items():
        v = str(val).lower()
        if name == "tooth_brushing_frequency":
            if "1" in v or "once" in v:
                recs.append("- Brush **twice daily** with fluoride toothpaste.")
            elif "irregular" in v:
                recs.append("- Set **fixed times** for brushing.")
        if name == "interdental_cleaning" and ("no" in v or "unknown" in v):
             recs.append("- Start **daily interdental cleaning**.")
        if name == "snacks_frequency" and ("3" in v or "often" in v):
            recs.append("- **Cut snacks** to max 2/day; sweets with meals.")
        if name == "sugar" and ("freq" in v or "daily" in v):
            recs.append("- **Reduce added sugars**.")
        if name == "carbonated_beverages" and ("freq" in v or "daily" in v):
            recs.append("- **Limit acidic drinks**; use a straw.")
            
    return recs

# =============== DATASET-DRIVEN UI OPTIONS ========================
def _mode_or_unknown(series: pd.Series) -> str:
    m = series.dropna().astype(str)
    return m.mode().iloc[0] if not m.empty and not m.mode().empty else "Unknown"

PREFERRED_ORDER = {
    "tooth_brushing_frequency": ["Irregular", "1/day", "2/day"],
    "interdental_cleaning": ["No", "Yes"],
    "mouth_rinse": ["No", "Yes"],
    "snacks_frequency": ["0/day", "1–2/day", "3+/day"],
    "sugar": ["None", "Occasional", "Frequent"],
    "carbonated_beverages": ["None", "Occasional", "Frequent"],
    "sticky_food": ["No", "Yes"],
    "hydration": ["Low", "Normal", "High"],
    "salivary_ph": ["Low", "Normal", "High"],
    "pocket_band": ["Low", "Medium", "High", "Unknown"], 
    "income_band": ["Low", "Medium", "High", "Unknown"],
}

def build_options_from_df(df: pd.DataFrame, cols):
    out = {}
    for c in cols:
        if c in df.columns:
            vals = df[c].dropna().astype(str).unique().tolist()
            if "Unknown" in vals: vals.remove("Unknown"); vals.append("Unknown")
            
            pref = PREFERRED_ORDER.get(c)
            if pref:
                seen = set()
                ordered = [v for v in pref if v in vals and not (v in seen or seen.add(v))]
                ordered += [v for v in vals if v not in set(pref)]
                vals = ordered
            else:
                vals = sorted(vals)
            out[c] = vals if vals else ["Unknown"]
        else:
            out[c] = ["Unknown"]
    return out

def default_from_df(df: pd.DataFrame, col: str) -> str:
    return _mode_or_unknown(df[col]) if col in df.columns else "Unknown"


# =============================== UI ===============================
st.title("🦷 AI tool for personalized awareness and education")

if not XGB_OK:
    st.caption("ℹ️ XGBoost not installed — using RandomForest only.")

if st.button("🔁 Force clear cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

df = load_data(DATA_PATH)

# SES prep using the module
idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42)
df, ses_cat_cols, raw_numeric_ses, ses_meta = ses_utils.prepare_ses(df, train_idx=idx_train)

# UI lists and dataset-driven options
beh_cols = [c for c in BEHAVIOR_COLS if c in df.columns]
cat_cols_all = beh_cols + [c for c in ses_cat_cols if c not in beh_cols]
beh_options = build_options_from_df(df, beh_cols)

# SES options will be built via ses_utils helper or manually if preferred
ses_cols_ui = [c for c in cat_cols_all if c not in beh_cols]
ses_options = build_options_from_df(df, ses_cols_ui)

# Train models
rf_pipe, xgb_pipe, metrics_all, num_cols, all_cat_cols, feat_names, risk_bins = \
    train_models(df, cat_cols_override=cat_cols_all, drop_num_cols=raw_numeric_ses)

# Model selector
model_choices = ["RandomForest"]
if XGB_OK:
    model_choices += ["XGBoost", "Blend (avg RF+XGB)"]
model_choice = st.selectbox("Model to use", options=model_choices)

# Show metrics
mcols = st.columns(len(model_choices))
for i, name in enumerate(model_choices):
    m = metrics_all.get(name)
    if m:
        with mcols[i]:
            st.metric(name, f"R² {m['R2']:.3f}", delta=f"MAE {m['MAE']:.2f}")

def predict_with_choice(X):
    if model_choice == "RandomForest":
        return rf_pipe.predict(X)
    elif model_choice == "XGBoost" and XGB_OK:
        return xgb_pipe.predict(X)
    else:  # Blend
        if XGB_OK:
            return 0.5 * (rf_pipe.predict(X) + xgb_pipe.predict(X))
        return rf_pipe.predict(X)

def active_pipe_for_explain():
    if model_choice == "RandomForest" or not XGB_OK:
        return rf_pipe, "RandomForest"
    if model_choice == "XGBoost":
        return xgb_pipe, "XGBoost"
    best = max(("RandomForest", "XGBoost"), key=lambda k: metrics_all[k]["R2"])
    return (rf_pipe if best == "RandomForest" else xgb_pipe), best

# ========================= MODEL VISUALIZATIONS ===================
st.subheader("Model visualizations")
tab_perf, tab_imp = st.tabs(["📈 Performance", "⭐ Global importance"])
with tab_perf:
    st.write("Performance plots loaded.") # (Placeholder for plots)

# ------------------------ INPUT UI -----------------------
st.subheader("Enter Elham Index (counts)")
left, right = st.columns(2)
elham_core = {}
present_elham_fields = [c for c in ELHAM_FIELDS_PRESENT if c in df.columns]
mid = len(present_elham_fields) // 2
for k in present_elham_fields[:mid]:
    with left:
        elham_core[k] = st.number_input(k, min_value=0, step=1, value=0)
for k in present_elham_fields[mid:]:
    with right:
        elham_core[k] = st.number_input(k, min_value=0, step=1, value=0)

# Behaviours UI
st.subheader("Behavior & lifestyle inputs")
beh_vals, cols = {}, st.columns(2)
for i, c in enumerate(beh_cols):
    opts = beh_options.get(c, ["Unknown"])
    default = default_from_df(df, c)
    with cols[i % 2]:
        beh_vals[c] = st.selectbox(c, options=opts, index=(opts.index(default) if default in opts else 0))

# SES UI via Module
ses_vals, ses_cols_used = ses_utils.build_ses_ui(df, ses_cat_cols)

# ------------------------- PREDICT -----------------------
if st.button("Predict + Explain"):
    X_row = {c: float(elham_core.get(c, 0)) for c in num_cols if c in elham_core}
    # Fill remaining numerics with median
    for c in num_cols:
        if c not in X_row:
            X_row[c] = df[c].median()

    for c in beh_cols:
        X_row[c] = beh_vals.get(c, default_from_df(df, c))
    
    # Add SES values
    X_row = ses_utils.include_ses_in_row(X_row, ses_vals, ses_cols_used, df)
    
    X_df = pd.DataFrame([X_row])

    # 1) Compute Elham index
    entered_index, per_item = compute_elham_from_inputs(elham_core)
    st.info(f"**Computed Elham’s Index: {entered_index:.0f}**")

    # 2) Predict
    y_hat = float(predict_with_choice(X_df)[0])
    st.success(f"Model prediction ({model_choice}): **{y_hat:.2f}**")

    # 3) Tier & Plan
    tier = index_tier(entered_index, risk_bins)
    st.info(f"Risk tier: **{tier.title()}**")
    for line in treatment_plan_from_elham(per_item, tier):
        st.markdown(line)

    # 4) Explain
    try:
        import shap
        active_pipe, active_name = active_pipe_for_explain()
        pre = active_pipe.named_steps["pre"]
        X_trans = pre.transform(X_df)
        explainer = shap.TreeExplainer(active_pipe.named_steps["reg"])
        shap_vals = explainer.shap_values(X_trans)

        feat_names_active = pre.get_feature_names_out().tolist()
        grouped = group_shap_by_original(feat_names_active, shap_vals, num_cols, all_cat_cols)

        st.subheader(f"Top drivers (SHAP · {active_name})")
        plot_bar(grouped[:12], "Top drivers")
        
        # SES SHAP via module
        ses_utils.ses_shap_panel(grouped, ses_cols_used)

    except Exception as e:
        st.warning(f"Explanations skipped: {e}")
