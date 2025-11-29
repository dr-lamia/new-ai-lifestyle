# ses_utils.py
# Lightweight helpers to integrate socio-economic (SES) factors in your Streamlit app.

import re
import numpy as np
import pandas as pd

# -------- fuzzy header map: canonical -> substrings to look for ----------
SES_FUZZY = {
    "school":              ["school"],
    "grade":               ["grade"],
    "house_ownership":     ["house_ow", "ownership", "rent", "own"],
    "i_live_with":         ["i_live_with", "live_with", "parents"],
    "average_income":      ["average_in", "family_in", "income", "household"],
    "pocket_money":        ["pocket_m", "allowance", "pocket"],
    "father_s_education":  ["father_s_e", "father edu", "father_edu"],
    "mother_s_education":  ["mother_s_e", "mother edu", "mother_edu"],
    "father_s_job":        ["father_s_j", "father job", "father occ"],
    "mother_s_job":        ["mother_s_j", "mother job", "mother occ"],
    "insurance":           ["insurance"],
    "access_to":           ["access_to", "access"],
    "frequency":           ["frequency", "visit"],
    "affordability":       ["afford"],
}

# -------------------- normalizers (compact categories) -------------------
def _t(x): return str(x).strip().lower()

def norm_school(v):
    s = _t(v)
    if any(k in s for k in ["public","government","gov","experimental"]): return "Public"
    if any(k in s for k in ["international","american","british","german","french","canadian","igcse","ib","intl"]): return "International"
    if "private" in s or "independent" in s: return "Private"
    return "Unknown"

def norm_grade(v):
    s = _t(v)
    if any(k in s for k in ["kg","nursery","primary","grade 1","grade 2","grade 3","grade 4","grade 5","grade 6"]): return "Primary"
    if any(k in s for k in ["prep","grade 7","grade 8","grade 9"]): return "Preparatory"
    if any(k in s for k in ["sec","grade 10","grade 11","grade 12"]): return "Secondary"
    return "Unknown"

def norm_house_ownership(v):
    s = _t(v)
    if "own" in s: return "Owned"
    if "rent" in s: return "Rented"
    return "Other/Unknown"

def norm_live_with(v):
    s = _t(v)
    if any(k in s for k in ["father and mother","both parents","two parents"]): return "Two parents"
    if any(k in s for k in ["single","mother only","father only","one parent"]): return "Single parent"
    if any(k in s for k in ["relative","grand","aunt","uncle","guardian","care"]): return "Relatives/Other"
    return "Unknown"

def _num(x):
    """
    Extract the first valid number from a string.
    Fixes issue where '3 to 5' became 35. Now returns 3.0.
    """
    s = str(x).strip()
    if not s or s.lower() in ["nan", "none", "unknown", "yes", "no"]:
        return None
    
    # Handle 'k' suffix for thousands (e.g., '5k' -> 5000)
    if s.lower().endswith("k"):
        try:
            return float(re.sub(r"[^\d.]", "", s)) * 1000.0
        except:
            pass
            
    # Regex to find the first sequence of digits (int or float)
    match = re.search(r"(\d+(\.\d+)?)", s)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None

def norm_parent_edu(v):
    s = _t(v)
    if any(k in s for k in ["phd","master","msc","postgrad","doctor"]): return "Postgrad"
    if any(k in s for k in ["uni","college","bsc","ba","license","licence"]): return "University"
    if any(k in s for k in ["secondary","high school","prep"]): return "Secondary"
    if "primary" in s or "elementary" in s: return "Primary"
    return "Unknown"

def norm_job(v):
    s = _t(v)
    if any(k in s for k in ["not working","no job","housewife","unemployed"]): return "Not working"
    if any(k in s for k in ["manager","engineer","doctor","dentist","pharmacist","teacher","accountant","lawyer"]): return "Professional/Manager"
    if s and s != "unknown": return "Worker/Clerk"
    return "Unknown"

def norm_insurance(v):
    s = _t(v)
    if s in {"yes","insured","y","1","covered"}: return "Insured"
    if s in {"no","uninsured","n","0"}:          return "Uninsured"
    return "Unknown"

def norm_access(v):
    s = _t(v)
    if any(k in s for k in ["easy","available","near"]): return "Easy"
    if any(k in s for k in ["hard","difficult","far","limited"]): return "Difficult"
    if s in {"moderate","average"}: return "Moderate"
    return "Unknown"

def norm_afford(v):
    s = _t(v)
    if any(k in s for k in ["cannot","can't","no","unaffordable"]): return "No"
    if any(k in s for k in ["hard","difficult","sometimes","partial"]): return "Hard"
    if any(k in s for k in ["yes","afford","can"]): return "Yes"
    return "Unknown"

def norm_visit_freq(v):
    s = _t(v)
    if any(k in s for k in ["6","12","regular","check","year","every"]): return "Regular"
    if any(k in s for k in ["pain","emergency","only when"]):           return "Pain-only"
    if "never" in s:                                                    return "Never"
    return "Occasional"

NORMALIZERS = {
    "school": norm_school,
    "grade": norm_grade,
    "house_ownership": norm_house_ownership,
    "i_live_with": norm_live_with,
    "father_s_education": norm_parent_edu,
    "mother_s_education": norm_parent_edu,
    "father_s_job": norm_job,
    "mother_s_job": norm_job,
    "insurance": norm_insurance,
    "access_to": norm_access,
    "affordability": norm_afford,
    "frequency": norm_visit_freq,
}

# ---------------------- column detection + prep --------------------------
def find_cols(df: pd.DataFrame, fuzzy_map=SES_FUZZY) -> dict:
    """Return {canonical_name: actual_column or None} using substring search."""
    low = {c: c.lower() for c in df.columns}
    out = {}
    for canon, keys in fuzzy_map.items():
        hit = None
        for c, lc in low.items():
            if any(k in lc for k in keys):
                hit = c; break
        out[canon] = hit
    return out

def prepare_ses(df: pd.DataFrame, train_idx=None):
    """
    Normalize SES text columns and create tertile bands for income and pocket_money.
    """
    df2 = df.copy()
    ses_map = find_cols(df2)
    
    # normalize text SES columns if present
    for k, col in ses_map.items():
        if col is None or col not in df2.columns: 
            continue
        if k in NORMALIZERS:
            df2[col] = df2[col].apply(NORMALIZERS[k])

    # compute income/pocket tertile bands
    def _bands(col_key, new_name):
        col = ses_map.get(col_key)
        if col is None or col not in df2.columns:
            return None
        
        idx = train_idx if train_idx is not None else df2.index
        # Convert to numeric using improved _num function
        s = df2.loc[idx, col].apply(_num).dropna()
        
        if len(s) >= 10:
            q1, q2 = s.quantile([0.34, 0.67])
            def get_band(v):
                val = _num(v)
                if val is None: return "Unknown"
                if val < q1: return "Low"
                if val < q2: return "Medium"
                return "High"
            
            df2[new_name] = df2[col].apply(get_band)
            return (q1, q2)
        else:
            df2[new_name] = "Unknown"
            return None

    inc_qs = _bands("average_income", "income_band")
    pocket_qs = _bands("pocket_money", "pocket_band")

    # SES categorical columns to use in the model
    ses_cat_cols = []
    # explicit list of keys that map to categorical columns
    cat_keys = ["school","grade","house_ownership","i_live_with",
                "father_s_education","mother_s_education",
                "father_s_job","mother_s_job",
                "insurance","access_to","frequency","affordability"]
                
    for key in cat_keys:
        col = ses_map.get(key)
        if col is not None and col in df2.columns:
            ses_cat_cols.append(col)
            
    # add derived bands
    if "income_band" in df2.columns: ses_cat_cols.append("income_band")
    if "pocket_band" in df2.columns: ses_cat_cols.append("pocket_band")

    # raw numeric SES to drop from num_cols (because we use bands or they are text)
    raw_numeric_drop = []
    for key in ["average_income", "pocket_money"]:
        col = ses_map.get(key)
        if col is not None and col in df2.columns:
            raw_numeric_drop.append(col)

    meta = {
        "ses_map": ses_map,
        "income_quantiles": inc_qs,
        "pocket_quantiles": pocket_qs,
    }
    return df2, ses_cat_cols, raw_numeric_drop, meta

# ------------------------- UI + explainability ---------------------------
def build_ses_ui(df: pd.DataFrame, ses_cat_cols):
    """Render SES select boxes; return (ses_vals: dict, used_cols: list)."""
    import streamlit as st
    ses_cols = [c for c in ses_cat_cols if c in df.columns]
    ses_vals = {}
    if not ses_cols:
        return ses_vals, []
    
    st.subheader("Socio-economic inputs")
    cols = st.columns(2)
    for i, c in enumerate(ses_cols):
        opts = sorted(df[c].astype(str).dropna().unique().tolist())
        # Ensure 'Unknown' is at the end or valid
        if "Unknown" in opts:
            opts.remove("Unknown")
            opts = opts + ["Unknown"]
        elif not opts:
            opts = ["Unknown"]
            
        default = df[c].mode(dropna=True).iloc[0] if not df[c].dropna().empty else opts[0]
        with cols[i % 2]:
            ses_vals[c] = st.selectbox(c, options=opts, index=(opts.index(default) if default in opts else 0))
    return ses_vals, ses_cols
