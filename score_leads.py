"""
score_leads.py  —  Callbook GTM Intelligence Engine
=====================================================
ML-enhanced scoring pipeline.

Two ML layers on top of the rule-based score:
  1. TF-IDF + Cosine Similarity  — measures how closely a company's
     text (job title + industry + description) matches Callbook's
     ideal customer profile. Adds up to 20 bonus points.

  2. K-Means Clustering (k=4)    — groups leads into behavioural
     archetypes based on [rule_score, tfidf_similarity,
     employee_count_log, recency_score, seniority_score].
     Each cluster gets a human-readable label.

Final score  =  rule_score  +  tfidf_bonus  (capped at 100)
Tier cutoffs: A >= 75 | B >= 50 | C < 50

Usage:
    python score_leads.py

Outputs:
    data/scored_leads.csv
    data/ml_artifacts.json   (cluster centres + feature names for dashboard)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ── ICP reference document ──────────────────────────────────────────────────
# This is the "golden description" of Callbook's ideal customer.
# TF-IDF cosine similarity is measured against this text.
ICP_REFERENCE = """
debt collection collections manager loan recovery borrower outreach
delinquency management charge-off reduction debt resolution
right party contact promise to pay call center collections agent
accounts receivable recovery FDCPA compliance collections strategy
consumer lending fintech personal loan auto loan student loan
loan servicing credit portfolio default management
AI collections automation digital collections omnichannel outreach
collections technology vendor evaluation lender software
"""


# ── Rule-based score (unchanged from v1) ────────────────────────────────────

def rule_score(row):
    score = 0
    signal = str(row.get("job_signal", "")).lower()
    if any(k in signal for k in ["vp", "director", "head of", "chief"]):
        score += 40
    elif any(k in signal for k in ["manager", "lead", "officer", "supervisor"]):
        score += 30
    elif any(k in signal for k in ["specialist", "analyst", "coordinator"]):
        score += 20
    else:
        score += 10

    size = _safe_int(row.get("employee_count", 0))
    if 50 <= size <= 600:       score += 25
    elif 600 < size <= 2500:    score += 15
    elif size > 2500:           score += 5
    else:                       score += 10

    combined = (str(row.get("industry", "")) + " " + str(row.get("description", ""))).lower()
    if any(x in combined for x in ["fintech", "consumer lending", "p2p", "consumer finance", "personal loan"]):
        score += 25
    elif any(x in combined for x in ["student loan", "equipment", "real estate", "small business lending", "auto loan"]):
        score += 18
    elif any(x in combined for x in ["banking", "financial services", "credit union", "insurance"]):
        score += 12
    else:
        score += 5

    days = _safe_int(row.get("days_since_posted", 99))
    if days <= 5:    score += 10
    elif days <= 10: score += 6
    elif days <= 14: score += 3

    return min(score, 100)


# ── ML Layer 1: TF-IDF Cosine Similarity ────────────────────────────────────

def compute_tfidf_similarity(df):
    """
    Builds a TF-IDF corpus from each lead's combined text and measures
    cosine similarity against the ICP reference document.
    Returns a Series of similarity scores (0.0 – 1.0).
    """
    texts = []
    for _, row in df.iterrows():
        combined = " ".join([
            str(row.get("job_signal", "")),
            str(row.get("industry", "")),
            str(row.get("description", "")),
            str(row.get("company_name", "")),
        ])
        texts.append(combined)

    corpus = texts + [ICP_REFERENCE]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=500,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Similarity between each lead and the ICP reference (last row)
    lead_vectors = tfidf_matrix[:-1]
    icp_vector   = tfidf_matrix[-1]
    similarities = cosine_similarity(lead_vectors, icp_vector).flatten()

    return similarities, vectorizer


def tfidf_bonus(similarity_score):
    """Convert 0-1 cosine similarity → 0-20 bonus points."""
    return round(float(similarity_score) * 20, 2)


# ── ML Layer 2: K-Means Clustering ──────────────────────────────────────────

CLUSTER_LABELS = {
    # assigned after fitting — see assign_cluster_labels()
}

CLUSTER_DESCRIPTIONS = {
    "Fast-Moving Fintechs":
        "High-fit, small-to-mid fintechs with urgent hiring. Best Callbook ICP — contact immediately.",
    "Enterprise Collections Scalers":
        "Large lenders actively scaling collections ops. High volume potential, longer sales cycle.",
    "Emerging Lenders":
        "Smaller companies with moderate fit. Good future pipeline — nurture with content.",
    "Peripheral Finance":
        "Adjacent industries (insurance, equipment, real estate). Lower fit, lower urgency.",
}

def build_cluster_features(df):
    """Build the 5-feature matrix for K-Means."""
    features = pd.DataFrame()

    features["rule_score"]      = df["rule_score"].fillna(50)
    features["tfidf_sim"]       = df["tfidf_similarity"].fillna(0)
    features["size_log"]        = np.log1p(df["employee_count"].fillna(100).clip(lower=1))
    features["recency"]         = df["days_since_posted"].fillna(14).clip(upper=30).apply(
                                    lambda d: max(0, 1 - d / 30))
    features["seniority"]       = df["job_signal"].apply(_seniority_score)

    return features


def _seniority_score(signal):
    s = str(signal).lower()
    if any(k in s for k in ["vp", "director", "head of", "chief"]):  return 1.0
    if any(k in s for k in ["manager", "lead", "officer"]):          return 0.6
    if any(k in s for k in ["specialist", "analyst"]):               return 0.3
    return 0.1


def fit_kmeans(feature_df, k=4, random_state=42):
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return labels, km, scaler


def assign_cluster_labels(df, labels, feature_df):
    """
    After fitting, name each cluster based on its centroid profile.
    Logic:
      - Highest avg tfidf_sim + seniority + small size  → Fast-Moving Fintechs
      - High rule_score + large size                    → Enterprise Scalers
      - Moderate scores                                 → Emerging Lenders
      - Lowest fit                                      → Peripheral Finance
    """
    df = df.copy()
    df["cluster_id"] = labels

    cluster_means = feature_df.copy()
    cluster_means["cluster_id"] = labels
    stats = cluster_means.groupby("cluster_id").mean()

    # Rank clusters on composite: tfidf_sim*2 + seniority - size_log*0.3
    stats["rank_score"] = stats["tfidf_sim"] * 2 + stats["seniority"] - stats["size_log"] * 0.3

    ranked = stats["rank_score"].sort_values(ascending=False).index.tolist()

    label_map = {
        ranked[0]: "Fast-Moving Fintechs",
        ranked[1]: "Enterprise Collections Scalers",
        ranked[2]: "Emerging Lenders",
        ranked[3]: "Peripheral Finance",
    }

    df["cluster_label"] = df["cluster_id"].map(label_map)
    return df, label_map, stats


# ── Tier + Outreach ──────────────────────────────────────────────────────────

def get_tier(score):
    if score >= 75: return "A"
    if score >= 50: return "B"
    return "C"


def get_outreach_angle(row):
    signal = str(row.get("job_signal", "")).lower()
    name   = row.get("company_name", "this company")
    size   = _safe_int(row.get("employee_count", 0))
    sim    = row.get("tfidf_similarity", 0)
    cluster = str(row.get("cluster_label", ""))

    if "vp" in signal or "director" in signal or "head" in signal or "chief" in signal:
        return f"Hiring senior collections leadership at {name} — pitch replacing that search with Callbook AI"
    elif "manager" in signal or "lead" in signal:
        return f"Scaling collections ops at {name} — pitch: Callbook automates what the new manager would own"
    elif size < 200:
        return f"Lean {size}-person team at {name} — pitch: Callbook does the work of 3–5 collectors"
    elif "Enterprise" in cluster:
        return f"Enterprise scaling at {name} — pitch: compliance-safe AI that integrates with existing stack"
    elif sim > 0.3:
        return f"Strong ICP match at {name} (similarity {sim:.2f}) — lead with ROI: recover more, spend less"
    else:
        return f"Active collections growth at {name} — pitch: AI recovery at scale, compliant and fast"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_int(val, default=0):
    try:
        return int(val)
    except:
        return default


def generate_sample_leads():
    """15-row demo dataset — used when no Apify data is present."""
    rows = [
        {"company_name":"OppFi","industry":"Fintech Lending","state":"IL","employee_count":420,
         "job_signal":"VP of Collections","days_since_posted":3,"website":"oppfi.com",
         "description":"Consumer lending platform for underbanked borrowers personal loans"},
        {"company_name":"LoanCore Capital","industry":"Consumer Lending","state":"TX","employee_count":180,
         "job_signal":"Collections Manager","days_since_posted":5,"website":"loancore.com",
         "description":"Personal loan originator debt recovery delinquency management"},
        {"company_name":"Upstart Network","industry":"Fintech Lending","state":"CA","employee_count":1200,
         "job_signal":"Head of Debt Recovery","days_since_posted":8,"website":"upstart.com",
         "description":"AI lending platform personal loans charge-off collections strategy"},
        {"company_name":"Avant Credit","industry":"Consumer Finance","state":"IL","employee_count":530,
         "job_signal":"Collections Analyst x3","days_since_posted":2,"website":"avant.com",
         "description":"Online personal loans middle-income borrowers accounts receivable"},
        {"company_name":"GreenSky","industry":"Home Improvement Lending","state":"GA","employee_count":800,
         "job_signal":"Loan Servicing Manager","days_since_posted":11,"website":"greensky.com",
         "description":"Point-of-sale financing loan servicing default management"},
        {"company_name":"Springleaf Financial","industry":"Consumer Finance","state":"IN","employee_count":2100,
         "job_signal":"Director of Collections","days_since_posted":6,"website":"springleaf.com",
         "description":"Personal loans consumer lending collections FDCPA compliance"},
        {"company_name":"LendingPoint","industry":"Fintech Lending","state":"GA","employee_count":310,
         "job_signal":"Collections Team Lead","days_since_posted":4,"website":"lendingpoint.com",
         "description":"Near-prime personal lending right party contact promise to pay"},
        {"company_name":"Navient","industry":"Student Loan Servicing","state":"DE","employee_count":7200,
         "job_signal":"VP Borrower Engagement","days_since_posted":14,"website":"navient.com",
         "description":"Loan management servicing borrower outreach digital collections"},
        {"company_name":"Marlin Business Services","industry":"Equipment Financing","state":"NJ","employee_count":340,
         "job_signal":"Collections Specialist x2","days_since_posted":7,"website":"marlinfinance.com",
         "description":"Equipment business financing collections specialist accounts receivable"},
        {"company_name":"PeerStreet","industry":"Real Estate Lending","state":"CA","employee_count":120,
         "job_signal":"Loan Workout Officer","days_since_posted":9,"website":"peerstreet.com",
         "description":"Real estate debt investments loan workout recovery"},
        {"company_name":"SoFi Technologies","industry":"Fintech Lending","state":"CA","employee_count":3100,
         "job_signal":"Collections Operations Manager","days_since_posted":12,"website":"sofi.com",
         "description":"Online personal finance lending collections operations manager"},
        {"company_name":"Oportun Financial","industry":"Consumer Lending","state":"CA","employee_count":3400,
         "job_signal":"Director of Loan Servicing","days_since_posted":5,"website":"oportun.com",
         "description":"Affordable credit immigrants loan servicing collections director"},
        {"company_name":"Prosper Marketplace","industry":"P2P Lending","state":"CA","employee_count":430,
         "job_signal":"Debt Recovery Specialist x4","days_since_posted":3,"website":"prosper.com",
         "description":"Peer to peer personal loans debt recovery delinquency charge-off"},
        {"company_name":"Figure Technologies","industry":"Fintech Lending","state":"CA","employee_count":560,
         "job_signal":"Head of Collections Strategy","days_since_posted":1,"website":"figure.com",
         "description":"Blockchain home equity loans collections strategy AI automation"},
        {"company_name":"CAN Capital","industry":"Small Business Lending","state":"NY","employee_count":290,
         "job_signal":"Collections Manager","days_since_posted":8,"website":"cancapital.com",
         "description":"SMB working capital loans collections manager recovery"},
    ]
    return pd.DataFrame(rows)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(df):
    print(f"Running ML pipeline on {len(df)} leads...")

    # Step 1: Rule-based score
    df["rule_score"] = df.apply(rule_score, axis=1)
    print("  [1/4] Rule scores computed")

    # Step 2: TF-IDF similarity
    similarities, vectorizer = compute_tfidf_similarity(df)
    df["tfidf_similarity"] = similarities
    df["tfidf_bonus"]       = df["tfidf_similarity"].apply(tfidf_bonus)
    df["score"]             = (df["rule_score"] + df["tfidf_bonus"]).clip(upper=100).round(1)
    print(f"  [2/4] TF-IDF similarity computed (avg: {similarities.mean():.3f})")

    # Step 3: K-Means clustering
    feat_df  = build_cluster_features(df)
    labels, km, scaler = fit_kmeans(feat_df, k=4)
    df, label_map, cluster_stats = assign_cluster_labels(df, labels, feat_df)
    print(f"  [3/4] K-Means clusters assigned: {dict(df['cluster_label'].value_counts())}")

    # Step 4: Tier + outreach angle
    df["tier"]           = df["score"].apply(get_tier)
    df["outreach_angle"] = df.apply(get_outreach_angle, axis=1)
    print("  [4/4] Tiers and outreach angles generated")

    # Save ML artifacts for dashboard visualisation
    artifacts = {
        "cluster_label_map":  {int(k): v for k, v in label_map.items()},
        "cluster_descriptions": CLUSTER_DESCRIPTIONS,
        "cluster_stats": cluster_stats.round(4).to_dict(),
        "feature_names": list(feat_df.columns),
        "icp_reference": ICP_REFERENCE.strip(),
        "tfidf_vocab_size": len(vectorizer.vocabulary_),
        "n_leads": len(df),
        "tier_counts": dict(df["tier"].value_counts()),
        "cluster_counts": dict(df["cluster_label"].value_counts()),
    }
    Path("data").mkdir(exist_ok=True)
    with open("data/ml_artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=2)

    return df


def main():
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    jobs_path = Path("data/raw/jobs.json")
    if jobs_path.exists():
        with open(jobs_path) as f:
            df = pd.DataFrame(json.load(f))
        print("Loaded data/raw/jobs.json")

        # Basic normalisation
        rename = {"title": "job_signal", "companyName": "company_name",
                  "location": "state", "employeeCount": "employee_count"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        companies_path = Path("data/raw/companies.json")
        if companies_path.exists():
            with open(companies_path) as f:
                co = pd.DataFrame(json.load(f))
            co = co.rename(columns={"name": "company_name", "employeeCount": "employee_count"})
            if "company_name" in co.columns:
                df = df.merge(co, on="company_name", how="left")

        if "days_since_posted" not in df.columns:
            df["days_since_posted"] = 7
    else:
        print("No raw data found — running on demo dataset.")
        df = generate_sample_leads()

    df = run_pipeline(df)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    out = Path("data/scored_leads.csv")
    df.to_csv(out, index=False)

    print(f"\nSaved {len(df)} leads to {out}")
    print(f"  Tier A: {(df.tier=='A').sum()}  Tier B: {(df.tier=='B').sum()}  Tier C: {(df.tier=='C').sum()}")
    print("Run `streamlit run dashboard.py` to view results.")


if __name__ == "__main__":
    main()
