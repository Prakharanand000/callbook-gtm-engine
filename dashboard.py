"""
Callbook GTM Intelligence Engine  —  ML Edition
=================================================
streamlit run dashboard.py

ML layers:
  • TF-IDF Cosine Similarity  (sklearn)
  • K-Means Clustering k=4    (sklearn)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Callbook GTM Engine", layout="wide", page_icon="📞")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
    .stApp{background:#ffffff;font-family:'DM Sans',sans-serif;}
    section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f172a 0%,#1e293b 100%);}
    section[data-testid="stSidebar"] *{color:#cbd5e1 !important;}
    h1{font-family:'Space Grotesk',sans-serif !important;color:#0f172a !important;font-weight:700 !important;}
    h2,h3{font-family:'Space Grotesk',sans-serif !important;color:#1e293b !important;}
    [data-testid="stMetric"]{background:white;padding:18px 16px;border-radius:14px;
        box-shadow:0 2px 8px rgba(0,0,0,.06);border:1px solid #e2e8f0;
        transition:transform .2s,box-shadow .2s;}
    [data-testid="stMetric"]:hover{transform:translateY(-2px);box-shadow:0 4px 16px rgba(0,0,0,.1);}
    [data-testid="stMetricValue"]{font-family:'Space Grotesk',sans-serif !important;font-size:1.5rem !important;color:#0f172a !important;font-weight:700 !important;}
    [data-testid="stMetricLabel"]{font-size:.8rem !important;color:#64748b !important;text-transform:uppercase;letter-spacing:.5px;}
    .stDataFrame{border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06);}
    .stProgress>div>div{background:linear-gradient(90deg,#0d9488,#06b6d4) !important;border-radius:8px;}
    .stTextInput input{border-radius:10px !important;border:2px solid #e2e8f0 !important;padding:12px !important;}
    .stTextInput input:focus{border-color:#0d9488 !important;box-shadow:0 0 0 3px rgba(13,148,136,.15) !important;}
    hr{border:none;height:1px;background:linear-gradient(90deg,transparent,#cbd5e1,transparent);margin:24px 0;}
    .stDeployButton,#MainMenu,header[data-testid="stHeader"]{display:none !important;}
    header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
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

CLUSTER_COLORS = {
    "Fast-Moving Fintechs":          "#059669",
    "Enterprise Collections Scalers": "#2563EB",
    "Emerging Lenders":              "#F59E0B",
    "Peripheral Finance":            "#94A3B8",
}

DEMO_ROWS = [
    {"company_name":"OppFi","industry":"Fintech Lending","state":"IL","employee_count":420,
     "job_signal":"VP of Collections","days_since_posted":3,"website":"oppfi.com",
     "description":"Consumer lending platform underbanked personal loans debt recovery charge-off FDCPA"},
    {"company_name":"LoanCore Capital","industry":"Consumer Lending","state":"TX","employee_count":180,
     "job_signal":"Collections Manager","days_since_posted":5,"website":"loancore.com",
     "description":"Personal loan originator delinquency management charge-off right party contact"},
    {"company_name":"Upstart Network","industry":"Fintech Lending","state":"CA","employee_count":1200,
     "job_signal":"Head of Debt Recovery","days_since_posted":8,"website":"upstart.com",
     "description":"AI lending platform personal loans charge-off collections strategy FDCPA compliance"},
    {"company_name":"Avant Credit","industry":"Consumer Finance","state":"IL","employee_count":530,
     "job_signal":"Collections Analyst x3","days_since_posted":2,"website":"avant.com",
     "description":"Online personal loans middle-income borrowers accounts receivable recovery promise to pay"},
    {"company_name":"GreenSky","industry":"Home Improvement Lending","state":"GA","employee_count":800,
     "job_signal":"Loan Servicing Manager","days_since_posted":11,"website":"greensky.com",
     "description":"Point-of-sale financing loan servicing default management debt resolution"},
    {"company_name":"Springleaf Financial","industry":"Consumer Finance","state":"IN","employee_count":2100,
     "job_signal":"Director of Collections","days_since_posted":6,"website":"springleaf.com",
     "description":"Personal loans consumer lending collections FDCPA compliance borrower outreach"},
    {"company_name":"LendingPoint","industry":"Fintech Lending","state":"GA","employee_count":310,
     "job_signal":"Collections Team Lead","days_since_posted":4,"website":"lendingpoint.com",
     "description":"Near-prime personal lending right party contact promise to pay delinquency"},
    {"company_name":"Navient","industry":"Student Loan Servicing","state":"DE","employee_count":7200,
     "job_signal":"VP Borrower Engagement","days_since_posted":14,"website":"navient.com",
     "description":"Loan management servicing borrower outreach digital collections omnichannel call center"},
    {"company_name":"Marlin Business Services","industry":"Equipment Financing","state":"NJ","employee_count":340,
     "job_signal":"Collections Specialist x2","days_since_posted":7,"website":"marlinfinance.com",
     "description":"Equipment business financing collections specialist accounts receivable recovery"},
    {"company_name":"PeerStreet","industry":"Real Estate Lending","state":"CA","employee_count":120,
     "job_signal":"Loan Workout Officer","days_since_posted":9,"website":"peerstreet.com",
     "description":"Real estate debt investments loan workout recovery default management"},
    {"company_name":"SoFi Technologies","industry":"Fintech Lending","state":"CA","employee_count":3100,
     "job_signal":"Collections Operations Manager","days_since_posted":12,"website":"sofi.com",
     "description":"Online personal finance lending collections operations manager digital collections"},
    {"company_name":"Oportun Financial","industry":"Consumer Lending","state":"CA","employee_count":3400,
     "job_signal":"Director of Loan Servicing","days_since_posted":5,"website":"oportun.com",
     "description":"Affordable credit immigrants loan servicing collections director charge-off reduction"},
    {"company_name":"Prosper Marketplace","industry":"P2P Lending","state":"CA","employee_count":430,
     "job_signal":"Debt Recovery Specialist x4","days_since_posted":3,"website":"prosper.com",
     "description":"Peer to peer personal loans debt recovery delinquency charge-off accounts receivable"},
    {"company_name":"Figure Technologies","industry":"Fintech Lending","state":"CA","employee_count":560,
     "job_signal":"Head of Collections Strategy","days_since_posted":1,"website":"figure.com",
     "description":"Blockchain home equity loans collections strategy AI automation digital collections vendor"},
    {"company_name":"CAN Capital","industry":"Small Business Lending","state":"NY","employee_count":290,
     "job_signal":"Collections Manager","days_since_posted":8,"website":"cancapital.com",
     "description":"SMB working capital loans collections manager recovery delinquency promise to pay"},
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_int(v, d=0):
    try: return int(float(v))
    except: return d

def _seniority(signal):
    s = str(signal).lower()
    if any(k in s for k in ["vp","director","head of","chief"]): return 1.0
    if any(k in s for k in ["manager","lead","officer","supervisor"]): return 0.6
    if any(k in s for k in ["specialist","analyst","coordinator"]): return 0.3
    return 0.1

def _rule_score(row):
    s = 0
    sig = str(row.get("job_signal","")).lower()
    if any(k in sig for k in ["vp","director","head of","chief"]): s += 40
    elif any(k in sig for k in ["manager","lead","officer","supervisor"]): s += 30
    elif any(k in sig for k in ["specialist","analyst","coordinator"]): s += 20
    else: s += 10

    size = _safe_int(row.get("employee_count", 0))
    if   50 <= size <= 600:  s += 25
    elif 600 < size <= 2500: s += 15
    elif size > 2500:        s += 5
    else:                    s += 10

    combined = (str(row.get("industry","")) + " " + str(row.get("description",""))).lower()
    if any(x in combined for x in ["fintech","consumer lending","p2p","consumer finance","personal loan"]): s += 25
    elif any(x in combined for x in ["student loan","equipment","real estate","small business"]): s += 18
    elif any(x in combined for x in ["banking","financial services","credit union","insurance"]): s += 12
    else: s += 5

    days = _safe_int(row.get("days_since_posted", 99))
    if   days <= 5:  s += 10
    elif days <= 10: s += 6
    elif days <= 14: s += 3
    return min(s, 100)

def _outreach_angle(row):
    sig  = str(row.get("job_signal","")).lower()
    name = row.get("company_name","this company")
    size = _safe_int(row.get("employee_count", 0))
    cl   = str(row.get("cluster_label",""))
    sim  = float(row.get("tfidf_similarity", 0) or 0)
    if "vp" in sig or "director" in sig or "head" in sig:
        return f"Hiring senior collections leadership at {name} — pitch replacing that search with Callbook AI"
    elif "manager" in sig or "lead" in sig:
        return f"Scaling collections ops at {name} — pitch: Callbook automates what the new manager would own"
    elif size < 200:
        return f"Lean {size}-person team at {name} — pitch: Callbook does the work of 3–5 collectors"
    elif "Enterprise" in cl:
        return f"Enterprise scaling at {name} — pitch: compliance-safe AI that integrates with existing stack"
    elif sim > 0.3:
        return f"Strong ICP match at {name} (similarity {sim:.2f}) — lead with ROI: recover more, spend less"
    return f"Active collections growth at {name} — pitch: AI recovery at scale, compliant and fast"


# ── Core ML pipeline (one cached function, always produces all columns) ───────
@st.cache_data(show_spinner="Running ML pipeline…")
def build_leads():
    """
    Load scored_leads.csv if it already has ML columns;
    otherwise build from DEMO_ROWS and run full ML pipeline.
    Always returns a DataFrame with: rule_score, tfidf_similarity,
    tfidf_bonus, score, cluster_label, tier, outreach_angle.
    """
    csv_path = Path("data/scored_leads.csv")
    ml_cols  = {"tfidf_similarity", "cluster_label", "rule_score"}

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        missing = ml_cols - set(df.columns)
        if not missing:
            # CSV already has all ML columns — use it directly
            for c in ["score","rule_score","tfidf_similarity","tfidf_bonus",
                      "employee_count","days_since_posted"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df

    # ── Fall through: run full pipeline on demo data ──────────────────────
    df = pd.DataFrame(DEMO_ROWS)

    # 1. Rule score
    df["rule_score"] = df.apply(_rule_score, axis=1)

    # 2. TF-IDF cosine similarity
    texts = [
        " ".join([
            str(r.get("job_signal","")),
            str(r.get("industry","")),
            str(r.get("description","")),
            str(r.get("company_name","")),
        ])
        for _, r in df.iterrows()
    ]
    corpus = texts + [ICP_REFERENCE]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=500,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1]).flatten()

    df["tfidf_similarity"] = sims.round(4)
    df["tfidf_bonus"]      = (sims * 20).round(2)
    df["score"]            = (df["rule_score"] + df["tfidf_bonus"]).clip(upper=100).round(1)

    # 3. K-Means clustering
    feat = pd.DataFrame({
        "rule_score": df["rule_score"].fillna(50),
        "tfidf_sim":  df["tfidf_similarity"].fillna(0),
        "size_log":   np.log1p(df["employee_count"].fillna(100).clip(lower=1)),
        "recency":    df["days_since_posted"].fillna(14).clip(upper=30)
                        .apply(lambda d: max(0.0, 1.0 - d / 30.0)),
        "seniority":  df["job_signal"].apply(_seniority),
    })

    scaler = StandardScaler()
    X      = scaler.fit_transform(feat)
    km     = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    df["cluster_id"] = labels
    feat["cluster_id"] = labels
    stats = feat.groupby("cluster_id").mean()
    stats["rank"] = stats["tfidf_sim"] * 2 + stats["seniority"] - stats["size_log"] * 0.3
    ranked = stats["rank"].sort_values(ascending=False).index.tolist()

    label_map = {
        ranked[0]: "Fast-Moving Fintechs",
        ranked[1]: "Enterprise Collections Scalers",
        ranked[2]: "Emerging Lenders",
        ranked[3]: "Peripheral Finance",
    }
    df["cluster_label"] = df["cluster_id"].map(label_map)

    # 4. Tier + outreach angle (must come after cluster_label exists)
    df["tier"]           = df["score"].apply(lambda x: "A" if x >= 75 else ("B" if x >= 50 else "C"))
    df["outreach_angle"] = df.apply(_outreach_angle, axis=1)

    # Persist so next reload is instant
    Path("data").mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)

    return df


# ── Load ──────────────────────────────────────────────────────────────────────
leads  = build_leads()
tier_a = int((leads["tier"] == "A").sum())
tier_b = int((leads["tier"] == "B").sum())

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div style="text-align:center;padding-bottom:8px;">'
    '<p style="font-size:1.4rem;font-weight:700;margin:0;letter-spacing:3px;">CALLBOOK GTM</p>'
    '<p style="font-size:.7rem;margin:6px 0 0 0;opacity:.6;">ML Intelligence Engine</p>'
    '</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio("", [
    "Overview",
    "Lead Pipeline",
    "Company Lookup",
    "ML Insights — TF-IDF",
    "ML Insights — Clusters",
    "Outreach Templates",
    "Scoring Methodology",
    "Apify Pipeline Guide",
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(leads):,} prospects scored**")
st.sidebar.markdown(f"🟢 Tier A: {tier_a}  |  🔵 Tier B: {tier_b}")
st.sidebar.markdown("✅ ML layers active")
st.sidebar.markdown("""
<div style="text-align:center;padding-top:12px;border-top:1px solid rgba(255,255,255,0.1);">
    <p style="font-size:.65rem;opacity:.4;margin:8px 0 0 0;">Built by</p>
    <p style="font-size:.85rem;font-weight:600;margin:2px 0 0 0;">The Data Gamblers</p>
    <p style="font-size:.6rem;opacity:.35;margin:4px 0 0 0;">Callbook GTM Hackathon 2026</p>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<h1 style="text-align:center;font-size:3rem;letter-spacing:4px;margin-bottom:4px;">CALLBOOK GTM ENGINE</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;font-style:italic;color:#64748b;font-size:1.1rem;">"Callbook has PMF. The bottleneck is pipeline."</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#94a3b8;">Hiring signals → ML scoring → personalized outreach. End-to-end.</p>', unsafe_allow_html=True)
    st.markdown("---")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Prospects", f"{len(leads):,}")
    c2.metric("Tier A Leads",    str(tier_a), "Ready to contact")
    c3.metric("Tier B Leads",    str(tier_b), "Worth nurturing")
    c4.metric("Avg ML Score",    f"{leads['score'].mean():.0f} / 100")
    c5.metric("ML Layers",       "2 active", "TF-IDF + K-Means")

    st.markdown("---")
    st.subheader("How the ML pipeline works")
    st.markdown("""
**Step 1 — Apify scrapes LinkedIn Jobs** for hiring signals (Collections Manager, VP Collections, Head of Debt Recovery…)

**Step 2 — Rule-based score (80 pts max)** — seniority of hire, company size, industry fit, posting recency.

**Step 3 — TF-IDF Cosine Similarity bonus (+20 pts)** — scikit-learn TF-IDF on bigrams across job title + industry + company description, cosine distance to Callbook's ICP reference text.

**Step 4 — K-Means Clustering (k=4)** — groups leads using 5 features: rule score, TF-IDF similarity, company size (log), recency, title seniority.

**Final score = rule score + TF-IDF bonus (capped at 100)**
    """)

    col1, col2 = st.columns(2)
    with col1:
        tc = leads["tier"].value_counts().reindex(["A","B","C"]).fillna(0)
        fig = px.bar(x=["Tier A","Tier B","Tier C"], y=tc.values,
                     color=["Tier A","Tier B","Tier C"],
                     color_discrete_map={"Tier A":"#059669","Tier B":"#2563EB","Tier C":"#94A3B8"},
                     labels={"x":"","y":"Companies"}, title="Lead Pipeline by Tier")
        fig.update_layout(showlegend=False, height=320, margin=dict(t=40,b=10),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cl = leads["cluster_label"].value_counts()
        fig2 = px.pie(values=cl.values, names=cl.index, title="Leads by ML Cluster",
                      color=cl.index, color_discrete_map=CLUSTER_COLORS)
        fig2.update_layout(height=320, margin=dict(t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("ML score vs rule score — what TF-IDF changed")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=leads["rule_score"], y=leads["score"],
        mode="markers+text",
        text=leads["company_name"], textposition="top center", textfont=dict(size=9),
        marker=dict(size=12, color=leads["tfidf_similarity"], colorscale="Teal",
                    showscale=True, colorbar=dict(title="TF-IDF<br>Similarity"),
                    line=dict(width=1, color="white")),
        hovertemplate="<b>%{text}</b><br>Rule: %{x}<br>ML Score: %{y}<extra></extra>"
    ))
    fig3.add_trace(go.Scatter(x=[0,100], y=[0,100], mode="lines",
                               line=dict(dash="dash", color="#e2e8f0", width=1), showlegend=False))
    fig3.update_layout(xaxis_title="Rule-Based Score", yaxis_title="Final ML Score",
                       height=440, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Points **above the dashed line** received a TF-IDF boost. Darker teal = higher semantic similarity to Callbook's ICP description.")


# ══════════════════════════════════════════════════════════════════════════════
# LEAD PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Lead Pipeline":
    st.title("Lead Pipeline")
    st.info("All scored prospects. Heatmap = ML score. Filter by tier, industry, or cluster.")

    fc1,fc2,fc3 = st.columns(3)
    ft  = fc1.selectbox("Tier",     ["All","A","B","C"])
    fi  = fc2.selectbox("Industry", ["All"]+sorted(leads["industry"].dropna().unique().tolist()))
    fcl = fc3.selectbox("Cluster",  ["All"]+sorted(leads["cluster_label"].dropna().unique().tolist()))

    filt = leads.copy()
    if ft  != "All": filt = filt[filt["tier"]==ft]
    if fi  != "All": filt = filt[filt["industry"]==fi]
    if fcl != "All": filt = filt[filt["cluster_label"]==fcl]

    st.markdown(f"**{len(filt):,} companies match**")

    show = ["company_name","industry","state","employee_count","job_signal",
            "days_since_posted","rule_score","tfidf_similarity","score","tier","cluster_label","outreach_angle"]
    cols = [c for c in show if c in filt.columns]
    fmt  = {"score":"{:.1f}","rule_score":"{:.0f}","tfidf_similarity":"{:.3f}",
            "employee_count":"{:,.0f}","days_since_posted":"{:.0f}d ago"}

    styled = filt[cols].sort_values("score", ascending=False).reset_index(drop=True)
    st.dataframe(
        styled.style
              .format({k:v for k,v in fmt.items() if k in styled.columns})
              .background_gradient(subset=["score"],           cmap="RdYlGn", vmin=0,   vmax=100)
              .background_gradient(subset=["tfidf_similarity"],cmap="Blues",  vmin=0,   vmax=0.6),
        use_container_width=True, height=520
    )

    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(leads, x="score", nbins=20, color="tier",
                           color_discrete_map={"A":"#059669","B":"#2563EB","C":"#94A3B8"},
                           title="Final ML Score Distribution")
        fig.update_layout(height=300, margin=dict(t=40),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.scatter(leads, x="tfidf_similarity", y="rule_score",
                          color="cluster_label", color_discrete_map=CLUSTER_COLORS,
                          size="score", hover_name="company_name",
                          title="TF-IDF Similarity vs Rule Score by Cluster")
        fig2.update_layout(height=300, margin=dict(t=40),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# COMPANY LOOKUP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Company Lookup":
    st.title("Company Lookup")
    st.info("Full ML scorecard per company: rule score, TF-IDF similarity, bonus points, cluster, recommended action.")

    search = st.text_input("Search company name", placeholder="Try: OppFi, Avant, Figure, LendingPoint…")
    if search:
        matches = leads[leads["company_name"].str.lower().str.contains(search.lower(), na=False)]
        if len(matches) == 0:
            st.warning(f"No results for '{search}'.")
        else:
            for _, row in matches.iterrows():
                name    = row["company_name"]
                score   = float(row.get("score", 0) or 0)
                tier    = row.get("tier","C")
                cluster = row.get("cluster_label","—")
                sim     = float(row.get("tfidf_similarity", 0) or 0)
                rule_sc = float(row.get("rule_score", score) or score)
                bonus   = float(row.get("tfidf_bonus", 0) or 0)

                with st.expander(f"{name}  |  ML Score: {score:.1f}/100  |  Tier {tier}  |  {cluster}"):
                    m1,m2,m3,m4,m5 = st.columns(5)
                    m1.metric("ML Score",     f"{score:.1f}/100")
                    m2.metric("Rule Score",   f"{rule_sc:.0f}/80")
                    m3.metric("TF-IDF Bonus", f"+{bonus:.1f} pts")
                    m4.metric("Tier",         f"Tier {tier}")
                    m5.metric("Cluster",      cluster)

                    st.progress(min(sim / 0.6, 1.0),
                                text=f"ICP Cosine Similarity: {sim:.4f}  (0 = no match → 0.6+ = strong match)")

                    st.markdown(f"**Hiring signal:** {row.get('job_signal','N/A')} — posted {row.get('days_since_posted','?')}d ago")
                    st.markdown("**Outreach angle:**")
                    st.info(row.get("outreach_angle",""))

                    cl_col = CLUSTER_COLORS.get(cluster,"#94A3B8")
                    cl_desc = CLUSTER_DESCRIPTIONS.get(cluster,"")
                    if cl_desc:
                        st.markdown(
                            f'<div style="border-left:4px solid {cl_col};padding-left:12px;margin-top:8px;">'
                            f'<b>Cluster — {cluster}:</b> {cl_desc}</div>',
                            unsafe_allow_html=True)

                    st.markdown("---")
                    if tier == "A":
                        st.success("**CONTACT NOW** — LinkedIn DM → email → KugelAudio voice note within 24h")
                    elif tier == "B":
                        st.info("**NURTURE** — Add to email sequence, follow up in 7 days")
                    else:
                        st.warning("**DEPRIORITIZE** — Monitor for new hiring signals")


# ══════════════════════════════════════════════════════════════════════════════
# ML INSIGHTS — TF-IDF
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ML Insights — TF-IDF":
    st.title("ML Insights — TF-IDF Cosine Similarity")
    st.markdown("""
**What this is:** Each company's text (job title + industry + description) is treated as a document.
We measure how semantically close it is to Callbook's ICP description using
**TF-IDF vectorization** + **cosine similarity** from scikit-learn.

**Why it matters:** Two companies can share the same rule score but have very different text signals.
A company mentioning *"FDCPA compliance, charge-off, right party contact, promise to pay"*
is a much warmer lead than one that only says *"financial services"*.
TF-IDF catches that difference and adds up to **+20 bonus points**.
    """)

    st.markdown("---")
    st.subheader("The ICP reference document")
    st.markdown("*Every company's text is compared against this:*")
    st.code(ICP_REFERENCE.strip(), language=None)

    st.markdown("---")
    st.subheader("Similarity scores — all companies")

    sim_df = leads[["company_name","industry","job_signal",
                     "tfidf_similarity","tfidf_bonus","rule_score","score","tier"]
                   ].copy().sort_values("tfidf_similarity", ascending=False).reset_index(drop=True)

    fig = px.bar(
        sim_df, x="tfidf_similarity", y="company_name", orientation="h",
        color="tfidf_similarity", color_continuous_scale="Teal",
        labels={"tfidf_similarity":"Cosine Similarity","company_name":""},
        title="TF-IDF Cosine Similarity — All Leads (sorted high → low)"
    )
    fig.update_layout(
        height=max(380, len(sim_df)*32),
        margin=dict(t=40, l=180),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Longer bar = more semantically similar to Callbook's ICP vocabulary. The TF-IDF bonus (up to +20 pts) is directly proportional to this value.")

    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Rule score vs TF-IDF bonus")
        fig2 = px.scatter(
            sim_df, x="rule_score", y="tfidf_bonus",
            size="tfidf_similarity", color="tier",
            color_discrete_map={"A":"#059669","B":"#2563EB","C":"#94A3B8"},
            hover_name="company_name",
            labels={"rule_score":"Rule Score (0-80)","tfidf_bonus":"TF-IDF Bonus (0-20 pts)"},
        )
        fig2.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Size = TF-IDF similarity. High rule score + high bonus = Tier A.")

    with c2:
        st.subheader("Similarity vs final score")
        fig3 = px.scatter(
            sim_df, x="tfidf_similarity", y="score",
            color="tier", hover_name="company_name",
            color_discrete_map={"A":"#059669","B":"#2563EB","C":"#94A3B8"},
            labels={"tfidf_similarity":"Cosine Similarity","score":"Final ML Score"},
        )
        fig3.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Positive slope confirms: higher TF-IDF similarity → higher final ML score.")

    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Top 5 — highest ICP match")
        st.dataframe(
            sim_df.head(5)[["company_name","industry","tfidf_similarity","tfidf_bonus","score"]]
            .style.format({"tfidf_similarity":"{:.4f}","tfidf_bonus":"{:.1f}","score":"{:.1f}"}),
            use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Bottom 5 — lowest ICP match")
        st.dataframe(
            sim_df.tail(5)[["company_name","industry","tfidf_similarity","tfidf_bonus","score"]]
            .style.format({"tfidf_similarity":"{:.4f}","tfidf_bonus":"{:.1f}","score":"{:.1f}"}),
            use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("How TF-IDF works — step by step")
    st.markdown("""
1. **Build corpus** — each company's text + the ICP reference = one document each.
2. **Fit TF-IDF** — term frequency weighted by inverse document frequency. Bigrams, max 500 features, sublinear TF scaling.
3. **Vectorize** — every document becomes a sparse high-dimensional vector.
4. **Cosine similarity** — angle between each company vector and the ICP vector. 1.0 = identical, 0.0 = no overlap.
5. **Bonus** — `similarity × 20` = up to +20 points. Final score capped at 100.
    """)
    st.code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = company_texts + [ICP_REFERENCE]   # companies + ICP reference
vec    = TfidfVectorizer(stop_words='english', ngram_range=(1,2),
                         max_features=500, sublinear_tf=True)
mat    = vec.fit_transform(corpus)

# similarity of each company to the ICP reference (last document)
sims   = cosine_similarity(mat[:-1], mat[-1]).flatten()
bonus  = sims * 20   # → 0 to 20 bonus points per lead
    """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# ML INSIGHTS — CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ML Insights — Clusters":
    st.title("ML Insights — K-Means Clustering")
    st.markdown("""
**What this is:** K-Means (k=4) clusters all leads into behavioural archetypes using 5 features:
rule score, TF-IDF similarity, log(company size), hiring recency, and title seniority.
Each cluster is auto-labelled based on its centroid profile.

**Why it matters:** Two companies can share the same score but be completely different in nature.
K-Means finds the natural groupings the rule score misses — just like PACEA did with nonprofit segments.
    """)

    st.markdown("---")
    st.subheader("The 4 clusters")
    for label, desc in CLUSTER_DESCRIPTIONS.items():
        count = int((leads["cluster_label"] == label).sum())
        col   = CLUSTER_COLORS[label]
        avg   = leads[leads["cluster_label"]==label]["score"].mean()
        st.markdown(
            f'<div style="border-left:4px solid {col};padding:12px 16px;border-radius:0 8px 8px 0;'
            f'background:#f8fafc;margin-bottom:8px;">'
            f'<b style="color:{col};font-size:1rem;">{label}</b>'
            f' <span style="color:#94a3b8;font-size:.85rem;">— {count} leads, avg score {avg:.1f}</span><br>'
            f'<span style="color:#475569;">{desc}</span></div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("PCA scatter — 5D feature space projected to 2D")

    feat = pd.DataFrame({
        "rule_score": leads["rule_score"].fillna(50),
        "tfidf_sim":  leads["tfidf_similarity"].fillna(0),
        "size_log":   np.log1p(leads["employee_count"].fillna(100).clip(lower=1)),
        "recency":    leads["days_since_posted"].fillna(14).clip(upper=30)
                        .apply(lambda d: max(0.0, 1.0 - d/30.0)),
        "seniority":  leads["job_signal"].apply(_seniority),
    })
    X_scaled = StandardScaler().fit_transform(feat)
    pca      = PCA(n_components=2, random_state=42)
    coords   = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        "PC1":     coords[:,0],
        "PC2":     coords[:,1],
        "company": leads["company_name"],
        "cluster": leads["cluster_label"],
        "score":   leads["score"],
    })
    ev = pca.explained_variance_ratio_
    fig = px.scatter(
        plot_df, x="PC1", y="PC2",
        color="cluster", size="score",
        hover_name="company",
        color_discrete_map=CLUSTER_COLORS,
        text="company",
        labels={"PC1": f"PC1 ({ev[0]*100:.0f}% variance explained)",
                "PC2": f"PC2 ({ev[1]*100:.0f}% variance explained)"},
        title=f"K-Means Clusters — PCA Projection (total variance explained: {ev[:2].sum()*100:.0f}%)"
    )
    fig.update_traces(textposition="top center", textfont_size=8)
    fig.update_layout(height=500, plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each dot = one company. Size = ML score. Well-separated clusters confirm K-Means found meaningful natural groupings — not random noise.")

    st.markdown("---")
    st.subheader("Cluster feature profiles — radar chart")

    feat["cluster"] = leads["cluster_label"].values
    cluster_means   = feat.groupby("cluster").mean()
    radar_df        = cluster_means.copy()
    for col in radar_df.columns:
        r = radar_df[col].max() - radar_df[col].min()
        if r > 0:
            radar_df[col] = (radar_df[col] - radar_df[col].min()) / r

    cats = ["Rule Score","TF-IDF Sim","Company Size","Recency","Seniority"]
    fig2 = go.Figure()
    for cl_name in radar_df.index:
        vals = radar_df.loc[cl_name].tolist() + [radar_df.loc[cl_name].tolist()[0]]
        fig2.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill="toself", name=cl_name,
            line_color=CLUSTER_COLORS.get(cl_name,"#999"),
        ))
    fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                       height=440, margin=dict(t=40),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Larger area = higher values across all 5 features. Green (Fast-Moving Fintechs) should dominate on TF-IDF and Seniority. Blue (Enterprise) on Company Size.")

    st.markdown("---")
    st.subheader("Cluster composition table")
    cl_table = leads.groupby("cluster_label").agg(
        Count      =("company_name",      "count"),
        Avg_Score  =("score",             "mean"),
        Avg_TF_IDF =("tfidf_similarity",  "mean"),
        Tier_A     =("tier", lambda x: int((x=="A").sum())),
        Tier_B     =("tier", lambda x: int((x=="B").sum())),
    ).round(3)
    st.dataframe(
        cl_table.style.format({"Avg_Score":"{:.1f}","Avg_TF_IDF":"{:.3f}"}),
        use_container_width=True)

    st.markdown("---")
    st.subheader("How K-Means works here")
    st.markdown("""
1. **Feature matrix** — 5 columns: rule_score, tfidf_similarity, log(employee_count), recency (0–1), seniority (0–1).
2. **StandardScaler** — zero mean, unit variance so no single feature dominates.
3. **KMeans(k=4, n_init=10)** — 10 random initializations to avoid bad local minima, picks the best.
4. **Auto-labelling** — clusters ranked by `tfidf_sim×2 + seniority − size_log×0.3`. Highest = "Fast-Moving Fintechs".
5. **PCA visualization** — 5D → 2D for the scatter above.
    """)
    st.code("""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features = df[['rule_score','tfidf_similarity','size_log','recency','seniority']]
X = StandardScaler().fit_transform(features)

km     = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X)
df['cluster'] = labels
    """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# OUTREACH TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Outreach Templates":
    st.title("Outreach Templates")
    st.info("3 templates keyed to hiring signal. Cluster-aware framing — Fast-Moving Fintechs get urgency, Enterprise gets integration/compliance.")

    tier_a_leads = leads[leads["tier"]=="A"].sort_values("score", ascending=False)
    options = tier_a_leads["company_name"].tolist() if len(tier_a_leads)>0 else leads["company_name"].tolist()
    selected = st.selectbox("Preview for:", options)
    row = leads[leads["company_name"]==selected].iloc[0]

    company  = selected
    signal   = row.get("job_signal","Collections Manager")
    industry = row.get("industry","fintech lending")
    size     = _safe_int(row.get("employee_count",300))
    cluster  = row.get("cluster_label","")
    sim      = float(row.get("tfidf_similarity",0) or 0)

    if cluster:
        cl_col = CLUSTER_COLORS.get(cluster,"#94A3B8")
        st.markdown(
            f'<div style="border-left:4px solid {cl_col};padding:8px 14px;border-radius:0 8px 8px 0;background:#f8fafc;margin-bottom:12px;">'
            f'<b style="color:{cl_col};">ML Cluster: {cluster}</b> — TF-IDF similarity: {sim:.3f}</div>',
            unsafe_allow_html=True)

    st.markdown("---")

    if "Enterprise" in cluster:
        cta = "integrates with your existing stack and handles FDCPA/TCPA compliance out of the box"
    elif sim > 0.35:
        cta = f"your team's focus on {industry} debt recovery is exactly what Callbook was built for"
    else:
        cta = f"already trusted by lenders in {industry} at your team size"

    st.subheader("Template A — VP/Director hiring signal")
    st.text_area("Subject:", f"Re: {signal} at {company} — worth 10 min", height=50, key="sa")
    st.text_area("Body:", f"""Hi [First Name],

Saw {company} is hiring a {signal}. That role is expensive to fill and takes 3–4 months to ramp.

Callbook AI gives your team the capability you're trying to hire for — without the headcount. It's an AI platform that helps lenders understand borrowers, decide the right next action, and recover more. {cta.capitalize()}.

Bootstrapped to $1M ARR in 8 months. Happy to show you 10 minutes of the product this week.

Worth a look?
[Your name]""", height=230, key="ba")

    st.markdown("---")
    st.subheader("Template B — Scaling ops angle")
    st.text_area("Subject:", f"{company} collections team — one thing worth knowing", height=50, key="sb")
    st.text_area("Body:", f"""Hi [First Name],

Noticed {company} is building out collections capacity (saw the {signal} posting).

Callbook is an AI platform built specifically for lenders to collect more by actually understanding their borrowers — not just automating old call scripts. Works well for {industry} teams at {size} employees.

Happy to send a 2-min video walkthrough. What's easier for you?

[Your name]""", height=210, key="bb")

    st.markdown("---")
    st.subheader("Template C — LinkedIn DM")
    st.text_area("DM:", f"Hi [First Name] — saw {company} is scaling collections ops. Building something relevant for {industry} teams your size. Mind if I share a quick breakdown? 2 min.", height=100, key="dc")

    st.markdown("---")
    st.markdown("""
    <div style="background:linear-gradient(135deg,#fef3c7,#fffbeb);padding:20px;border-radius:14px;border-left:4px solid #f59e0b;">
    <h3 style="margin-top:0;">🎙️ KugelAudio Integration (Sponsor)</h3>
    <p>Add a 30-second personalized voice note. 3× reply rate vs cold email alone.<br><br>
    <b>Script:</b> "Hi [Name], saw [Company] is hiring for [Signal] — we built an AI platform that does exactly what that hire would do, faster and cheaper. Worth 10 minutes?"</p>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Scoring Methodology":
    st.title("Scoring Methodology")
    st.dataframe(pd.DataFrame({
        "Layer":    ["Rule-based score","TF-IDF cosine bonus","K-Means cluster label"],
        "Max Pts":  ["80","20","N/A"],
        "Method":   ["Hand-coded weights","sklearn TF-IDF + cosine_similarity","sklearn KMeans(k=4)"],
        "Purpose":  ["Fast explainable baseline","Semantic ICP fit from text","Behavioural grouping"],
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Rule-based weights")
    st.dataframe(pd.DataFrame({
        "Component":  ["Job Signal","Company Size","Industry Fit","Recency"],
        "Max Points": [40,25,25,10],
        "Tier A":     ["VP/Director/Head","50–600 employees","Fintech/Consumer Lending","≤5 days"],
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    c1.metric("Tier A","≥75 pts");  c1.markdown("Contact within 24h")
    c2.metric("Tier B","50–74 pts");c2.markdown("Nurture sequence")
    c3.metric("Tier C","<50 pts");  c3.markdown("Monitor, deprioritize")


# ══════════════════════════════════════════════════════════════════════════════
# APIFY PIPELINE GUIDE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Apify Pipeline Guide":
    st.title("Apify Pipeline — Run It Yourself")
    st.subheader("Step 1 — Setup")
    st.markdown("1. Go to [console.apify.com](https://console.apify.com)\n2. Apply hackathon promo code ($55 credits)")

    st.subheader("Step 2 — LinkedIn Jobs Scraper")
    st.code('{\n  "queries":["Collections Manager","VP of Collections","Head of Debt Recovery",\n             "Director Collections","Loan Servicing Manager"],\n  "locationName":"United States",\n  "datePosted":"past-month",\n  "limit":100\n}\n# Actor: bebity/linkedin-jobs-scraper → save as data/raw/jobs.json', language="json")

    st.subheader("Step 3 — Company Enrichment")
    st.code('# Actor: harvestapi/linkedin-company\n# Input: company names from jobs.json\n# Save as data/raw/companies.json', language="bash")

    st.subheader("Step 4 — Run ML pipeline")
    st.code("pip install -r requirements.txt\npython score_leads.py\nstreamlit run dashboard.py", language="bash")

    st.subheader("Cost estimate")
    st.dataframe(pd.DataFrame({
        "Actor":   ["LinkedIn Jobs","LinkedIn Company Details"],
        "Credits": ["~$3–5","~$4–6"],
        "Output":  ["100–200 job records","100–200 company profiles"],
    }), use_container_width=True, hide_index=True)
    st.success("~$10–12 per full run. $55 credits = 4–5 full pipeline refreshes.")
