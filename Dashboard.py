# ruff: noqa
"""
================================================================
  CAC 2026 — DIGITAL LENDING STRATEGIC INTELLIGENCE DASHBOARD
  ─────────────────────────────────────────────────────────────
  Interactive Streamlit dashboard answering:
    Q1. Customer Segment Risk & Repayment Behaviors
    Q2. Acquisition Channels & Onboarding Impact
    Q3. Loan Products, Ticket Sizes & Tenures
    Q4. Pricing, Approval & Tenure Strategy
    Q5. Senior Leadership Performance Monitoring
================================================================
"""

import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os

# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="CAC 2026 · Digital Lending Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN TOKENS ───────────────────────────────────────────
BG       = "#0d0f18"
CARD     = "#161926"
BORDER   = "#252840"
TEXT     = "#e8eaf0"
SUB      = "#7c82a0"
GREEN    = "#00c896"
AMBER    = "#f0a500"
RED      = "#ff4560"
BLUE     = "#3b82f6"
PURPLE   = "#8b5cf6"
TEAL     = "#06b6d4"
PINK     = "#ec4899"
LIME     = "#84cc16"
ORANGE   = "#f97316"
INDIGO   = "#6366f1"
SKY      = "#0ea5e9"

RISK_COLORS = {
    "Prime":      GREEN,
    "Near-Prime": BLUE,
    "Subprime":   AMBER,
    "Thin-File":  RED,
}
TIER_ORDER  = ["Prime", "Near-Prime", "Subprime", "Thin-File"]
GRADE_ORDER = ["A+", "A", "A-", "B", "B-", "C+", "C", "D"]
CSEQ        = [BLUE, GREEN, AMBER, PURPLE, RED, TEAL, PINK, LIME, ORANGE, INDIGO, SKY]

CH_COLORS = {
    "Organic-App":      GREEN,
    "Referral":         LIME,
    "Bank-Partnership": BLUE,
    "Corporate-Tie-Up": TEAL,
    "NBFC-Embedded":    PURPLE,
    "DSA-Agent":        AMBER,
    "Paid-Digital":     RED,
}

PLOTLY_TEMPLATE = dict(
    paper_bgcolor=BG, plot_bgcolor=CARD,
    font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif"),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=SUB)),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=SUB)),
    legend=dict(bgcolor=CARD, bordercolor=BORDER),
    margin=dict(l=40, r=20, t=50, b=40),
)

def apply_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT)),
        height=height,
        **PLOTLY_TEMPLATE,
    )
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False)
    return fig

# ─── CUSTOM CSS ──────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Inter', 'Segoe UI', sans-serif;
      background-color: {BG};
      color: {TEXT};
  }}

  /* Main background */
  .stApp {{ background-color: {BG}; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
      background-color: {CARD};
      border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] .sidebar-content {{ background-color: {CARD}; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
      background-color: {CARD};
      border-radius: 10px;
      padding: 4px;
      gap: 4px;
      border: 1px solid {BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
      background-color: transparent;
      color: {SUB};
      border-radius: 8px;
      padding: 8px 16px;
      font-weight: 500;
      font-size: 13px;
      border: none;
      transition: all 0.2s;
  }}
  .stTabs [aria-selected="true"] {{
      background-color: {BLUE} !important;
      color: white !important;
  }}
  .stTabs [data-baseweb="tab"]:hover {{ color: {TEXT} !important; }}

  /* Metric cards */
  [data-testid="metric-container"] {{
      background-color: {CARD};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 16px !important;
      transition: transform 0.2s, box-shadow 0.2s;
  }}
  [data-testid="metric-container"]:hover {{
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  }}
  [data-testid="stMetricLabel"] {{ color: {SUB} !important; font-size: 12px !important; font-weight: 500; }}
  [data-testid="stMetricValue"] {{ color: {TEXT} !important; font-weight: 700; font-size: 24px !important; }}
  [data-testid="stMetricDelta"] {{ font-size: 11px !important; }}

  /* Containers / cards */
  .dash-card {{
      background-color: {CARD};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 16px;
  }}
  .section-header {{
      color: {TEXT};
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 4px;
      padding-bottom: 8px;
      border-bottom: 2px solid {BLUE};
      display: inline-block;
  }}
  .insight-box {{
      background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(139,92,246,0.08));
      border: 1px solid rgba(59,130,246,0.3);
      border-radius: 10px;
      padding: 14px 18px;
      margin: 10px 0;
      font-size: 13px;
      line-height: 1.6;
      color: {TEXT};
  }}
  .insight-box b {{ color: {BLUE}; }}
  .warn-box {{
      background: linear-gradient(135deg, rgba(240,165,0,0.12), rgba(249,115,22,0.08));
      border: 1px solid rgba(240,165,0,0.3);
      border-radius: 10px;
      padding: 14px 18px;
      margin: 10px 0;
      font-size: 13px;
      line-height: 1.6;
      color: {TEXT};
  }}
  .warn-box b {{ color: {AMBER}; }}
  .risk-chip {{
      display: inline-block;
      padding: 2px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      margin: 2px;
  }}
  .chip-green  {{ background: rgba(0,200,150,0.15); color: {GREEN}; border: 1px solid {GREEN}; }}
  .chip-blue   {{ background: rgba(59,130,246,0.15); color: {BLUE};  border: 1px solid {BLUE}; }}
  .chip-amber  {{ background: rgba(240,165,0,0.15);  color: {AMBER}; border: 1px solid {AMBER}; }}
  .chip-red    {{ background: rgba(255,69,96,0.15);  color: {RED};   border: 1px solid {RED}; }}

  /* Streamlit elements */
  .stSelectbox > div > div, .stMultiSelect > div > div {{
      background-color: {CARD};
      border: 1px solid {BORDER};
      color: {TEXT};
  }}
  hr {{ border-color: {BORDER}; }}
  .stMarkdown p {{ color: {TEXT}; line-height: 1.7; }}

  /* Sidebar logo area */
  .logo-header {{
      text-align: center;
      padding: 16px 0 20px;
  }}
  .logo-header h2 {{
      background: linear-gradient(90deg, {BLUE}, {PURPLE});
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-size: 22px;
      font-weight: 700;
      margin: 0;
  }}
  .logo-header p {{
      color: {SUB};
      font-size: 11px;
      margin: 4px 0 0;
  }}
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING ────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data(show_spinner="Loading datasets…")
def load_data():
    cust  = pd.read_csv(os.path.join(BASE, "dataset/01_customers.csv"))
    loans = pd.read_csv(os.path.join(BASE, "dataset/02_loans.csv"))
    rep   = pd.read_csv(os.path.join(BASE, "dataset/03_repayments.csv"))
    beh   = pd.read_csv(os.path.join(BASE, "dataset/04_behavioral_signals.csv"))
    out   = pd.read_csv(os.path.join(BASE, "dataset/05_outcomes.csv"))

    loans["origination_date"] = pd.to_datetime(loans["origination_date"])
    loans["cohort_quarter"]   = loans["origination_date"].dt.to_period("Q").astype(str)
    loans["cohort_month"]     = loans["origination_date"].dt.to_period("M").astype(str)
    rep["due_date"]           = pd.to_datetime(rep["due_date"])
    rep["due_month"]          = rep["due_date"].dt.to_period("M").astype(str)
    rep["cal_month"]          = rep["due_date"].dt.month

    master = (loans
              .merge(cust, on="customer_id", how="left")
              .merge(out,  on=["loan_id","customer_id"], how="left"))
    active = master[master["cooling_off_exit"] == 0].copy()

    # ticket / tenure bands
    active["ticket_band"] = pd.cut(
        active["ticket_size"],
        bins=[0, 25000, 50000, 100000, 200000, 500000, 2e6],
        labels=["<25K", "25-50K", "50-100K", "100-200K", "200-500K", "500K+"]
    )
    active["tenure_band"] = pd.cut(
        active["tenure_months"],
        bins=[0, 3, 6, 12, 24, 36, 48, 60],
        labels=["1-3M", "4-6M", "7-12M", "13-24M", "25-36M", "37-48M", "49-60M"]
    )

    return cust, loans, rep, beh, out, master, active

@st.cache_data(show_spinner="Loading model artefacts…")
def load_model_outputs():
    with open(os.path.join(BASE, "model_output/metrics.json")) as f:
        metrics = json.load(f)
    feat_imp = pd.read_csv(os.path.join(BASE, "model_output/feature_importance.csv"))
    scorecard = pd.read_csv(os.path.join(BASE, "model_output/scorecard.csv"), nrows=15000)
    roc       = pd.read_csv(os.path.join(BASE, "model_output/roc_data.csv"))
    cm        = pd.read_csv(os.path.join(BASE, "model_output/confusion_matrix.csv"), index_col=0)
    return metrics, feat_imp, scorecard, roc, cm


cust, loans, rep, beh, out, master, active = load_data()
metrics, feat_imp, scorecard, roc_data, cm = load_model_outputs()


# ─── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-header">
      <h2>📊 CAC 2026</h2>
      <p>Digital Lending · Strategic Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown(f"<p style='color:{SUB}; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em;'>Portfolio at a Glance</p>", unsafe_allow_html=True)
    total_loans   = len(active)
    default_rate  = active["default_flag"].mean() * 100
    collection_eff = rep.groupby("due_month").agg(p=("amount_paid","sum"), d=("emi_due","sum")).assign(e=lambda x: x.p/x.d*100)["e"].mean()
    avg_ltv       = active["customer_ltv"].mean()
    npa_cr        = (active[active["default_flag"]==1]["ticket_size"].sum()) / 1e7

    st.metric("Active Loans",       f"{total_loans:,}")
    st.metric("Overall Default Rate", f"{default_rate:.1f}%", delta=f"95% CI ±0.2pp")
    st.metric("Collection Efficiency", f"{collection_eff:.1f}%")
    st.metric("Avg Customer LTV",    f"₹{avg_ltv:,.0f}")
    st.metric("NPA Exposure",        f"₹{npa_cr:.1f} Cr")

    st.divider()
    st.markdown(f"<p style='color:{SUB}; font-size:12px;'>50K customers · 70K loans · Jan 2021–Jun 2024</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SUB}; font-size:12px;'>Model: LightGBM EWS · AUC {metrics['auc_test']:.3f}</p>", unsafe_allow_html=True)


# ─── PORTFOLIO OVERVIEW HEADER ───────────────────────────────
st.markdown(f"<h1 style='color:{TEXT}; font-size:28px; font-weight:700; margin-bottom:4px;'>Digital Lending Portfolio · Strategic Intelligence</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{SUB}; font-size:14px; margin-bottom:20px;'>CAC 2026 Project · 50,000 Customers · 70,000 Loans · GradientBoosting EWS</p>", unsafe_allow_html=True)

# KPI row
c1, c2, c3, c4, c5, c6 = st.columns(6)
kpi_data = [
    (c1, "Customers",          "50,000",      None,     None),
    (c2, "Total Loans",        "70,000",      None,     None),
    (c3, "Default Rate",       f"{default_rate:.2f}%", "-0.5pp YTD", False),
    (c4, "Collection Eff.",    f"{collection_eff:.1f}%", "+0.1pp",   True),
    (c5, "Avg LTV",            f"₹{avg_ltv/1000:.0f}K",  None,       None),
    (c6, "EWS AUC",            f"{metrics['auc_test']:.3f}", "LightGBM", None),
]
for col, label, val, delta, inv in kpi_data:
    with col:
        if delta:
            st.metric(label, val, delta=delta, delta_color="inverse" if inv else "normal")
        else:
            st.metric(label, val)

st.markdown("---")

# ─── MAIN TABS ───────────────────────────────────────────────
tab_overview, tab_q1, tab_q2, tab_q3, tab_q4, tab_q5 = st.tabs([
    "🏠 Overview",
    "Q1 · Customer Segments",
    "Q2 · Acquisition Channels",
    "Q3 · Products & Structure",
    "Q4 · Pricing & Approval",
    "Q5 · Leadership Monitoring",
])


# ══════════════════════════════════════════════════════════════
# OVERVIEW TAB
# ══════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<p class="section-header">Portfolio Summary</p>', unsafe_allow_html=True)

    # Default rate by tier — donut + bar side by side
    col_a, col_b, col_c = st.columns([1.2, 1, 1])

    with col_a:
        tier_vol = active.groupby("risk_tier")["loan_id"].count().reindex(TIER_ORDER)
        fig_donut = go.Figure(go.Pie(
            labels=tier_vol.index, values=tier_vol.values,
            hole=0.55,
            marker=dict(colors=[RISK_COLORS[t] for t in tier_vol.index], line=dict(color=BG, width=2)),
            textfont=dict(color=TEXT, size=12),
            hovertemplate="<b>%{label}</b><br>%{value:,} loans<br>%{percent}<extra></extra>",
        ))
        fig_donut.add_annotation(text="Portfolio<br>Mix", x=0.5, y=0.5, font=dict(size=14, color=TEXT), showarrow=False)
        apply_layout(fig_donut, "Loan Volume by Risk Tier", height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        tier_dr = active.groupby("risk_tier")["default_flag"].mean().mul(100).reindex(TIER_ORDER)
        fig_dr = go.Figure(go.Bar(
            x=TIER_ORDER, y=tier_dr.values,
            marker=dict(color=[RISK_COLORS[t] for t in TIER_ORDER],
                        line=dict(color=BG, width=1)),
            text=[f"{v:.1f}%" for v in tier_dr.values],
            textposition="outside",
            textfont=dict(color=TEXT, size=11),
            hovertemplate="<b>%{x}</b><br>Default Rate: %{y:.1f}%<extra></extra>",
        ))
        apply_layout(fig_dr, "Default Rate by Risk Tier", height=320)
        fig_dr.update_yaxes(title_text="Default Rate (%)", title_font=dict(color=SUB))
        st.plotly_chart(fig_dr, use_container_width=True)

    with col_c:
        # LTV & RAR by tier
        ltv_rar = active.groupby("risk_tier").agg(
            avg_ltv=("customer_ltv","mean"),
            avg_rar=("risk_adjusted_return","mean")
        ).reindex(TIER_ORDER)
        fig_eco = make_subplots(specs=[[{"secondary_y": True}]])
        fig_eco.add_trace(go.Bar(
            name="Avg LTV (₹K)", x=TIER_ORDER, y=ltv_rar["avg_ltv"]/1000,
            marker_color=[RISK_COLORS[t] for t in TIER_ORDER],
            text=[f"₹{v:.0f}K" for v in ltv_rar["avg_ltv"]/1000],
            textposition="outside", textfont=dict(color=TEXT, size=10),
        ), secondary_y=False)
        fig_eco.add_trace(go.Scatter(
            name="RAR", x=TIER_ORDER, y=ltv_rar["avg_rar"]*100,
            mode="lines+markers", line=dict(color=AMBER, width=2.5),
            marker=dict(size=8, color=AMBER),
        ), secondary_y=True)
        fig_eco.update_layout(height=320, paper_bgcolor=BG, plot_bgcolor=CARD,
                               font=dict(color=TEXT), legend=dict(bgcolor=CARD, bordercolor=BORDER),
                               margin=dict(l=40,r=40,t=50,b=40),
                               title=dict(text="LTV & Risk-Adjusted Return", font=dict(size=14,color=TEXT)))
        fig_eco.update_xaxes(gridcolor=BORDER)
        fig_eco.update_yaxes(gridcolor=BORDER, title_text="Avg LTV (₹K)", secondary_y=False)
        fig_eco.update_yaxes(title_text="RAR (%)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_eco, use_container_width=True)

    # Repayment status breakdown
    st.markdown('<p class="section-header">Repayment & Behavioral Signals</p>', unsafe_allow_html=True)
    col_d, col_e = st.columns(2)

    with col_d:
        ps_counts = rep["payment_status"].value_counts()
        status_colors = {"Paid-On-Time": GREEN, "Late-1-30": AMBER, "Late-31-60": ORANGE,
                         "Partial": BLUE, "Missed": RED}
        fig_ps = go.Figure(go.Pie(
            labels=ps_counts.index, values=ps_counts.values, hole=0.45,
            marker=dict(colors=[status_colors.get(s, PURPLE) for s in ps_counts.index],
                        line=dict(color=BG, width=2)),
            textfont=dict(color=TEXT, size=11),
            hovertemplate="<b>%{label}</b><br>%{value:,} records<br>%{percent}<extra></extra>",
        ))
        fig_ps.add_annotation(text="Payment<br>Status", x=0.5, y=0.5, font=dict(size=12, color=TEXT), showarrow=False)
        apply_layout(fig_ps, "Repayment Status Mix (1.14M Records)", height=300)
        st.plotly_chart(fig_ps, use_container_width=True)

    with col_e:
        # Seasonal on-time rate
        seasonal = rep.groupby("cal_month")["payment_status"].apply(
            lambda x: (x=="Paid-On-Time").mean()*100
        ).reset_index()
        seasonal.columns = ["Month","OnTime%"]
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        seasonal["MonthName"] = seasonal["Month"].map(month_names)
        festival    = [4, 10, 11]
        bar_colors  = [GREEN if m in festival else BLUE for m in seasonal["Month"]]
        fig_season  = go.Figure(go.Bar(
            x=seasonal["MonthName"], y=seasonal["OnTime%"],
            marker_color=bar_colors, text=[f"{v:.1f}%" for v in seasonal["OnTime%"]],
            textposition="outside", textfont=dict(color=TEXT, size=9),
            hovertemplate="<b>%{x}</b><br>On-Time: %{y:.2f}%<extra></extra>",
        ))
        fig_season.add_annotation(x=0.72, y=0.95, xref="paper", yref="paper",
                                   text="🟢 Festival months", showarrow=False,
                                   font=dict(color=GREEN, size=11))
        apply_layout(fig_season, "Seasonal On-Time Payment Rate (Festival Effect +5.9pp)", height=300)
        st.plotly_chart(fig_season, use_container_width=True)

    # Key insights row
    st.markdown('<p class="section-header">Key Cross-Cutting Insights</p>', unsafe_allow_html=True)
    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        st.markdown(f"""<div class="insight-box">
        <b>📌 Sweet Spot:</b> Near-Prime segment achieves the <b>highest RAR (0.200)</b>
        — better than even Prime — because APR sufficiently compensates for the 11% default rate.
        Grow this segment with disciplined FOIR controls.
        </div>""", unsafe_allow_html=True)
    with ci2:
        st.markdown(f"""<div class="warn-box">
        <b>⚠️ Channel Penalty:</b> Defaulted borrowers had <b>+14% higher CAC</b> (₹1,454 vs ₹1,277).
        Paid-Digital & DSA-Agent channels are the double-penalty combination —
        most expensive AND highest default risk.
        </div>""", unsafe_allow_html=True)
    with ci3:
        st.markdown(f"""<div class="insight-box">
        <b>🔔 EWS Signal:</b> All 3 behavioral signals (CFC, Balance Volatility, FOIR)
        deteriorate significantly <b>3 months before default</b>. The EWS model
        catches <b>35% of delinquencies</b> before DPD-30 at AUC {metrics['auc_test']:.3f}.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Q1 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════
with tab_q1:
    st.markdown(f"""
    <p class="section-header">Q1 · Customer Segments — Risk & Repayment Behaviors</p>
    <p style='color:{SUB}; font-size:13px; margin-bottom:16px;'>
    Which customer segments exhibit materially different risk and repayment behaviors,
    and what attributes define these segments?
    </p>""", unsafe_allow_html=True)

    # ── Segment KPIs ──
    seg_stats = active.groupby("risk_tier").agg(
        n_loans        = ("loan_id","count"),
        default_rate   = ("default_flag","mean"),
        avg_income     = ("monthly_income","mean"),
        avg_bureau     = ("bureau_score","mean"),
        profitable_pct = ("is_profitable","mean"),
        avg_ltv        = ("customer_ltv","mean"),
        avg_rar        = ("risk_adjusted_return","mean"),
    ).reindex(TIER_ORDER)
    seg_stats["default_rate"]   *= 100
    seg_stats["profitable_pct"] *= 100
    seg_stats["avg_rar"]        *= 100

    cols_kpi = st.columns(4)
    tier_configs = [
        ("Prime",      GREEN, "🟢"),
        ("Near-Prime", BLUE,  "🔵"),
        ("Subprime",   AMBER, "🟡"),
        ("Thin-File",  RED,   "🔴"),
    ]
    for col, (tier, color, icon) in zip(cols_kpi, tier_configs):
        with col:
            row = seg_stats.loc[tier]
            st.markdown(f"""
            <div style='background:{CARD}; border:1px solid {color}40; border-left:4px solid {color};
                        border-radius:10px; padding:16px;'>
              <div style='font-size:11px; color:{SUB}; font-weight:600; text-transform:uppercase; letter-spacing:0.06em;'>{icon} {tier}</div>
              <div style='font-size:26px; font-weight:700; color:{color}; margin:6px 0;'>{row.default_rate:.1f}%</div>
              <div style='font-size:11px; color:{SUB};'>Default Rate</div>
              <hr style='border-color:{BORDER}; margin:8px 0;'>
              <div style='font-size:12px; color:{TEXT};'>₹{row.avg_income/1000:.0f}K Avg Income</div>
              <div style='font-size:12px; color:{TEXT};'>Bureau: {row.avg_bureau:.0f}</div>
              <div style='font-size:12px; color:{TEXT};'>{row.profitable_pct:.1f}% Profitable</div>
              <div style='font-size:12px; color:{color}; font-weight:600;'>₹{row.avg_ltv/1000:.0f}K Avg LTV</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: Employment heatmap + Behavioral signals ──
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        from matplotlib.colors import LinearSegmentedColormap
        emp_order   = ["Salaried-Govt","Salaried-Private","Self-Employed-Prof",
                       "Self-Employed-Business","Gig-Worker","Daily-Wage","Unemployed","Student"]
        def_heat = (active.groupby(["employment_type","risk_tier"])["default_flag"]
                    .mean().mul(100).round(2).unstack(fill_value=0)
                    .reindex(columns=TIER_ORDER, fill_value=0)
                    .reindex([e for e in emp_order if e in active["employment_type"].unique()], fill_value=0))

        z_vals = def_heat.values
        fig_heat = go.Figure(go.Heatmap(
            z=z_vals,
            x=def_heat.columns.tolist(),
            y=def_heat.index.tolist(),
            colorscale=[[0, GREEN], [0.4, AMBER], [1, RED]],
            text=[[f"{v:.1f}%" for v in row] for row in z_vals],
            texttemplate="%{text}",
            textfont=dict(size=11, color=TEXT),
            hovertemplate="<b>%{y} × %{x}</b><br>Default Rate: %{z:.1f}%<extra></extra>",
            colorbar=dict(title=dict(text="Default Rate (%)", font=dict(color=SUB)), tickfont=dict(color=SUB)),
        ))
        apply_layout(fig_heat, "Employment × Risk Tier — Default Rate Heatmap (%) [Cramer's V = 0.60]", height=380)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        # Pre-default behavioral signal deterioration
        signals = ["Cash Flow Consistency", "Balance Volatility", "FOIR (EMI Ratio)"]
        normal_vals   = [0.6935, 0.2410, 0.1362]
        predef_6to4   = [0.0618, 0.4677, 0.1958]
        predef_3to1   = [0.0556, 0.4694, 0.2266]

        fig_sig = go.Figure()
        for i, (sig, nv, p64, p31) in enumerate(zip(signals, normal_vals, predef_6to4, predef_3to1)):
            fig_sig.add_trace(go.Bar(name="Normal", x=[sig], y=[nv],
                                     marker_color=GREEN, showlegend=(i==0),
                                     hovertemplate=f"<b>{sig}</b><br>Normal: {nv}<extra></extra>"))
            fig_sig.add_trace(go.Bar(name="Pre-Default 6-4M", x=[sig], y=[p64],
                                     marker_color=AMBER, showlegend=(i==0),
                                     hovertemplate=f"<b>{sig}</b><br>6-4M before: {p64}<extra></extra>"))
            fig_sig.add_trace(go.Bar(name="Pre-Default 3-1M", x=[sig], y=[p31],
                                     marker_color=RED, showlegend=(i==0),
                                     hovertemplate=f"<b>{sig}</b><br>3-1M before: {p31}<extra></extra>"))
        fig_sig.update_layout(barmode="group", height=380,
                               paper_bgcolor=BG, plot_bgcolor=CARD,
                               font=dict(color=TEXT), margin=dict(l=40,r=20,t=50,b=40),
                               legend=dict(bgcolor=CARD, bordercolor=BORDER),
                               title=dict(text="Behavioral Signal Deterioration Before Default (all p<0.001)",
                                          font=dict(size=14, color=TEXT)))
        fig_sig.update_xaxes(gridcolor=BORDER)
        fig_sig.update_yaxes(gridcolor=BORDER)
        st.plotly_chart(fig_sig, use_container_width=True)

    # ── Row 2: DPD Roll Rates + IV Rankings ──
    col3, col4 = st.columns(2)

    with col3:
        # DPD Roll-Rate matrix
        from_states = ["Current", "DPD 1-30", "DPD 31-60"]
        to_states   = ["Current", "DPD 1-30", "DPD 31-60", "DPD 61-90", "DPD 90+"]
        z_roll = [
            [87.57, 6.24, 6.19, 0.00, 0.00],
            [82.52, 8.86, 8.62, 0.00, 0.00],
            [83.63, 8.37, 8.01, 0.00, 0.00],
        ]
        fig_roll = go.Figure(go.Heatmap(
            z=z_roll, x=to_states, y=from_states,
            colorscale=[[0, GREEN],[0.5,AMBER],[1,RED]],
            text=[[f"{v:.1f}%" for v in row] for row in z_roll],
            texttemplate="%{text}", textfont=dict(size=12, color=TEXT),
            colorbar=dict(title=dict(text="Probability (%)", font=dict(color=SUB)), tickfont=dict(color=SUB)),
            hovertemplate="<b>%{y} → %{x}</b><br>Probability: %{z:.1f}%<extra></extra>",
        ))
        apply_layout(fig_roll, "DPD Roll-Rate Transition Matrix (Cure Rate DPD 1-30→Current = 82.5%)", height=320)
        st.plotly_chart(fig_roll, use_container_width=True)

    with col4:
        # IV ranking bar
        iv_data = pd.DataFrame({
            "Feature":    ["Risk Grade","Risk Tier","Bureau Score","Emp. Type",
                           "Monthly Income","APR","Emp. Stability","Acq. Channel",
                           "Ticket Size","Tenure Months","Geo Tier"],
            "IV":         [0.5298,0.4850,0.3820,0.3179,0.2706,0.2578,0.2118,
                           0.0786,0.0757,0.0046,0.0001],
            "Strength":   ["Strong","Strong","Strong","Strong","Medium","Medium","Medium",
                           "Weak","Weak","Useless","Useless"],
        }).sort_values("IV", ascending=True)
        color_map = {"Strong": GREEN, "Medium": BLUE, "Weak": AMBER, "Useless": RED}
        fig_iv = go.Figure(go.Bar(
            x=iv_data["IV"], y=iv_data["Feature"],
            orientation="h",
            marker=dict(color=[color_map[s] for s in iv_data["Strength"]]),
            text=[f"{v:.4f} [{s}]" for v, s in zip(iv_data["IV"], iv_data["Strength"])],
            textposition="outside", textfont=dict(color=TEXT, size=10),
            hovertemplate="<b>%{y}</b><br>IV: %{x:.4f}<extra></extra>",
        ))
        fig_iv.add_vline(x=0.3, line_dash="dash", line_color=GREEN, annotation_text="Strong (0.3)",
                         annotation_font_color=GREEN)
        fig_iv.add_vline(x=0.1, line_dash="dash", line_color=AMBER, annotation_text="Weak (0.1)",
                         annotation_font_color=AMBER)
        apply_layout(fig_iv, "Information Value (IV) — Feature Predictive Power Ranking", height=320)
        fig_iv.update_xaxes(title_text="IV Score", title_font=dict(color=SUB))
        st.plotly_chart(fig_iv, use_container_width=True)

    st.markdown("""<div class="insight-box">
    <b>Key Finding:</b> The portfolio has four behaviorally distinct segments. <b>Near-Prime is the strategic sweet spot</b> —
    highest RAR (0.200) despite 11% default rate. <b>Subprime is the highest-risk</b> (1 in 5 loans defaults).
    Employment type (Cramer's V = 0.60) is the strongest structural risk signal:
    <span class="risk-chip chip-green">Salaried-Govt → Prime</span>
    <span class="risk-chip chip-red">Gig/Daily-Wage → Thin-File/Subprime</span>.
    Behavioral signals deteriorate measurably 3 months before default — the EWS is statistically validated.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Q2 — ACQUISITION CHANNELS
# ══════════════════════════════════════════════════════════════
with tab_q2:
    st.markdown(f"""
    <p class="section-header">Q2 · Acquisition Channels & Onboarding Impact</p>
    <p style='color:{SUB}; font-size:13px; margin-bottom:16px;'>
    How do acquisition channels and onboarding strategies impact portfolio quality,
    customer lifetime value, and unit economics?
    </p>""", unsafe_allow_html=True)

    ch_order = (active.groupby("acquisition_channel")["loan_id"]
                .count().sort_values(ascending=False).index.tolist())
    ch_stats = active.groupby("acquisition_channel").agg(
        n             = ("loan_id","count"),
        default_rate  = ("default_flag","mean"),
        avg_cac       = ("cost_of_acquisition","mean"),
        avg_ltv       = ("customer_ltv","mean"),
        profitable_pct= ("is_profitable","mean"),
        avg_rar       = ("risk_adjusted_return","mean"),
    ).reindex(ch_order)
    ch_stats["default_rate"]   *= 100
    ch_stats["profitable_pct"] *= 100
    ch_stats["avg_rar"]        *= 100
    ch_stats["ltv_cac"]         = ch_stats["avg_ltv"] / ch_stats["avg_cac"]
    ch_stats["vol_pct"]         = ch_stats["n"] / ch_stats["n"].sum() * 100

    # Cooling-off from outcomes (include all)
    cool_by_ch = master.groupby("acquisition_channel").agg(
        n=("loan_id","count"), exits=("cooling_off_exit","sum")
    ).reset_index()
    cool_by_ch["exit_rate"] = cool_by_ch["exits"] / cool_by_ch["n"] * 100
    cool_by_ch = cool_by_ch.set_index("acquisition_channel").reindex(ch_order)
    ch_stats["cooling_off"] = cool_by_ch["exit_rate"]

    bar_colors = [CH_COLORS.get(c, BLUE) for c in ch_stats.index]

    # ── Channel KPI cards ──
    col_ch_kpi = st.columns(len(ch_order))
    for col, ch in zip(col_ch_kpi, ch_order):
        row = ch_stats.loc[ch]
        color = CH_COLORS.get(ch, BLUE)
        with col:
            st.markdown(f"""
            <div style='background:{CARD}; border:1px solid {color}40; border-left:3px solid {color};
                        border-radius:8px; padding:10px; text-align:center;'>
              <div style='font-size:9px; color:{SUB}; font-weight:600; text-transform:uppercase;'>{ch.replace("-"," ")}</div>
              <div style='font-size:18px; font-weight:700; color:{color};'>₹{row.avg_cac:.0f}</div>
              <div style='font-size:9px; color:{SUB};'>CAC</div>
              <div style='font-size:11px; color:{TEXT};'>{row.default_rate:.1f}% Default</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: CAC vs Default + Volume Share ──
    col1, col2 = st.columns(2)

    with col1:
        # Scatter: CAC vs Default Rate, bubble = volume
        fig_scat = go.Figure()
        for ch in ch_order:
            row = ch_stats.loc[ch]
            color = CH_COLORS.get(ch, BLUE)
            fig_scat.add_trace(go.Scatter(
                x=[row.avg_cac], y=[row.default_rate],
                mode="markers+text",
                marker=dict(size=row.n/100, color=color, opacity=0.8,
                            line=dict(color="white", width=1.5)),
                text=[ch], textposition="top center",
                textfont=dict(color=TEXT, size=10),
                name=ch,
                hovertemplate=f"<b>{ch}</b><br>CAC: ₹{{x:,.0f}}<br>Default: {{y:.1f}}%<br>Volume: {row.n:,}<extra></extra>",
            ))
        fig_scat.add_vline(x=ch_stats["avg_cac"].mean(), line_dash="dash", line_color=SUB)
        fig_scat.add_hline(y=ch_stats["default_rate"].mean(), line_dash="dash", line_color=SUB)
        apply_layout(fig_scat, "CAC vs Default Rate by Channel (Bubble Size = Volume) — Double Penalty Zone", height=380)
        fig_scat.update_xaxes(title_text="Avg CAC (₹)", title_font=dict(color=SUB))
        fig_scat.update_yaxes(title_text="Default Rate (%)", title_font=dict(color=SUB))
        fig_scat.update_layout(showlegend=False)
        st.plotly_chart(fig_scat, use_container_width=True)

    with col2:
        # LTV/CAC Ratio bar
        fig_ltvcac = go.Figure(go.Bar(
            x=ch_stats.index, y=ch_stats["ltv_cac"],
            marker=dict(color=bar_colors, line=dict(color=BG, width=1)),
            text=[f"{v:.1f}x" for v in ch_stats["ltv_cac"]],
            textposition="outside", textfont=dict(color=TEXT, size=11),
            hovertemplate="<b>%{x}</b><br>LTV/CAC: %{y:.2f}x<extra></extra>",
        ))
        fig_ltvcac.add_hline(y=1.0, line_dash="dash", line_color=RED,
                              annotation_text="Break-Even = 1.0x", annotation_font_color=RED)
        apply_layout(fig_ltvcac, "LTV / CAC Ratio by Channel — Unit Economics Health", height=380)
        fig_ltvcac.update_yaxes(title_text="LTV / CAC Ratio", title_font=dict(color=SUB))
        st.plotly_chart(fig_ltvcac, use_container_width=True)

    # ── Row 2: Cooling-Off + RAR ──
    col3, col4 = st.columns(2)

    with col3:
        fig_cool = go.Figure(go.Bar(
            x=ch_stats.index, y=ch_stats["cooling_off"],
            marker=dict(color=bar_colors, line=dict(color=BG, width=1)),
            text=[f"{v:.1f}%" for v in ch_stats["cooling_off"]],
            textposition="outside", textfont=dict(color=TEXT, size=11),
            hovertemplate="<b>%{x}</b><br>Cooling-Off Exit: %{y:.1f}%<extra></extra>",
        ))
        fig_cool.add_hline(y=7.0, line_dash="dash", line_color=RED,
                           annotation_text="⚠ RBI Compliance Flag (>7%)", annotation_font_color=RED)
        apply_layout(fig_cool, "Cooling-Off Exit Rate by Channel (%) — RBI Compliance Monitor", height=330)
        fig_cool.update_yaxes(title_text="Exit Rate (%)", title_font=dict(color=SUB))
        st.plotly_chart(fig_cool, use_container_width=True)

    with col4:
        fig_rar = go.Figure(go.Bar(
            x=ch_stats.index, y=ch_stats["avg_rar"],
            marker=dict(color=bar_colors, line=dict(color=BG, width=1)),
            text=[f"{v:.1f}%" for v in ch_stats["avg_rar"]],
            textposition="outside", textfont=dict(color=TEXT, size=11),
            hovertemplate="<b>%{x}</b><br>RAR: %{y:.2f}%<extra></extra>",
        ))
        fig_rar.add_hline(y=0, line_dash="dash", line_color=RED)
        apply_layout(fig_rar, "Risk-Adjusted Return (%) by Channel — True Profitability", height=330)
        fig_rar.update_yaxes(title_text="RAR (%)", title_font=dict(color=SUB))
        st.plotly_chart(fig_rar, use_container_width=True)

    # ── Channel summary table ──
    st.markdown('<p class="section-header">Channel Integrated Scorecard</p>', unsafe_allow_html=True)
    def channel_row_style(row):
        if row["Channel"] in ["Paid-Digital", "DSA-Agent"]:
            return [f"color:{RED}"] * len(row)
        elif row["Channel"] in ["Referral", "Organic-App"]:
            return [f"color:{GREEN}"] * len(row)
        return [""] * len(row)

    ch_table = pd.DataFrame({
        "Channel":       ch_stats.index,
        "Volume (Loans)":ch_stats["n"].map(lambda x: f"{x:,}"),
        "Vol Share":     ch_stats["vol_pct"].map(lambda x: f"{x:.1f}%"),
        "Mean CAC (₹)":  ch_stats["avg_cac"].map(lambda x: f"₹{x:,.0f}"),
        "Default Rate":  ch_stats["default_rate"].map(lambda x: f"{x:.1f}%"),
        "Avg LTV (₹)":   ch_stats["avg_ltv"].map(lambda x: f"₹{x:,.0f}"),
        "LTV/CAC":       ch_stats["ltv_cac"].map(lambda x: f"{x:.1f}x"),
        "RAR":           ch_stats["avg_rar"].map(lambda x: f"{x:.1f}%"),
        "Cool-Off Exit": ch_stats["cooling_off"].map(lambda x: f"{x:.1f}%"),
    })
    st.dataframe(ch_table, use_container_width=True, hide_index=True)

    st.markdown("""<div class="warn-box">
    <b>⚠️ Strategic Alert — The Double Penalty:</b>
    <b>Paid-Digital</b> (27.9% of volume, ₹2,310 CAC) and <b>DSA-Agent</b> (₹1,575 CAC) channels
    are simultaneously the most expensive AND highest-default channels.
    Defaulted borrowers had <b>+14% higher CAC</b> (₹1,454 vs ₹1,277) — a structural correlation.
    Shifting <b>10% of volume</b> from Paid-Digital to Referral/Organic saves ~₹1,890/loan CAC
    AND reduces portfolio PD by ~2-3pp simultaneously. <b>Cooling-off rates >6.9% in these channels
    are a live RBI compliance flag.</b>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Q3 — PRODUCTS, TICKET SIZES & TENURES
# ══════════════════════════════════════════════════════════════
with tab_q3:
    st.markdown(f"""
    <p class="section-header">Q3 · Loan Products, Ticket Sizes & Tenures</p>
    <p style='color:{SUB}; font-size:13px; margin-bottom:16px;'>
    Which loan products, ticket sizes, and tenures deliver the strongest balance
    between growth and risk?
    </p>""", unsafe_allow_html=True)

    prod_stats = active.groupby("product_type").agg(
        n             = ("loan_id","count"),
        default_rate  = ("default_flag","mean"),
        avg_ticket    = ("ticket_size","mean"),
        median_ticket = ("ticket_size","median"),
        avg_apr       = ("apr","mean"),
        avg_tenure    = ("tenure_months","mean"),
        profitable_pct= ("is_profitable","mean"),
        avg_ltv       = ("customer_ltv","mean"),
        avg_rar       = ("risk_adjusted_return","mean"),
    ).reset_index()
    prod_stats["default_rate"]   *= 100
    prod_stats["profitable_pct"] *= 100
    prod_stats["avg_apr"]        *= 100
    prod_stats["avg_rar"]        *= 100
    prod_stats["vol_pct"]         = prod_stats["n"] / prod_stats["n"].sum() * 100

    prod_colors = {p: CSEQ[i % len(CSEQ)] for i, p in enumerate(prod_stats["product_type"])}

    # ── Row 1: Growth-Risk scatter + Product volume bar ──
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        avg_dr   = prod_stats["default_rate"].mean()
        avg_prof = prod_stats["profitable_pct"].mean()
        fig_prod_scatter = go.Figure()

        for i, row in prod_stats.iterrows():
            color = CSEQ[i % len(CSEQ)]
            fig_prod_scatter.add_trace(go.Scatter(
                x=[row["default_rate"]], y=[row["profitable_pct"]],
                mode="markers+text",
                marker=dict(size=row["n"]/60, color=color, opacity=0.85,
                            line=dict(color="white", width=1.5)),
                text=[row["product_type"]], textposition="top center",
                textfont=dict(color=TEXT, size=10),
                name=row["product_type"],
                hovertemplate=(
                    f"<b>{row['product_type']}</b><br>"
                    f"Default Rate: {row['default_rate']:.1f}%<br>"
                    f"Profitable: {row['profitable_pct']:.1f}%<br>"
                    f"Avg Ticket: ₹{row['avg_ticket']/1000:.0f}K<br>"
                    f"Volume: {row['n']:,}<extra></extra>"
                ),
            ))

        fig_prod_scatter.add_vline(x=avg_dr, line_dash="dot", line_color=SUB,
                                   annotation_text="Avg Default", annotation_font_color=SUB)
        fig_prod_scatter.add_hline(y=avg_prof, line_dash="dot", line_color=SUB,
                                   annotation_text="Avg Profitable%", annotation_font_color=SUB)

        # Quadrant labels
        fig_prod_scatter.add_annotation(x=avg_dr*0.4, y=avg_prof+3,
            text="🟢 Grow Aggressively", showarrow=False, font=dict(color=GREEN, size=10))
        fig_prod_scatter.add_annotation(x=avg_dr*1.6, y=avg_prof-5,
            text="🔴 Manage Carefully", showarrow=False, font=dict(color=RED, size=10))

        apply_layout(fig_prod_scatter,
                     "Product Growth vs Risk Matrix — Bubble = Volume, Axes = Default Rate vs Profitability",
                     height=400)
        fig_prod_scatter.update_xaxes(title_text="Default Rate (%) ← Lower is Better", title_font=dict(color=SUB))
        fig_prod_scatter.update_yaxes(title_text="Profitable Loans (%) ↑ Higher is Better", title_font=dict(color=SUB))
        fig_prod_scatter.update_layout(showlegend=False)
        st.plotly_chart(fig_prod_scatter, use_container_width=True)

    with col2:
        # Product volume + default side by side bar
        fig_prod_vd = make_subplots(rows=2, cols=1,
                                     subplot_titles=["Volume Share (%)", "Default Rate (%)"],
                                     vertical_spacing=0.15)
        prod_s = prod_stats.sort_values("n", ascending=False)
        colors_prod = [CSEQ[i % len(CSEQ)] for i in range(len(prod_s))]

        fig_prod_vd.add_trace(go.Bar(
            x=prod_s["product_type"], y=prod_s["vol_pct"],
            marker_color=colors_prod,
            text=[f"{v:.1f}%" for v in prod_s["vol_pct"]],
            textposition="outside", textfont=dict(color=TEXT, size=10),
            showlegend=False,
        ), row=1, col=1)

        fig_prod_vd.add_trace(go.Bar(
            x=prod_s["product_type"], y=prod_s["default_rate"],
            marker_color=colors_prod,
            text=[f"{v:.1f}%" for v in prod_s["default_rate"]],
            textposition="outside", textfont=dict(color=TEXT, size=10),
            showlegend=False,
        ), row=2, col=1)

        fig_prod_vd.update_layout(height=400, paper_bgcolor=BG, plot_bgcolor=CARD,
                                   font=dict(color=TEXT), margin=dict(l=40,r=20,t=50,b=60))
        fig_prod_vd.update_xaxes(tickangle=20, tickfont=dict(size=9), gridcolor=BORDER)
        fig_prod_vd.update_yaxes(gridcolor=BORDER)
        st.plotly_chart(fig_prod_vd, use_container_width=True)

    # ── Row 2: Ticket Bands + Tenure Bands ──
    col3, col4 = st.columns(2)

    with col3:
        tb_dr = (active.groupby("ticket_band", observed=True)["default_flag"]
                 .agg(["mean","count"]).reset_index())
        tb_dr.columns = ["band","default_rate","n"]
        tb_dr["default_rate"] *= 100
        tb_dr["band"] = tb_dr["band"].astype(str)

        fig_ticket = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ticket.add_trace(go.Bar(
            name="Volume", x=tb_dr["band"], y=tb_dr["n"],
            marker_color=BLUE, opacity=0.6,
            hovertemplate="<b>%{x}</b><br>Volume: %{y:,}<extra></extra>",
        ), secondary_y=False)
        fig_ticket.add_trace(go.Scatter(
            name="Default Rate", x=tb_dr["band"], y=tb_dr["default_rate"],
            mode="lines+markers", line=dict(color=RED, width=2.5),
            marker=dict(size=9, color=RED),
            hovertemplate="<b>%{x}</b><br>Default Rate: %{y:.1f}%<extra></extra>",
        ), secondary_y=True)
        fig_ticket.update_layout(height=350, paper_bgcolor=BG, plot_bgcolor=CARD,
                                  font=dict(color=TEXT), legend=dict(bgcolor=CARD, bordercolor=BORDER),
                                  margin=dict(l=40,r=40,t=50,b=40),
                                  title=dict(text="Ticket Size Band — Volume vs Default Rate",
                                             font=dict(size=14,color=TEXT)))
        fig_ticket.update_xaxes(gridcolor=BORDER)
        fig_ticket.update_yaxes(title_text="Loan Volume", secondary_y=False, gridcolor=BORDER)
        fig_ticket.update_yaxes(title_text="Default Rate (%)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_ticket, use_container_width=True)

    with col4:
        ten_dr = (active.groupby("tenure_band", observed=True)["default_flag"]
                  .agg(["mean","count"]).reset_index())
        ten_dr.columns = ["band","default_rate","n"]
        ten_dr["default_rate"] *= 100
        ten_dr["band"] = ten_dr["band"].astype(str)

        fig_tenure = make_subplots(specs=[[{"secondary_y": True}]])
        fig_tenure.add_trace(go.Bar(
            name="Volume", x=ten_dr["band"], y=ten_dr["n"],
            marker_color=PURPLE, opacity=0.6,
            hovertemplate="<b>%{x}</b><br>Volume: %{y:,}<extra></extra>",
        ), secondary_y=False)
        fig_tenure.add_trace(go.Scatter(
            name="Default Rate", x=ten_dr["band"], y=ten_dr["default_rate"],
            mode="lines+markers", line=dict(color=RED, width=2.5),
            marker=dict(size=9, color=RED),
            hovertemplate="<b>%{x}</b><br>Default Rate: %{y:.1f}%<extra></extra>",
        ), secondary_y=True)
        fig_tenure.update_layout(height=350, paper_bgcolor=BG, plot_bgcolor=CARD,
                                  font=dict(color=TEXT), legend=dict(bgcolor=CARD, bordercolor=BORDER),
                                  margin=dict(l=40,r=40,t=50,b=40),
                                  title=dict(text="Tenure Band — Volume vs Default Rate",
                                             font=dict(size=14,color=TEXT)))
        fig_tenure.update_xaxes(gridcolor=BORDER)
        fig_tenure.update_yaxes(title_text="Loan Volume", secondary_y=False, gridcolor=BORDER)
        fig_tenure.update_yaxes(title_text="Default Rate (%)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_tenure, use_container_width=True)

    # ── Product × Risk Tier Heatmap ──
    prod_tier_dr = (active.groupby(["product_type","risk_tier"])["default_flag"]
                    .mean().mul(100).round(2).unstack(fill_value=0)
                    .reindex(columns=TIER_ORDER, fill_value=0))
    fig_prod_heat = go.Figure(go.Heatmap(
        z=prod_tier_dr.values,
        x=prod_tier_dr.columns.tolist(),
        y=prod_tier_dr.index.tolist(),
        colorscale=[[0, GREEN],[0.4,AMBER],[1,RED]],
        text=[[f"{v:.1f}%" for v in row] for row in prod_tier_dr.values],
        texttemplate="%{text}", textfont=dict(size=11, color=TEXT),
        colorbar=dict(title=dict(text="Default Rate (%)", font=dict(color=SUB)), tickfont=dict(color=SUB)),
        hovertemplate="<b>%{y} × %{x}</b><br>Default Rate: %{z:.1f}%<extra></extra>",
    ))
    apply_layout(fig_prod_heat, "Default Rate (%) — Product × Risk Tier Heatmap", height=280)
    st.plotly_chart(fig_prod_heat, use_container_width=True)

    # Strategy table
    st.markdown('<p class="section-header">Product Strategy Matrix</p>', unsafe_allow_html=True)
    strat_df = pd.DataFrame([
        {"Product": "Two-Wheeler Loan",      "Median Ticket": "₹82K",  "Avg APR": "~14%", "Skew": "0.37 (Low)", "Strategy": "🟢 Grow Aggressively — Collateral-backed, lowest tail risk"},
        {"Product": "Consumer-Durable",      "Median Ticket": "₹50K",  "Avg APR": "~16%", "Skew": "0.59 (Low)", "Strategy": "🟢 Grow Aggressively — Predictable, tight ticket band"},
        {"Product": "BNPL",                  "Median Ticket": "₹25K",  "Avg APR": "~18%", "Skew": "0.63 (Low)", "Strategy": "🟡 Grow Selectively — Use as NTC onboarding ladder"},
        {"Product": "Personal Loan (Prime)", "Median Ticket": "₹133K", "Avg APR": "~12%", "Skew": "1.10",       "Strategy": "🟡 Grow Selectively — Largest segment, requires FOIR controls"},
        {"Product": "Education Loan",        "Median Ticket": "₹141K", "Avg APR": "~11%", "Skew": "1.14",       "Strategy": "🟡 Grow Selectively — Long tenure; moratorium risk"},
        {"Product": "SME-Working-Capital",   "Median Ticket": "₹115K", "Avg APR": "~15%", "Skew": "2.60 (High)","Strategy": "🔴 Manage Carefully — High tail risk, cash-flow underwriting needed"},
        {"Product": "Personal Loan (Sub.)",  "Median Ticket": "₹80K",  "Avg APR": "~22%", "Skew": "1.10",       "Strategy": "🔴 Restrict/Reprice — RAR marginal at 0.177; cap or re-price"},
    ])
    st.dataframe(strat_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# Q4 — PRICING, APPROVAL & TENURE STRATEGY
# ══════════════════════════════════════════════════════════════
with tab_q4:
    st.markdown(f"""
    <p class="section-header">Q4 · Pricing, Approval & Tenure Strategy</p>
    <p style='color:{SUB}; font-size:13px; margin-bottom:16px;'>
    How can pricing, approval, or tenure strategies be tailored across segments
    to improve overall portfolio outcomes?
    </p>""", unsafe_allow_html=True)

    grade_stats = active.groupby("origination_risk_grade").agg(
        actual_dr = ("default_flag","mean"),
        avg_apr   = ("apr","mean"),
        n         = ("loan_id","count"),
        avg_ltv   = ("customer_ltv","mean"),
        avg_rar   = ("risk_adjusted_return","mean"),
    ).reindex(GRADE_ORDER).dropna().reset_index()
    grade_stats["actual_dr"] *= 100
    grade_stats["avg_apr"]   *= 100
    grade_stats["avg_rar"]   *= 100
    grade_stats["fair_apr"]   = 8.0 + grade_stats["actual_dr"] * 2.0
    grade_stats["pricing_gap"]= grade_stats["avg_apr"] - grade_stats["fair_apr"]

    # ── Row 1: APR vs Default + Pricing Gap ──
    col1, col2 = st.columns(2)

    with col1:
        fig_apr = make_subplots(specs=[[{"secondary_y": True}]])
        grade_colors_bar = [GREEN]*3 + [BLUE]*2 + [AMBER]*2 + [RED]
        fig_apr.add_trace(go.Bar(
            name="Avg APR (%)", x=grade_stats["origination_risk_grade"], y=grade_stats["avg_apr"],
            marker_color=grade_colors_bar, opacity=0.75,
            text=[f"{v:.1f}%" for v in grade_stats["avg_apr"]],
            textposition="outside", textfont=dict(color=TEXT, size=10),
            hovertemplate="<b>Grade %{x}</b><br>APR: %{y:.1f}%<extra></extra>",
        ), secondary_y=False)
        fig_apr.add_trace(go.Scatter(
            name="Default Rate (%)", x=grade_stats["origination_risk_grade"], y=grade_stats["actual_dr"],
            mode="lines+markers", line=dict(color=RED, width=2.5, dash="solid"),
            marker=dict(size=9, color=RED),
            hovertemplate="<b>Grade %{x}</b><br>Default: %{y:.1f}%<extra></extra>",
        ), secondary_y=True)
        fig_apr.update_layout(height=380, paper_bgcolor=BG, plot_bgcolor=CARD,
                               font=dict(color=TEXT), legend=dict(bgcolor=CARD, bordercolor=BORDER),
                               margin=dict(l=40,r=40,t=50,b=40),
                               title=dict(text="APR Calibration vs Actual Default Rate by Risk Grade (Spearman ρ=0.42)",
                                          font=dict(size=13,color=TEXT)))
        fig_apr.update_xaxes(gridcolor=BORDER, title_text="Risk Grade")
        fig_apr.update_yaxes(title_text="Avg APR (%)", secondary_y=False, gridcolor=BORDER)
        fig_apr.update_yaxes(title_text="Default Rate (%)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_apr, use_container_width=True)

    with col2:
        gap_colors = [GREEN if v > 0 else RED for v in grade_stats["pricing_gap"]]
        fig_gap = go.Figure(go.Bar(
            x=grade_stats["origination_risk_grade"], y=grade_stats["pricing_gap"],
            marker=dict(color=gap_colors, line=dict(color=BG, width=1)),
            text=[f"{v:+.1f}pp" for v in grade_stats["pricing_gap"]],
            textposition="outside", textfont=dict(color=TEXT, size=11),
            hovertemplate="<b>Grade %{x}</b><br>Gap: %{y:+.2f}pp<br>(+ = over-priced, - = under-priced)<extra></extra>",
        ))
        fig_gap.add_hline(y=0, line_color=TEXT, line_width=1.5)
        fig_gap.add_annotation(x=0.02, y=0.95, xref="paper", yref="paper",
                                text="🟢 Over-priced = pricing compensation adequate",
                                showarrow=False, font=dict(color=GREEN, size=10))
        fig_gap.add_annotation(x=0.02, y=0.06, xref="paper", yref="paper",
                                text="🔴 Under-priced = insufficient risk premium",
                                showarrow=False, font=dict(color=RED, size=10))
        apply_layout(fig_gap, "Pricing Gap Analysis — APR vs Fair Rate (8% base + 2× Default Rate)", height=380)
        fig_gap.update_yaxes(title_text="Pricing Gap (pp)", title_font=dict(color=SUB))
        st.plotly_chart(fig_gap, use_container_width=True)

    # ── Row 2: Tenure × Risk Tier + Approval TAT ──
    col3, col4 = st.columns(2)

    with col3:
        ten_tier = active.groupby(["tenure_band","risk_tier"], observed=True).agg(
            default_rate=("default_flag","mean"),
        ).reset_index()
        ten_tier["default_rate"] *= 100
        ten_tier["tenure_band"]   = ten_tier["tenure_band"].astype(str)

        fig_ten_tier = go.Figure()
        for tier, color in RISK_COLORS.items():
            sub = ten_tier[ten_tier["risk_tier"]==tier].sort_values("tenure_band")
            fig_ten_tier.add_trace(go.Scatter(
                x=sub["tenure_band"], y=sub["default_rate"],
                mode="lines+markers", name=tier,
                line=dict(color=color, width=2.5),
                marker=dict(size=8, color=color),
                hovertemplate=f"<b>{tier}</b> · %{{x}}<br>Default: %{{y:.1f}}%<extra></extra>",
            ))
        apply_layout(fig_ten_tier, "Default Rate by Tenure Band × Risk Tier — Tenure Strategy Guide", height=350)
        fig_ten_tier.update_xaxes(title_text="Tenure Band", title_font=dict(color=SUB))
        fig_ten_tier.update_yaxes(title_text="Default Rate (%)", title_font=dict(color=SUB))
        st.plotly_chart(fig_ten_tier, use_container_width=True)

    with col4:
        tat_dr = active.groupby("approval_turnaround_days").agg(
            n=("loan_id","count"), default_rate=("default_flag","mean")
        ).reset_index()
        tat_dr["default_rate"] *= 100

        fig_tat = make_subplots(specs=[[{"secondary_y": True}]])
        fig_tat.add_trace(go.Bar(
            name="Volume", x=tat_dr["approval_turnaround_days"].astype(str), y=tat_dr["n"],
            marker_color=TEAL, opacity=0.7,
            hovertemplate="<b>TAT %{x} days</b><br>Volume: %{y:,}<extra></extra>",
        ), secondary_y=False)
        fig_tat.add_trace(go.Scatter(
            name="Default Rate", x=tat_dr["approval_turnaround_days"].astype(str), y=tat_dr["default_rate"],
            mode="lines+markers", line=dict(color=RED, width=2.5),
            marker=dict(size=9, color=RED),
            hovertemplate="<b>TAT %{x} days</b><br>Default: %{y:.1f}%<extra></extra>",
        ), secondary_y=True)
        fig_tat.update_layout(height=350, paper_bgcolor=BG, plot_bgcolor=CARD,
                               font=dict(color=TEXT), legend=dict(bgcolor=CARD, bordercolor=BORDER),
                               margin=dict(l=40,r=40,t=50,b=40),
                               title=dict(text="Approval Turnaround Days vs Default Rate (Speed ≠ Risk)",
                                          font=dict(size=13,color=TEXT)))
        fig_tat.update_xaxes(gridcolor=BORDER, title_text="Turnaround (Days)")
        fig_tat.update_yaxes(title_text="Loan Volume", secondary_y=False, gridcolor=BORDER)
        fig_tat.update_yaxes(title_text="Default Rate (%)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_tat, use_container_width=True)

    # ── Pricing recommendation table ──
    st.markdown('<p class="section-header">Tailored Pricing & Approval Recommendations</p>', unsafe_allow_html=True)
    c_pr1, c_pr2 = st.columns(2)

    with c_pr1:
        pricing_reco = pd.DataFrame([
            {"Segment": "Prime (A/A+)",          "Current APR": "11-12%",    "Recommended APR": "Keep — well-priced",  "FOIR Cap": "<50%", "Approval": "Fast-track"},
            {"Segment": "Near-Prime (B/B-)",      "Current APR": "14-15%",    "Recommended APR": "+20-30 bps",          "FOIR Cap": "<40%", "Approval": "Standard"},
            {"Segment": "Subprime (C/C+)",        "Current APR": "17-18%",    "Recommended APR": "Keep or +50 bps",     "FOIR Cap": "<35%", "Approval": "Selective"},
            {"Segment": "Thin-File / Grade D",    "Current APR": "15-16%",    "Recommended APR": "+150-200 bps ⚠",      "FOIR Cap": "<30%", "Approval": "Alt-data only"},
            {"Segment": "DSA-Agent Channel",      "Current APR": "Grade rate", "Recommended APR": "+50-75 bps premium",  "FOIR Cap": "Tighter", "Approval": "Extra review"},
            {"Segment": "Paid-Digital Channel",   "Current APR": "Grade rate", "Recommended APR": "+75-100 bps premium", "FOIR Cap": "Tighter", "Approval": "Extra review"},
        ])
        st.dataframe(pricing_reco, use_container_width=True, hide_index=True)

    with c_pr2:
        tenure_reco = pd.DataFrame([
            {"Segment": "BNPL / Consumer-Durable (Subprime)",  "Max Tenure": "12-18 months",  "Rationale": "Short exposure limits LGD; quick behavior assessment"},
            {"Segment": "Personal Loan (Near-Prime)",           "Max Tenure": "24-36 months",  "Rationale": "Balanced EMI affordability vs. vintage exposure"},
            {"Segment": "Personal Loan (Subprime)",             "Max Tenure": "24 months max", "Rationale": "Restricts interest income loss on early default"},
            {"Segment": "SME-Working-Capital",                  "Max Tenure": "3-18 months",   "Rationale": "Linked to cash-flow cycle; avoid long unsecured exposure"},
            {"Segment": "Education Loan",                       "Max Tenure": "60 months",     "Rationale": "Moratorium periods justified; monitor post-study income"},
            {"Segment": "Thin-File (NTC Ladder)",               "Max Tenure": "Start: 6-12M",  "Rationale": "BNPL → Consumer Durable → Personal Loan progression"},
        ])
        st.dataframe(tenure_reco, use_container_width=True, hide_index=True)

    st.markdown("""<div class="warn-box">
    <b>⚠️ Critical Finding — Grade D Under-pricing:</b>
    Grade D (Thin-File/NTC) carries a <b>15.4% default rate</b> (comparable to C+ at 16.8%)
    but is priced at a <b>lower median APR (20.69% vs. 22.83%)</b>. This structural under-pricing
    stems from insufficient bureau data. Immediate action: <b>Uplift Grade D APR by 150-200 bps</b>
    or implement behavioral-score-based dynamic pricing for NTC borrowers.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Q5 — SENIOR LEADERSHIP MONITORING
# ══════════════════════════════════════════════════════════════
with tab_q5:
    st.markdown(f"""
    <p class="section-header">Q5 · Senior Leadership Performance Monitoring</p>
    <p style='color:{SUB}; font-size:13px; margin-bottom:16px;'>
    What performance metrics and views should senior leadership monitor
    to proactively manage risk and growth?
    </p>""", unsafe_allow_html=True)

    # ── Live KPI Row ──
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    scorecard_data = metrics["scorecard"]
    red_delinq = next(b["actual_delinquency_rate"] for b in scorecard_data if b["bucket"]=="RED")
    amb_delinq = next(b["actual_delinquency_rate"] for b in scorecard_data if b["bucket"]=="AMBER")
    red_share  = next(b["share_pct"] for b in scorecard_data if b["bucket"]=="RED")

    with m1: st.metric("Overall Default Rate", f"{default_rate:.2f}%", delta="±0.2pp (95% CI)")
    with m2: st.metric("Collection Efficiency", f"{collection_eff:.1f}%", delta="+0.1pp", delta_color="normal")
    with m3: st.metric("EWS Model AUC", f"{metrics['auc_test']:.4f}", delta="Gini 24.9%")
    with m4: st.metric("🔴 RED Bucket Delinq.", f"{red_delinq:.1f}%", delta=f"{red_share:.1f}% of active loans", delta_color="inverse")
    with m5: st.metric("🟡 AMBER Delinq.", f"{amb_delinq:.1f}%", delta="Proactive window", delta_color="off")
    with m6: st.metric("Profitable Loans", f"{active['is_profitable'].mean()*100:.1f}%", delta="Target >72%")

    st.markdown("---")

    # ── Monthly Portfolio KPIs ──
    monthly_kpi = (rep.groupby("due_month").agg(
        total_due   = ("emi_due","sum"),
        total_paid  = ("amount_paid","sum"),
        on_time_cnt = ("payment_status", lambda x: (x=="Paid-On-Time").sum()),
        missed_cnt  = ("payment_status", lambda x: (x=="Missed").sum()),
        total_cnt   = ("loan_id","count"),
    ).assign(
        collection_eff = lambda d: d.total_paid / d.total_due * 100,
        on_time_rate   = lambda d: d.on_time_cnt / d.total_cnt * 100,
        missed_rate    = lambda d: d.missed_cnt  / d.total_cnt * 100,
    ).reset_index().sort_values("due_month").iloc[1:-1])

    col1, col2 = st.columns(2)

    with col1:
        fig_coll = go.Figure()
        fig_coll.add_trace(go.Scatter(
            x=monthly_kpi["due_month"], y=monthly_kpi["collection_eff"],
            mode="lines", name="Collection Efficiency",
            line=dict(color=GREEN, width=2.5),
            fill="tozeroy", fillcolor=f"rgba(0,200,150,0.1)",
            hovertemplate="<b>%{x}</b><br>Collection Efficiency: %{y:.2f}%<extra></extra>",
        ))
        fig_coll.add_hline(y=99.5, line_dash="dash", line_color=AMBER,
                           annotation_text="⚠ Amber Threshold 99.5%", annotation_font_color=AMBER)
        fig_coll.add_hline(y=98.0, line_dash="dash", line_color=RED,
                           annotation_text="🔴 Red Threshold 98.0%", annotation_font_color=RED)
        apply_layout(fig_coll, "Monthly Collection Efficiency (%) — Real-Time Portfolio Health", height=320)
        fig_coll.update_yaxes(title_text="Collection Efficiency (%)", title_font=dict(color=SUB))
        st.plotly_chart(fig_coll, use_container_width=True)

    with col2:
        fig_ot = go.Figure()
        fig_ot.add_trace(go.Scatter(
            x=monthly_kpi["due_month"], y=monthly_kpi["on_time_rate"],
            mode="lines", name="On-Time Rate",
            line=dict(color=BLUE, width=2.5),
            fill="tozeroy", fillcolor=f"rgba(59,130,246,0.1)",
            hovertemplate="<b>%{x}</b><br>On-Time Rate: %{y:.2f}%<extra></extra>",
        ))
        fig_ot.add_trace(go.Scatter(
            x=monthly_kpi["due_month"], y=monthly_kpi["missed_rate"],
            mode="lines", name="Missed Rate",
            line=dict(color=RED, width=2, dash="dot"),
            hovertemplate="<b>%{x}</b><br>Missed Rate: %{y:.2f}%<extra></extra>",
        ))
        apply_layout(fig_ot, "Monthly On-Time Payment & Missed Rate — Repayment Quality Trend", height=320)
        fig_ot.update_yaxes(title_text="Rate (%)", title_font=dict(color=SUB))
        st.plotly_chart(fig_ot, use_container_width=True)

    # ── EWS Model Performance ──
    st.markdown('<p class="section-header">EWS Model Performance — Green / Amber / Red Scorecard</p>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)

    # Scorecard buckets
    bucket_meta = {
        "GREEN": (GREEN, "🟢", "P < 0.20", "Low Risk — Standard Monitoring"),
        "AMBER": (AMBER, "🟡", "0.20 ≤ P < 0.45", "Medium Risk — Proactive Outreach"),
        "RED":   (RED,   "🔴", "P ≥ 0.45", "High Risk — Collections Escalation"),
    }
    for col, b in zip([col3, col4, col5], scorecard_data):
        bname  = b["bucket"]
        color, icon, thresh, action = bucket_meta[bname]
        with col:
            st.markdown(f"""
            <div style='background:{CARD}; border:1px solid {color}50; border-top:4px solid {color};
                        border-radius:12px; padding:20px; text-align:center;'>
              <div style='font-size:28px;'>{icon}</div>
              <div style='font-size:20px; font-weight:700; color:{color}; margin:8px 0;'>{bname}</div>
              <div style='font-size:11px; color:{SUB};'>{thresh}</div>
              <hr style='border-color:{BORDER}; margin:12px 0;'>
              <div style='font-size:24px; font-weight:700; color:{TEXT};'>{b["n"]:,}</div>
              <div style='font-size:11px; color:{SUB}; margin-bottom:8px;'>Active Loan-Observations</div>
              <div style='font-size:14px; color:{color}; font-weight:600;'>{b["share_pct"]:.1f}% of Portfolio</div>
              <div style='font-size:13px; color:{TEXT}; margin-top:6px;'>Delinquency Rate: <b style="color:{color};">{b["actual_delinquency_rate"]:.1f}%</b></div>
              <div style='font-size:13px; color:{TEXT};'>Catch Rate: <b>{b["catch_rate"]:.1f}%</b></div>
              <hr style='border-color:{BORDER}; margin:12px 0;'>
              <div style='font-size:11px; color:{SUB}; line-height:1.5;'>{action}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── ROC Curve + Feature Importance ──
    col6, col7 = st.columns([1, 1.2])

    with col6:
        roc_s = roc_data.sample(min(5000, len(roc_data))).sort_values("fpr")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=roc_s["fpr"], y=roc_s["tpr"],
            mode="lines", name=f"LightGBM (AUC={metrics['auc_test']:.3f})",
            line=dict(color=BLUE, width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        ))
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(color=SUB, dash="dash", width=1.5))
        apply_layout(fig_roc, f"ROC Curve — AUC {metrics['auc_test']:.4f} | Gini {metrics['gini_test']:.1f}% | KS {metrics['ks_statistic_pct']:.1f}%", height=360)
        fig_roc.update_xaxes(title_text="False Positive Rate", title_font=dict(color=SUB))
        fig_roc.update_yaxes(title_text="True Positive Rate", title_font=dict(color=SUB))
        st.plotly_chart(fig_roc, use_container_width=True)

    with col7:
        top_fi = feat_imp.nlargest(15, "importance").sort_values("importance", ascending=True)
        cat_colors = {"Origination": BLUE, "Bank Behavioral": GREEN, "Repayment Behavioral": AMBER}
        fig_fi = go.Figure(go.Bar(
            x=top_fi["importance"], y=top_fi["feature"],
            orientation="h",
            marker=dict(color=[cat_colors.get(c, PURPLE) for c in top_fi["category"]]),
            text=[f"{v}" for v in top_fi["importance"]],
            textposition="outside", textfont=dict(color=TEXT, size=10),
            hovertemplate="<b>%{y}</b><br>Importance: %{x}<br>Category: %{customdata}<extra></extra>",
            customdata=top_fi["category"],
        ))
        # Legend annotations
        for i, (cat, color) in enumerate(cat_colors.items()):
            fig_fi.add_annotation(x=0.98, y=0.95-i*0.07, xref="paper", yref="paper",
                                   text=f"■ {cat}", showarrow=False,
                                   font=dict(color=color, size=10), xanchor="right")
        apply_layout(fig_fi, "Top 15 Feature Importances — EWS LightGBM Model", height=360)
        fig_fi.update_xaxes(title_text="MDI Importance Score", title_font=dict(color=SUB))
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Intervention matrix ──
    st.markdown('<p class="section-header">Intervention Action Matrix — Operations Guide</p>', unsafe_allow_html=True)
    inter_df = pd.DataFrame([
        {"Bucket": "🟢 GREEN", "Probability": "P < 0.20", "Primary Action": "Monthly review; no outreach", "Secondary Action": "Auto-debit confirmation + digital statement", "Contact Mode": "Digital only", "Escalation Rule": "Miss 1 payment → AMBER"},
        {"Bucket": "🟡 AMBER", "Probability": "0.20 ≤ P < 0.45", "Primary Action": "WhatsApp/SMS reminder 7 days before due", "Secondary Action": "IVR call 3 days before due; offer EMI date change", "Contact Mode": "Digital + 1 phone call", "Escalation Rule": "Miss 1 payment → RED immediately"},
        {"Bucket": "🔴 RED", "Probability": "P ≥ 0.45", "Primary Action": "Collections call within 24 hours of flag", "Secondary Action": "Offer restructuring/settlement; legal notice prep", "Contact Mode": "Direct call + field visit if ticket >₹50K", "Escalation Rule": "DPD >30 → NPA provisioning + recovery team"},
    ])
    st.dataframe(inter_df, use_container_width=True, hide_index=True)

    # ── ECL Summary ──
    st.markdown('<p class="section-header">ECL (Expected Credit Loss) Dashboard — Ind-AS 109</p>', unsafe_allow_html=True)
    ecl_c1, ecl_c2, ecl_c3, ecl_c4 = st.columns(4)
    with ecl_c1: st.metric("PD (Portfolio)", f"{default_rate:.2f}%", "95% CI ±0.2pp")
    with ecl_c2: st.metric("Mean LGD (₹)", "₹28,647", "Median ₹19,914")
    with ecl_c3: st.metric("Recovery Rate", "47.3%", "Std Dev 15.75%")
    with ecl_c4: st.metric("ECL / Loan (Est.)", "₹4,858", "PD × EAD × (1-RR)")

    # Tier-level ECL bar
    tier_ecl = pd.DataFrame({
        "Risk Tier": TIER_ORDER,
        "PD (%)":    [3.1, 11.0, 20.2, 15.4],
        "Avg EAD (₹K)": [43.9, 31.5, 19.0, 15.3],
        "LGD Rate (%)": [52.7]*4,
    })
    tier_ecl["ECL (₹K)"] = tier_ecl["PD (%)"]/100 * tier_ecl["Avg EAD (₹K)"] * tier_ecl["LGD Rate (%)"]/100

    fig_ecl = go.Figure(go.Bar(
        x=tier_ecl["Risk Tier"], y=tier_ecl["ECL (₹K)"],
        marker=dict(color=[RISK_COLORS[t] for t in TIER_ORDER],
                    line=dict(color=BG, width=1)),
        text=[f"₹{v:.2f}K" for v in tier_ecl["ECL (₹K)"]],
        textposition="outside", textfont=dict(color=TEXT, size=12),
        hovertemplate="<b>%{x}</b><br>ECL: ₹%{y:.2f}K per loan<extra></extra>",
    ))
    apply_layout(fig_ecl, "Expected Credit Loss (₹K/Loan) by Risk Tier — ECL = PD × EAD × LGD", height=280)
    fig_ecl.update_yaxes(title_text="ECL (₹K per Loan)", title_font=dict(color=SUB))
    st.plotly_chart(fig_ecl, use_container_width=True)

    # ── Three Imperatives footer ──
    st.markdown("---")
    st.markdown('<p class="section-header">Three Strategic Imperatives</p>', unsafe_allow_html=True)
    si1, si2, si3 = st.columns(3)
    imperatives = [
        (si1, "1. Channel Rebalancing", BLUE, "Shift 10% of volume from Paid-Digital to Referral/Organic → saves ~₹1,890 CAC/loan AND reduces portfolio PD by 2-3pp. Net portfolio ROE improvement estimated 150-200 bps."),
        (si2, "2. Behavioral EWS Deployment", GREEN, "Deploy the 3-month pre-default signal window (CFC + Balance Volatility + FOIR). Model currently catches 35% of delinquencies before DPD-30. Target AUC >0.70 with full behavioral engineering."),
        (si3, "3. Subprime Repricing + NTC Ladder", AMBER, "Raise Grade D APR by 150-200 bps to correct structural under-pricing. Launch BNPL → Consumer-Durable → Personal Loan ladder to graduate NTC borrowers safely."),
    ]
    for col, title, color, text in imperatives:
        with col:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,{color}18,{color}08);
                        border:1px solid {color}40; border-top:3px solid {color};
                        border-radius:10px; padding:18px;'>
              <div style='font-size:13px; font-weight:700; color:{color}; margin-bottom:8px;'>{title}</div>
              <div style='font-size:12px; color:{TEXT}; line-height:1.6;'>{text}</div>
            </div>
            """, unsafe_allow_html=True)
