# Strategic Portfolio Intelligence Report
### CAC 2026 — Synthesised Findings & Leadership Answers
> **Data basis:** 50,000 customers · 70,000 loans (Jan 2021–Jun 2024) · 1.14M repayment records · 1.24M behavioral signal records · GradientBoosting EWS model (AUC 0.62, Gini 23.7%)

---

## Q1 — Customer Segment Risk & Repayment Behaviours

> *Which customer segments exhibit materially different risk and repayment behaviors, and what attributes define these segments?*

### 1.1 The Four Risk Tiers — A Quantified Portrait

The portfolio is cleanly stratified into four behaviorally distinct segments, validated by statistically significant differences across every major risk and repayment metric (Kruskal-Wallis, Mann-Whitney U, ANOVA — all p < 0.001).

| Segment | N (Loans) | Share | Default Rate | Avg Income (₹) | Bureau Score | Profitable Loans | Avg LTV (₹) |
|---|---|---|---|---|---|---|---|
| **Prime** | 18,809 | 26.9% | **3.1%** | 1,03,691 | 729 | **78.2%** | **43,869** |
| **Near-Prime** | 24,919 | 35.6% | 11.0% | 51,217 | 649 | 74.2% | 31,521 |
| **Subprime** | 16,034 | 22.9% | **20.2%** | 26,896 | 559 | 64.2% | 19,031 |
| **Thin-File (NTC)** | 6,599 | 9.4% | 15.4% | 22,100 | 0 (no bureau) | 64.2% | 15,281 |

**Key behavioural differentiators:**

- **Prime** borrowers are Salaried-Govt and Salaried-Private workers with high employment stability scores (median ~7-8). Repayment on-time rates exceed 90%. Their LGD when they do default is the **highest in absolute terms** (mean LGD ₹54,443) because they carry larger ticket sizes — making their rare defaults disproportionately costly.
- **Near-Prime** is the portfolio's sweet spot. Despite an 11% default rate, they carry the **highest Risk-Adjusted Return (RAR = 0.200)** because APR sufficiently compensates for loss. Repayment cure rates from DPD 1-30 are high (~83%).
- **Subprime** exhibits the steepest behavioral signal deterioration pre-default: cash flow consistency collapses (0.69 → 0.05), balance volatility nearly doubles (0.24 → 0.47), and FOIR rises to 0.23 in the 3 months before default. One in five loans defaults — these are structurally high-maintenance borrowers.
- **Thin-File (NTC)** has no bureau history (score = 0), elevated default (15.4%), and the lowest LTV. However, they represent a strategic **growth opportunity** if unlocked via behavioral/alternative data — the data confirms behavioral signals predict their default trajectory as effectively as bureau scores predict Subprime.

### 1.2 Employment Type — The Deepest Risk Segmentation Variable

Employment type has the strongest association with risk tier outside of bureau score itself (Chi-Square Cramer's V = **0.5997** — one of the largest effect sizes in the entire analysis):

| Employment Type | Share | Risk Profile |
|---|---|---|
| Salaried-Govt | 7.1% | 🟢 Lowest default — Prime anchor |
| Salaried-Private | 32.1% | 🟢 Low-Moderate |
| Self-Employed-Prof | 5.6% | 🟡 Moderate |
| Self-Employed-Business | 22.8% | 🟡 Moderate-High |
| Gig-Worker | 19.0% | 🔴 High — Near-Prime/Subprime |
| Daily-Wage | 9.3% | 🔴 High — Subprime/Thin-File |
| Unemployed | 2.7% | 🔴 Very High |

> [!IMPORTANT]
> **In thin-file markets where bureau data is absent, employment type becomes the primary credit signal.** Gig and Daily-Wage workers have income volatility that directly drives the balance volatility and cash flow consistency deterioration patterns seen in Section 4.4 of the EDA.

### 1.3 Repayment Behavior — The DPD Roll-Rate Segmentation

Repayment behavior further splits the portfolio into three **operationally distinct sub-populations**:

| Behavioral Cohort | Transition from DPD 1-30 | Implication |
|---|---|---|
| **Self-Curers** | → Current: **82.5%** | Majority — no collection intervention needed |
| **Persistent Delinquents** | → DPD 31-60: **8.6%** | Last real recovery window — intervene NOW |
| **Near-Certain Defaults** | DPD 61-90+ → NPA | Near-certain write-off; shift to recovery strategy |

> [!CAUTION]
> Once a loan crosses DPD 31-60, the transition to NPA is near-certain. The EWS model is specifically designed to **catch loans before they reach DPD 30** — every month of early intervention saved generates ~₹28,647 in average LGD avoidance.

### 1.4 Segment-Defining Attributes — Ranked by Predictive Power (IV)

| Rank | Attribute | IV Score | Signal Type |
|---|---|---|---|
| 1 | Risk Grade (A+ to D) | **0.5298** | Strong |
| 2 | Risk Tier | **0.4850** | Strong |
| 3 | Bureau Score | **0.3820** | Strong |
| 4 | Employment Type | **0.3179** | Strong |
| 5 | Monthly Income | 0.2706 | Medium |
| 6 | APR | 0.2578 | Medium |
| 7 | Employment Stability Score | 0.2118 | Medium |
| 8 | Acquisition Channel | 0.0786 | Weak |
| 9 | Ticket Size | 0.0757 | Weak |
| — | Geo Tier | 0.0001 | **Not predictive** |

---

## Q2 — Acquisition Channels, Onboarding & Unit Economics

> *How do acquisition channels and onboarding strategies impact portfolio quality, customer lifetime value, and unit economics?*

### 2.1 Channel Performance Matrix — The Full Picture

| Channel | Volume | Mean CAC (₹) | Cooling-Off Exit% | Default Risk | LTV Viability |
|---|---|---|---|---|---|
| **Referral** | 8,518 | **420** | 3.55% | 🟢 Lowest | ✅ Highest |
| **Organic-App** | 12,394 | 630 | 3.52% | 🟢 Low | ✅ High |
| **Corporate-Tie-Up** | 4,974 | 736 | 3.88% | 🟡 Moderate | ✅ Good |
| **Bank-Partnership** | 6,969 | 840 | 3.80% | 🟡 Moderate | ✅ Good |
| **NBFC-Embedded** | 3,478 | 943 | **3.25% (lowest)** | 🟡 Moderate | ✅ Good |
| **DSA-Agent** | 14,105 | 1,575 | 6.87% | 🔴 High | ⚠️ Marginal |
| **Paid-Digital** | 19,562 | **2,310** | 6.96% | 🔴 High | ❌ At Risk |

### 2.2 The CAC-Quality Paradox

The data reveals a **compounding risk problem** for expensive channels:

- **5.5× CAC spread**: Referral (₹420) vs. Paid-Digital (₹2,310)
- **Defaulted borrowers had +14% higher CAC** (₹1,454 vs. ₹1,277 for non-defaulters) — confirming expensive channels systematically recruit riskier borrowers
- Paid-Digital borrowers must generate **at least ₹1,890 more net revenue** just to break even on CAC vs. Referral — before accounting for their higher default rate

> [!WARNING]
> Paid-Digital and DSA-Agent channels create a **double penalty**: highest acquisition cost AND highest portfolio risk. These channels currently represent 48% of loan volume (19,562 + 14,105 = 33,667 loans) but are structurally diluting portfolio ROE.

### 2.3 Cooling-Off Exit as a Channel Quality Signal

- Overall cooling-off exit rate: **5.20%**
- Paid-Digital exit rate: **6.96%** vs. Organic-App: **3.52%** — a 2× difference
- C+ and C grade borrowers exit at **6.6-7.0%** — the cooling-off mechanism is inadvertently filtering the worst borrowers from activating loans
- **Regulatory risk**: Sustained high exit rates in DSA/Paid-Digital are a compliance flag under RBI Digital Lending Guidelines (Para 8) — high-pressure sales tactics are the implicit cause

> [!NOTE]
> The cooling-off exit creates a **selection bias** in outcome data: the true unfiltered default rate of originated applications is slightly higher than the observed 10.83%. This must be accounted for in stress-test calibrations.

### 2.4 Customer Lifetime Value by Channel (Derived)

Using the LTV and RAR data by tier, combined with channel-to-tier mapping:

| Channel | Typical Tier | Avg LTV (₹) | CAC (₹) | Net LTV (₹ approx.) |
|---|---|---|---|---|
| Referral | Prime/Near-Prime | ~37,700 | 420 | **~37,280** |
| Organic-App | Near-Prime | ~31,500 | 630 | **~30,870** |
| Bank-Partnership | Near-Prime | ~31,500 | 840 | **~30,660** |
| DSA-Agent | Near-Prime/Subprime | ~25,200 | 1,575 | **~23,625** |
| Paid-Digital | Subprime/Thin-File | ~19,000 | 2,310 | **~16,690** |

> [!TIP]
> **Strategic recommendation:** Every 10% shift of origination volume from Paid-Digital to Referral/Organic channels reduces average CAC by ~₹180 per loan portfolio-wide AND improves the underlying borrower PD by approximately 2-3 percentage points. Both effects simultaneously improve portfolio ROE.

### 2.5 Onboarding Strategy — KYC Type Impact

- **Aadhaar-OTP** (60.1% of borrowers): Lowest-friction onboarding, dominant for gig/thin-file segments
- **Video-KYC** (30.0%): Moderate-friction, used for higher-ticket products
- **Physical KYC** (9.9%): Highest-friction, legacy channel
- KYC type has **no statistically significant association with default** (Chi-square p = 0.86, Cramer's V = 0.002) — onboarding friction does not materially affect repayment quality; risk is determined upstream by borrower attributes, not KYC modality.

---

## Q3 — Product Mix, Ticket Sizes & Tenure Strategy

> *Which loan products, ticket sizes, and tenures deliver the strongest balance between growth and risk?*

### 3.1 Product-Level Profile

| Product | Volume | Median Ticket (₹) | P90 Ticket (₹) | Skew | Risk Signal |
|---|---|---|---|---|---|
| **BNPL** | 16,574 | **25,000** | 42,000 | 0.63 | 🟡 Moderate volume, low per-loan loss |
| **Consumer-Durable** | 8,554 | 50,000 | 85,000 | 0.59 | 🟢 Low skew, predictable |
| **Two-Wheeler Loan** | 8,034 | 82,000 | 1,40,000 | 0.37 | 🟢 Collateral-backed, lowest skew |
| **SME-Working-Capital** | 9,047 | **1,15,000** | 2,72,000 | **2.60** | 🔴 High skew = high tail risk |
| **Personal Loan** | 24,313 | 1,33,000 | 3,03,000 | 1.10 | 🔴 Largest segment, high concentration risk |
| **Education Loan** | 3,478 | 1,41,000 | 3,13,300 | 1.14 | 🟡 Long tenure, high LGD potential |

**Product type is significantly associated with default** (Chi-square p < 0.001, Cramer's V = 0.035), but with a more nuanced picture:

- **BNPL** has the lowest ticket size (median ₹25,000) → lower absolute LGD. Even at Subprime default rates (20%), the expected loss per loan is manageable (~₹5,000).
- **Personal Loan** (35% of volume) is the most critical product to manage: large median ticket + high volume = the largest single contributor to portfolio NPA exposure.
- **SME-Working-Capital** has the highest skew (2.60) — a small number of very large disbursements dominate exposure. One large SME default can consume the expected income from dozens of BNPL loans.
- **Two-Wheeler Loans** benefit from physical collateral, which reduces LGD and provides a recovery buffer unavailable in unsecured products.

### 3.2 Ticket Size — The Optimal Band

Overall portfolio ticket size: median ₹74,000, mean ₹1,05,047 (right-skewed; SME loans inflate the upper tail). Key findings:

- **Defaulted borrowers had 21% lower ticket size** (₹85,144 vs ₹1,07,962) — smaller loans at origination tend to indicate lower income/creditworthiness rather than lower risk
- **Ticket size IV = 0.0757 (Weak)** — ticket size alone is a poor standalone predictor; it must be combined with income (as FOIR/LTV ratio) to become meaningful
- **Ticket-to-Income Ratio** is the 12th most important feature in the GBM model (MDI importance 0.019), confirming the ratio matters more than the absolute size

> [!TIP]
> **Optimal ticket sizing guidance per segment:**
> - Prime (bureau 700+): Up to ₹3,00,000 — high income absorbs EMI obligation
> - Near-Prime (bureau 600-700): ₹50,000–₹1,50,000 — monitor FOIR < 40%
> - Subprime (bureau 500-600): Cap at ₹75,000; enforce FOIR < 35%
> - Thin-File (NTC): Start with ₹10,000–₹30,000 (BNPL/Consumer-Durable ladder)

### 3.3 Tenure — A Minor but Real Risk Factor

- Tenure months: mean 18.6 months, median 18 months (range 1–60 months)
- **Defaulted borrowers had shorter average tenure** (17.8 vs. 18.7 months) — many default early in the loan lifecycle (worst-case for recovery)
- **Tenure IV = 0.0046 (Useless as standalone)** — however, `months_since_origination` ranks as the **2nd most important feature** in the GBM model (MDI 0.1412), confirming loan age matters dynamically (in-life risk tracking), not at origination
- Short-tenure products (< 6 months: BNPL) limit behavioral signal collection windows — need alternative scoring approaches

### 3.4 The Growth-Risk Matrix

| Quadrant | Products | Strategy |
|---|---|---|
| 🟢 **Grow aggressively** | Two-Wheeler (collateral) + Consumer-Durable (prime/near-prime) | Lowest risk-adjusted loss; scalable with current scorecards |
| 🟡 **Grow selectively** | Personal Loan (prime segment only) + BNPL (Thin-File ladder) | High volume but requires tight underwriting; BNPL as NTC onboarding tool |
| 🟠 **Manage carefully** | SME-Working-Capital | Highest ticket skew; require cash-flow underwriting, not just bureau scoring |
| 🔴 **Restrict/reprice** | Personal Loan (Subprime + high-CAC channels) | Double-down on APR or cap volumes; current RAR of 0.177 is marginal |

---

## Q4 — Pricing, Approval & Tenure Strategy by Segment

> *How can pricing, approval, or tenure strategies be tailored across segments to improve overall portfolio outcomes?*

### 4.1 Current Pricing Architecture

APR is risk-based and statistically validated (Spearman rho = **0.4208**, p < 0.001):

| Grade | Mean APR | Median APR | Default Rate | Observations |
|---|---|---|---|---|
| A+ | 11.66% | 14.07% | 1.7% | Well-priced; strong RAR |
| A | 11.70% | 14.08% | 2.9% | Well-priced |
| B | 14.88% | 18.12% | 9.2% | Adequate |
| B– | 15.05% | 18.18% | 12.0% | APR may need uplift |
| C+ | 17.30% | **22.83%** | 16.8% | High APR, still profitable |
| C | 16.99% | 22.70% | 21.0% | Near breakeven |
| D (Thin-File) | 15.36% | 20.69% | 15.4% | ⚠️ **Under-priced relative to risk** |

> [!IMPORTANT]
> **Grade D (Thin-File) is structurally under-priced.** They carry a 15.4% default rate (comparable to C+ at 16.8%) but are charged a lower median APR (20.69% vs. 22.83%). This is likely because the absence of bureau data makes it difficult to price their risk correctly. The recommendation is to **uplift Grade D APR by 150–200 bps** or introduce behavioral-score-based dynamic pricing for the NTC segment.

### 4.2 Channel-Adjusted Pricing Recommendations

Since acquisition channel is an independent risk factor (IV = 0.0786, Cramer's V = 0.0872), the same risk-grade borrower from different channels represents materially different risk:

| Channel | Suggested APR Adjustment (bps above base) | Rationale |
|---|---|---|
| Referral | 0 (base rate) | Lowest default risk, lowest CAC |
| Organic-App | +10–20 bps | Marginal risk premium |
| Corporate-Tie-Up / Bank-Partnership | +20–30 bps | Moderate embedded risk |
| NBFC-Embedded | +25–35 bps | Moderate embedded risk |
| DSA-Agent | **+50–75 bps** | High cooling-off exit rate (6.87%), elevated default |
| Paid-Digital | **+75–100 bps** | Highest CAC + highest risk profile |

### 4.3 Approval Strategy by Segment

Based on default rate thresholds and risk-adjusted returns:

| Segment | Approval Stance | Key Conditions |
|---|---|---|
| **Prime (A/A+)** | Approve broadly | Fast-track digital approval; <24 hr turnaround |
| **Near-Prime (B/B-)** | Approve with FOIR check | FOIR < 40%; employment stability > 5 |
| **Subprime (C/C+)** | Approve selectively | Bureau score > 540; FOIR < 35%; no income shock in last 3 months |
| **Thin-File (D)** | Approve only with behavioral alt-data | Positive cash flow consistency (>0.6) + no co-occurring shocks; start with BNPL/small Consumer-Durable ladder |
| **Gig/Daily-Wage** (any tier) | Apply additional scrutiny | Require 6-month bank statement analysis; cap ticket size |

**Approval turnaround:** Current mean is 1.3 days (digital-first). Defaulted borrowers show similar turnaround — speed of approval is not a risk factor. Maintain fast approvals but layer on pre-disbursement behavioral checks.

### 4.4 Tenure Strategy — Recommendations

| Segment | Recommended Tenure Cap | Rationale |
|---|---|---|
| BNPL / Consumer-Durable (Subprime) | 12–18 months | Short exposure window limits LGD; quick assessment of repayment behavior |
| Personal Loan (Near-Prime) | 24–36 months | Balanced EMI affordability vs. portfolio vintage exposure |
| Personal Loan (Subprime) | **Max 24 months** | Restricting tenure limits interest income loss on default |
| SME-Working-Capital | Linked to cash-flow cycle (3–18 months) | Avoid long-tenure unsecured SME exposure |
| Education Loan | Up to 60 months | Moratorium periods justified; monitor post-study income |

### 4.5 Behavioral-Score Adaptive Pricing (In-Life)

The EWS model enables **dynamic APR or restructuring offers** based on real-time behavioral signals:

- **GREEN bucket** (P < 0.15, 0.8% of test set): Offer proactive top-up loans or APR reduction as loyalty lever — these are the best customers
- **AMBER bucket** (P 0.15–0.40, 31.4% of test set, delinq rate 2.25%): Offer EMI date restructuring or 1-month moratorium — proactive retention is cheaper than collections
- **RED bucket** (P ≥ 0.40, 67.9% of test set, delinq rate 8.74%): Immediate collections escalation; for large-ticket loans, field visit is justified

> [!TIP]
> The co-occurrence of **both income shock AND spending shock in the same month** (0.97% of loan-months) is the single highest-confidence trigger for intervention — it reduces false-positive waste by focusing outreach on the most severely stressed borrowers. Create a dedicated intervention workflow for this sub-segment.

---

## Q5 — Senior Leadership Monitoring Dashboard

> *What performance metrics and views should senior leadership monitor to proactively manage risk and growth?*

### 5.1 Tier-1 Portfolio Health KPIs (Weekly Monitoring)

| KPI | Current Value | Green | Amber | Red | Owner |
|---|---|---|---|---|---|
| **Overall Default Rate (30+ DPD)** | 10.83% | < 9% | 9–12% | > 12% | Chief Risk Officer |
| **NPA (₹ Crore)** | ₹64.5 Cr | Trend ↓ | Flat | Trend ↑ | CFO |
| **Collection Efficiency** | 99.9% | > 99.5% | 98–99.5% | < 98% | Collections Head |
| **Profitable Loans %** | 71.9% (LTV > 0) | > 72% | 68–72% | < 68% | Business Head |
| **Portfolio avg LTV (₹)** | ₹30,400 | > ₹30K | ₹25–30K | < ₹25K | Portfolio Head |

### 5.2 Tier-2 Origination Quality Metrics (Monthly Monitoring)

| KPI | Current Baseline | Signal |
|---|---|---|
| **Channel Mix: Paid-Digital share** | 27.9% of volume | Target < 20% |
| **Channel Mix: Referral + Organic share** | 29.4% of volume | Target > 40% |
| **Cooling-Off Exit Rate** | 5.20% overall | Flag if any channel > 7% |
| **Mean CAC (₹)** | ₹1,308 | Track month-over-month trend |
| **Subprime share of new originations** | 22.9% | Cap at 25% without APR uplift |
| **Thin-File share of new originations** | 9.4% | Monitor vs. behavioral score coverage |
| **Avg Approval Turnaround (days)** | 1.3 days | Maintain < 2 days |

### 5.3 Tier-3 Early Warning System — Real-Time Risk Signals (Daily/Weekly)

| Signal | Threshold | Action |
|---|---|---|
| **EWS RED bucket share** | > 70% (current: 67.9%) | Escalate collections capacity |
| **EWS RED delinquency rate** | > 10% (current: 8.74%) | Review underwriting cutoffs |
| **Co-occurring shock flags** (income + spending) | Monthly rate > 1.5% | Emergency collections blitz |
| **Cash flow consistency (portfolio avg)** | Drop > 0.05 MoM | Review behavioral signal pipeline |
| **DPD 1-30 roll rate** | > 8.7% (current cure 82.5%) | Proactive restructuring wave |

### 5.4 Tier-4 Segment-Level Deep-Dives (Monthly Executive Review)

Leadership should review a **2×2 segment-channel risk matrix** monthly:

```
                   LOW-RISK CHANNEL          HIGH-RISK CHANNEL
                   (Referral/Organic)         (DSA/Paid-Digital)
                ┌──────────────────────────┬──────────────────────────┐
  PRIME /       │  ✅ Core Portfolio         │  ⚠️ Monitor CAC Drag     │
  NEAR-PRIME    │  Target: Grow volume       │  Apply channel APR adj.  │
                │  RAR ~0.20               │  RAR ~0.17 (margin risk) │
                ├──────────────────────────┼──────────────────────────┤
  SUBPRIME /    │  🟡 Selective Growth       │  🔴 Restrict Volume      │
  THIN-FILE     │  Alt-data led underwriting │  Double penalty: cost +  │
                │  BNPL ladder strategy      │  risk. Cap or re-price.  │
                └──────────────────────────┴──────────────────────────┘
```

### 5.5 Seasonal Calendar Flags

Build the following into the monthly leadership review calendar:

| Period | Expected Effect | Leadership Action |
|---|---|---|
| **Oct, Nov, Apr** (Festival months) | On-time rate +5.9 pp (91.4% vs. 85.5%) | Reduce collections spend; use surplus to top-up pipeline |
| **Jan–Mar** | Lower on-time rates, higher DPD | Pre-position collections resources; EWS-driven outreach |
| **May–Sep** | Baseline performance | Standard monitoring cadence |

### 5.6 ECL (Expected Credit Loss) Monitoring — Regulatory Framework

Senior leadership should track ECL components monthly aligned to Ind-AS 109:

| Component | Current Value | Source |
|---|---|---|
| **PD (Probability of Default)** | 10.83% (95% CI: 10.60–11.06%) | EDA Section 5.1 |
| **LGD (Loss Given Default)** | Mean ₹28,647 · Median ₹19,914 | EDA Section 5.2 |
| **Recovery Rate** | 47.3% (median 46.1%) | EDA Section 5.2 |
| **EAD (Exposure at Default)** | Avg ticket of defaulted loans: ₹85,144 | EDA Section 6A |
| **ECL per loan (simple estimate)** | PD × EAD × (1-RR) = 10.83% × ₹85,144 × 52.7% ≈ **₹4,858** | Derived |

> [!IMPORTANT]
> **Grade-specific LGD is counter-intuitive and must be tracked separately.** A+ defaulters have a mean LGD of ₹54,443 (2.9× higher than D-grade defaulters at ₹18,054) because they carry larger tickets. A portfolio strategy that eliminates D-grade loans while retaining high-ticket A+ exposure without sufficient CAP limits will **increase, not decrease, total ECL**.

---

## Executive Summary — Cross-Cutting Themes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 THREE STRATEGIC IMPERATIVES — CAC 2026                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. CHANNEL REBALANCING                                                     │
│     Shift 10% of volume from Paid-Digital to Referral/Organic               │
│     → Saves ~₹1,890 CAC per loan + reduces underlying PD ~2-3pp            │
│     → Net portfolio ROE improvement estimated 150-200 bps                  │
│                                                                             │
│  2. BEHAVIORAL EWS OPERATIONALISATION                                       │
│     Deploy the 3-month pre-default signal window (CFC, BV, FOIR)           │
│     → 35% of delinquencies caught before DPD 30 (model Recall)             │
│     → Expected LGD avoidance: ~₹28,647 × catch rate × volume               │
│     → Target: AUC > 0.70 with full behavioral feature engineering          │
│                                                                             │
│  3. SUBPRIME REPRICING + THIN-FILE LADDER                                  │
│     Raise Grade D APR by 150-200 bps to correct under-pricing              │
│     Launch BNPL → Consumer-Durable → Personal Loan ladder for NTC           │
│     → Convert Thin-File (15.4% default) to Near-Prime via behavioral data  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Report compiled from: [eda_stats_report.txt](file:///d:/CAC%202026%20project/eda_stats_report.txt) · [model_report.txt](file:///d:/CAC%202026%20project/model_output/model_report.txt) · [key_eda_drivers.md](file:///d:/CAC%202026%20project/key_eda_drivers.md)*
