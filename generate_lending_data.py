"""
=============================================================
  DIGITAL LENDING — SYNTHETIC DATASET GENERATOR
  Covers all 7 data areas from the project brief:
  Customer Profile, Loan & Product, Repayment Behavior,
  Behavioral Signals, Acquisition, Time Dimension, Outcomes
=============================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────
# 0. MASTER CONFIG
# ─────────────────────────────────────────────
N_CUSTOMERS   = 50_000
N_LOANS       = 70_000          # some customers have repeat loans
START_DATE    = datetime(2021, 1, 1)
END_DATE      = datetime(2024, 6, 30)

print("=" * 60)
print("  DIGITAL LENDING — SYNTHETIC DATA GENERATOR")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. GEOGRAPHY SETUP  (India-realistic)
# ─────────────────────────────────────────────
CITIES = {
    # city: (tier, state, avg_income_multiplier)
    "Mumbai":       ("Metro",  "Maharashtra",  1.6),
    "Delhi":        ("Metro",  "Delhi",        1.5),
    "Bengaluru":    ("Metro",  "Karnataka",    1.55),
    "Hyderabad":    ("Metro",  "Telangana",    1.4),
    "Chennai":      ("Metro",  "Tamil Nadu",   1.35),
    "Pune":         ("Tier-2", "Maharashtra",  1.2),
    "Ahmedabad":    ("Tier-2", "Gujarat",      1.15),
    "Jaipur":       ("Tier-2", "Rajasthan",    1.0),
    "Lucknow":      ("Tier-2", "Uttar Pradesh",0.95),
    "Bhopal":       ("Tier-2", "Madhya Pradesh",0.9),
    "Nagpur":       ("Tier-2", "Maharashtra",  0.95),
    "Surat":        ("Tier-2", "Gujarat",      1.1),
    "Patna":        ("Tier-3", "Bihar",        0.75),
    "Agra":         ("Tier-3", "Uttar Pradesh",0.78),
    "Nashik":       ("Tier-3", "Maharashtra",  0.85),
    "Varanasi":     ("Tier-3", "Uttar Pradesh",0.72),
    "Rajkot":       ("Tier-3", "Gujarat",      0.88),
    "Indore":       ("Tier-3", "Madhya Pradesh",0.82),
    "Coimbatore":   ("Tier-3", "Tamil Nadu",   0.90),
    "Visakhapatnam":("Tier-3", "Andhra Pradesh",0.85),
}

CITY_NAMES   = list(CITIES.keys())
# Realistic city weight: metros ~30%, tier2 ~40%, tier3 ~30%
CITY_WEIGHTS = []
for c in CITY_NAMES:
    tier = CITIES[c][0]
    if tier == "Metro":   CITY_WEIGHTS.append(6)
    elif tier == "Tier-2":CITY_WEIGHTS.append(4)
    else:                  CITY_WEIGHTS.append(2.5)
CITY_WEIGHTS = np.array(CITY_WEIGHTS) / sum(CITY_WEIGHTS)

# ─────────────────────────────────────────────
# 2. RISK TIER DEFINITIONS
# ─────────────────────────────────────────────
RISK_TIERS = {
    # tier: (weight, base_PD, income_mean, income_std, bureau_score_mean, bureau_score_std)
    "Prime":      (0.28, 0.030, 85_000,  25_000, 730, 40),
    "Near-Prime": (0.37, 0.085, 42_000,  15_000, 650, 45),
    "Subprime":   (0.25, 0.180, 22_000,  10_000, 560, 55),
    "Thin-File":  (0.10, 0.130, 18_000,   8_000,   0,  0),  # no bureau score
}

# ─────────────────────────────────────────────
# 3. GENERATE CUSTOMERS
# ─────────────────────────────────────────────
print("\n[1/6] Generating customer profiles...")

def generate_customers(n):
    np.random.seed(42)

    # Risk tier assignment
    tier_names   = list(RISK_TIERS.keys())
    tier_weights = [RISK_TIERS[t][0] for t in tier_names]
    risk_tier    = np.random.choice(tier_names, size=n, p=tier_weights)

    # Geography
    city_arr  = np.random.choice(CITY_NAMES, size=n, p=CITY_WEIGHTS)
    tier_arr  = np.array([CITIES[c][0] for c in city_arr])
    state_arr = np.array([CITIES[c][1] for c in city_arr])
    inc_mult  = np.array([CITIES[c][2] for c in city_arr])

    # Age  (18–58, skewed toward 25–40)
    age = np.clip(np.random.normal(32, 8, n).astype(int), 18, 58)

    # Gender
    gender = np.random.choice(["Male", "Female", "Other"],
                               size=n, p=[0.63, 0.36, 0.01])

    # Employment type  (varies by risk tier)
    emp_map = {
        "Prime":      ["Salaried-Private", "Salaried-Govt", "Self-Employed-Prof"],
        "Near-Prime": ["Salaried-Private", "Self-Employed-Business", "Gig-Worker"],
        "Subprime":   ["Gig-Worker", "Self-Employed-Business", "Daily-Wage", "Unemployed"],
        "Thin-File":  ["Gig-Worker", "Daily-Wage", "Self-Employed-Business", "Student"],
    }
    emp_prob = {
        "Prime":      [0.55, 0.25, 0.20],
        "Near-Prime": [0.45, 0.35, 0.20],
        "Subprime":   [0.35, 0.30, 0.25, 0.10],
        "Thin-File":  [0.30, 0.30, 0.25, 0.15],
    }
    employment = np.array([
        np.random.choice(emp_map[rt], p=emp_prob[rt])
        for rt in risk_tier
    ])

    # Employment stability score  1–10
    emp_stability_base = {
        "Salaried-Govt": 9, "Salaried-Private": 7,
        "Self-Employed-Prof": 6, "Self-Employed-Business": 5,
        "Gig-Worker": 4, "Daily-Wage": 3,
        "Student": 3, "Unemployed": 1,
    }
    emp_stability = np.array([
        np.clip(emp_stability_base[e] + np.random.randint(-1, 2), 1, 10)
        for e in employment
    ])

    # Monthly income (log-normal, city-adjusted)
    income_base = np.array([RISK_TIERS[rt][2] for rt in risk_tier], dtype=float)
    income_sd   = np.array([RISK_TIERS[rt][3] for rt in risk_tier], dtype=float)
    monthly_income = np.clip(
        np.random.lognormal(np.log(income_base * inc_mult), 0.35, n),
        5_000, 5_00_000
    ).astype(int)

    # Bureau score
    bureau_mean = np.array([RISK_TIERS[rt][4] for rt in risk_tier], dtype=float)
    bureau_std  = np.array([RISK_TIERS[rt][5] for rt in risk_tier], dtype=float)
    bureau_score = np.where(
        risk_tier == "Thin-File",
        0,  # no score
        np.clip(np.random.normal(bureau_mean, bureau_std, n).astype(int), 300, 900)
    )

    # New-to-credit flag
    ntc_flag = (bureau_score == 0).astype(int)

    # Existing relationship with lender
    existing_customer = np.random.choice([0, 1], size=n, p=[0.60, 0.40])

    # KYC type
    kyc_type = np.random.choice(
        ["Aadhaar-OTP", "Video-KYC", "Physical"],
        size=n, p=[0.60, 0.30, 0.10]
    )

    df = pd.DataFrame({
        "customer_id":        [f"CUST{str(i).zfill(7)}" for i in range(1, n+1)],
        "risk_tier":          risk_tier,
        "city":               city_arr,
        "geo_tier":           tier_arr,
        "state":              state_arr,
        "age":                age,
        "gender":             gender,
        "employment_type":    employment,
        "employment_stability_score": emp_stability,
        "monthly_income":     monthly_income,
        "bureau_score":       bureau_score.astype(int),
        "new_to_credit_flag": ntc_flag,
        "existing_customer":  existing_customer,
        "kyc_type":           kyc_type,
    })
    return df

customers = generate_customers(N_CUSTOMERS)
print(f"   ✓ {len(customers):,} customers  |  Risk mix: {customers.risk_tier.value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 4. GENERATE LOANS & PRODUCTS
# ─────────────────────────────────────────────
print("\n[2/6] Generating loan & product records...")

PRODUCTS = {
    # name: (min_ticket, max_ticket, tenures_months, base_APR, risk_premium_factor)
    "Personal-Loan":      (20_000,  5_00_000, [12,18,24,36,48],    0.14, 1.0),
    "SME-Working-Capital":(50_000, 20_00_000, [6,12,18,24],        0.16, 1.1),
    "BNPL":               (1_000,   50_000,  [1,2,3,6],            0.00, 0.0),  # 0% promo APR
    "Two-Wheeler-Loan":   (30_000,  1_50_000, [12,24,36],          0.13, 0.9),
    "Consumer-Durable":   (5_000,   1_00_000, [6,12,18,24],        0.15, 1.0),
    "Education-Loan":     (50_000,  5_00_000, [12,24,36,48,60],    0.12, 0.8),
}

ACQUISITION_CHANNELS = {
    # channel: (weight, base_CAC_INR, approval_rate, quality_multiplier_on_PD)
    "Organic-App":        (0.18, 600,   0.55, 0.75),
    "Referral":           (0.12, 400,   0.60, 0.70),
    "Paid-Digital":       (0.28, 2200,  0.38, 1.25),
    "DSA-Agent":          (0.20, 1500,  0.52, 1.30),
    "Bank-Partnership":   (0.10, 800,   0.65, 0.90),
    "Corporate-Tie-Up":   (0.07, 700,   0.70, 0.80),
    "NBFC-Embedded":      (0.05, 900,   0.45, 1.10),
}

RISK_GRADE_MAP = {
    # (risk_tier, bureau_band): origination_risk_grade
    ("Prime",      "750+"):  "A+",
    ("Prime",      "700-749"):"A",
    ("Prime",      "<700"):  "A-",
    ("Near-Prime", "700+"):  "B+",
    ("Near-Prime", "650-699"):"B",
    ("Near-Prime", "<650"):  "B-",
    ("Subprime",   "600+"):  "C+",
    ("Subprime",   "<600"):  "C",
    ("Thin-File",  "NTC"):   "D",
}

def bureau_band(score):
    if score == 0:   return "NTC"
    elif score >= 750: return "750+"
    elif score >= 700: return "700-749"
    elif score >= 650: return "650-699"
    elif score >= 600: return "600+"
    else:              return "<600"

def get_risk_grade(tier, score):
    band = bureau_band(score)
    key  = (tier, band)
    # fallback
    if key not in RISK_GRADE_MAP:
        if tier == "Prime":      return "A-"
        if tier == "Near-Prime": return "B-"
        if tier == "Subprime":   return "C"
        return "D"
    return RISK_GRADE_MAP[key]

def generate_loans(customers, n_loans):
    np.random.seed(99)
    loan_rows = []

    # Assign loans to customers (repeat borrowers for ~28%)
    cust_ids = customers["customer_id"].values
    loan_cust = np.random.choice(cust_ids, size=n_loans, replace=True)

    # Build customer lookup
    cust_lookup = customers.set_index("customer_id")

    ch_names   = list(ACQUISITION_CHANNELS.keys())
    ch_weights = np.array([ACQUISITION_CHANNELS[c][0] for c in ch_names])
    ch_weights /= ch_weights.sum()

    prod_names   = list(PRODUCTS.keys())
    prod_weights = np.array([0.40, 0.15, 0.20, 0.10, 0.10, 0.05])

    date_range = (END_DATE - START_DATE).days

    for i, cust_id in enumerate(loan_cust):
        row   = cust_lookup.loc[cust_id]
        tier  = row["risk_tier"]
        score = int(row["bureau_score"])
        income= int(row["monthly_income"])

        # Product selection  (SME less likely for individuals)
        if row["employment_type"] in ["Salaried-Private","Salaried-Govt","Self-Employed-Prof"]:
            pw = np.array([0.45, 0.05, 0.20, 0.12, 0.12, 0.06])
        elif row["employment_type"] in ["Self-Employed-Business"]:
            pw = np.array([0.20, 0.40, 0.15, 0.10, 0.10, 0.05])
        else:
            pw = np.array([0.30, 0.05, 0.35, 0.12, 0.14, 0.04])
        pw /= pw.sum()

        product_name = np.random.choice(prod_names, p=pw)
        prod         = PRODUCTS[product_name]

        # Ticket size  (capped at 5× monthly income for unsecured)
        max_ticket = min(prod[1], income * 5)
        min_ticket = prod[0]
        if max_ticket < min_ticket:
            max_ticket = min_ticket * 1.5
        ticket_size = int(np.random.lognormal(
            np.log((min_ticket + max_ticket) / 2), 0.4
        ))
        ticket_size = int(np.clip(ticket_size, min_ticket, max_ticket) / 1000) * 1000

        # Tenure
        tenure_months = int(np.random.choice(prod[2]))

        # Pricing — APR
        base_apr = prod[3]
        risk_premium = {
            "Prime": 0.0, "Near-Prime": 0.04,
            "Subprime": 0.09, "Thin-File": 0.07
        }[tier]
        apr_noise  = np.random.uniform(-0.01, 0.02)
        final_apr  = round(base_apr + risk_premium * prod[4] + apr_noise, 4)
        final_apr  = max(0.0, min(final_apr, 0.48))  # cap at 48%

        # Processing fee (0.5%–3% of ticket)
        processing_fee = int(ticket_size * np.random.uniform(0.005, 0.03))

        # Origination risk grade
        risk_grade = get_risk_grade(tier, score)

        # Acquisition channel
        channel      = np.random.choice(ch_names, p=ch_weights)
        ch_data      = ACQUISITION_CHANNELS[channel]
        cac          = int(ch_data[1] * np.random.uniform(0.8, 1.3))
        approval_days= int(np.random.choice([0,1,2,3,5], p=[0.30,0.35,0.20,0.10,0.05]))

        # Origination date
        orig_days = int(np.random.uniform(0, date_range))
        orig_date = START_DATE + timedelta(days=orig_days)

        # Cooling-off exit (product of channel + risk tier)
        cooling_exit_prob = 0.03
        if channel in ["Paid-Digital","DSA-Agent"]: cooling_exit_prob += 0.03
        if tier in ["Subprime","Thin-File"]:        cooling_exit_prob += 0.02
        cooling_off_exit = int(np.random.random() < cooling_exit_prob)

        # EMI calculation (reducing balance)
        if final_apr > 0 and product_name != "BNPL":
            r = final_apr / 12
            emi = round(ticket_size * r * (1 + r)**tenure_months /
                        ((1 + r)**tenure_months - 1), 2)
        else:
            emi = round(ticket_size / tenure_months, 2)

        loan_rows.append({
            "loan_id":           f"LN{str(i).zfill(8)}",
            "customer_id":       cust_id,
            "product_type":      product_name,
            "ticket_size":       ticket_size,
            "tenure_months":     tenure_months,
            "apr":               final_apr,
            "processing_fee":    processing_fee,
            "emi_amount":        emi,
            "origination_date":  orig_date.strftime("%Y-%m-%d"),
            "origination_risk_grade": risk_grade,
            "acquisition_channel":   channel,
            "cost_of_acquisition":   cac,
            "approval_turnaround_days": approval_days,
            "cooling_off_exit":  cooling_off_exit,
        })

    return pd.DataFrame(loan_rows)

loans = generate_loans(customers, N_LOANS)
print(f"   ✓ {len(loans):,} loans  |  Products: {loans.product_type.value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 5. REPAYMENT BEHAVIOR
# ─────────────────────────────────────────────
print("\n[3/6] Generating repayment behavior...")

def generate_repayment(loans, customers):
    np.random.seed(77)

    cust_lookup = customers.set_index("customer_id")[
        ["risk_tier","employment_stability_score","monthly_income"]
    ]

    rows = []

    # PD per risk grade
    grade_pd = {
        "A+": 0.018, "A": 0.030, "A-": 0.045,
        "B+": 0.065, "B": 0.090, "B-": 0.120,
        "C+": 0.160, "C": 0.200,
        "D":  0.150,
    }
    # Channel quality multiplier on PD
    ch_mult = {ch: ACQUISITION_CHANNELS[ch][3] for ch in ACQUISITION_CHANNELS}

    FESTIVAL_MONTHS = {10, 11, 4}   # Oct/Nov (Diwali/Navratri), Apr (harvest)

    for _, loan in loans.iterrows():
        if loan["cooling_off_exit"] == 1:
            continue   # loan never activated

        cust        = cust_lookup.loc[loan["customer_id"]]
        grade       = loan["origination_risk_grade"]
        base_pd     = grade_pd.get(grade, 0.10)
        channel_adj = ch_mult.get(loan["acquisition_channel"], 1.0)
        adj_pd      = min(base_pd * channel_adj, 0.80)

        tenure = int(loan["tenure_months"])
        emi    = float(loan["emi_amount"])
        orig_d = datetime.strptime(loan["origination_date"], "%Y-%m-%d")
        emp_stab = int(cust["employment_stability_score"])

        # Default timing — if this loan defaults, which month?
        will_default = np.random.random() < adj_pd
        if will_default:
            # Defaults cluster in earlier months for higher-risk
            default_month = int(np.random.beta(2, 5) * tenure) + 1
            default_month = max(1, min(default_month, tenure))
        else:
            default_month = tenure + 1  # never defaults

        stress_start = max(1, default_month - 3)   # stress starts ~3 months before default

        loan_status   = "Active"
        cumulative_dpd = 0
        current_dpd   = 0
        prev_dpd      = 0

        for month in range(1, tenure + 1):
            pay_date = orig_d + timedelta(days=30 * month)
            due_date = pay_date

            # Seasonal payment boost
            festival_boost = 0.06 if pay_date.month in FESTIVAL_MONTHS else 0.0

            # Stress phase behavior
            in_stress = month >= stress_start and will_default

            # Payment behavior probability
            if not in_stress:
                # Healthy: mostly on-time
                base_on_time = 0.85 + (emp_stab - 5) * 0.02 + festival_boost
                on_time_prob = np.clip(base_on_time, 0.50, 0.97)
                late_prob    = 1 - on_time_prob
                partial_prob = 0.03
            else:
                stress_depth = month - stress_start
                on_time_prob = max(0.05, 0.6 - stress_depth * 0.15)
                late_prob    = max(0.10, 0.30 + stress_depth * 0.10)
                partial_prob = 0.20 + stress_depth * 0.05

            rand = np.random.random()

            if rand < on_time_prob:
                payment_status = "Paid-On-Time"
                amount_paid    = emi
                dpd            = 0
            elif rand < on_time_prob + late_prob * 0.5:
                payment_status = "Late-1-30"
                amount_paid    = emi
                dpd            = np.random.randint(1, 31)
            elif rand < on_time_prob + late_prob:
                payment_status = "Late-31-60"
                amount_paid    = emi
                dpd            = np.random.randint(31, 61)
            elif rand < on_time_prob + late_prob + partial_prob:
                payment_status = "Partial"
                amount_paid    = round(emi * np.random.uniform(0.3, 0.9), 2)
                dpd            = np.random.randint(1, 45)
            else:
                payment_status = "Missed"
                amount_paid    = 0.0
                dpd            = 30 + (month - stress_start) * 15

            dpd = int(min(dpd, 180))

            # DPD bucket
            if dpd == 0:        dpd_bucket = "Current"
            elif dpd <= 30:     dpd_bucket = "DPD_1-30"
            elif dpd <= 60:     dpd_bucket = "DPD_31-60"
            elif dpd <= 90:     dpd_bucket = "DPD_61-90"
            else:               dpd_bucket = "DPD_90+"

            # Update loan status
            if dpd > 90:
                loan_status = "NPA"
            elif dpd > 30:
                loan_status = "SMA-2"
            elif dpd > 0:
                loan_status = "SMA-1"
            else:
                loan_status = "Standard"

            cumulative_dpd = max(cumulative_dpd, dpd)

            # Check actual default
            if month == default_month and will_default:
                loan_status = "Defaulted"

            rows.append({
                "repayment_id":    f"REP{len(rows):010d}",
                "loan_id":         loan["loan_id"],
                "customer_id":     loan["customer_id"],
                "installment_no":  month,
                "due_date":        due_date.strftime("%Y-%m-%d"),
                "payment_date":    (due_date + timedelta(days=dpd)).strftime("%Y-%m-%d"),
                "emi_due":         emi,
                "amount_paid":     amount_paid,
                "payment_status":  payment_status,
                "dpd":             dpd,
                "dpd_bucket":      dpd_bucket,
                "loan_status":     loan_status,
                "is_defaulted":    int(loan_status == "Defaulted"),
            })

            if loan_status == "Defaulted":
                break

    return pd.DataFrame(rows)

repayments = generate_repayment(loans, customers)
print(f"   ✓ {len(repayments):,} repayment records")
print(f"   ✓ Default rate: {repayments.is_defaulted.mean()*100:.2f}%")

# ─────────────────────────────────────────────
# 6. BEHAVIORAL SIGNALS  (monthly)
# ─────────────────────────────────────────────
print("\n[4/6] Generating behavioral signals...")

def generate_behavioral_signals(loans, customers, repayments):
    np.random.seed(55)

    cust_lookup  = customers.set_index("customer_id")[
        ["risk_tier","monthly_income","employment_type","employment_stability_score"]
    ]
    # Map loan to its default month for signal generation
    defaulted_loans = repayments[repayments["is_defaulted"] == 1][["loan_id","installment_no"]].rename(
        columns={"installment_no": "default_month"}
    ).drop_duplicates("loan_id")
    loan_default = defaulted_loans.set_index("loan_id")["default_month"].to_dict()

    rows = []
    for _, loan in loans.iterrows():
        if loan["cooling_off_exit"] == 1:
            continue
        lid    = loan["loan_id"]
        cid    = loan["customer_id"]
        cust   = cust_lookup.loc[cid]
        income = int(cust["monthly_income"])
        tier   = cust["risk_tier"]
        emp    = cust["employment_type"]
        tenure = int(loan["tenure_months"])
        def_m  = loan_default.get(lid, tenure + 1)

        # Base cash flow parameters by tier
        cf_vol_base = {"Prime": 0.08, "Near-Prime": 0.15,
                       "Subprime": 0.28, "Thin-File": 0.35}[tier]
        bal_vol_base= {"Prime": 0.10, "Near-Prime": 0.20,
                       "Subprime": 0.35, "Thin-File": 0.40}[tier]

        for month in range(1, tenure + 1):
            months_to_default = def_m - month
            in_stress = months_to_default <= 3 and def_m <= tenure

            # Cash flow consistency (0-1, higher = more consistent)
            if not in_stress:
                cf_consistency = np.clip(
                    np.random.beta(7, 2) - cf_vol_base * np.random.random(), 0.2, 1.0
                )
            else:
                cf_consistency = np.clip(
                    np.random.beta(2, 5) - 0.1 * (3 - months_to_default), 0.05, 0.6
                )

            # Balance volatility (std of daily balance / avg balance)
            bal_volatility = np.clip(
                bal_vol_base * np.random.lognormal(0, 0.5) +
                (0.15 if in_stress else 0),
                0.01, 1.5
            )

            # Income shock flag (sudden 30%+ drop in inflows)
            if in_stress:
                income_shock = int(np.random.random() < 0.35)
            elif tier in ["Gig-Worker","Daily-Wage","Subprime"]:
                income_shock = int(np.random.random() < 0.08)
            else:
                income_shock = int(np.random.random() < 0.02)

            # Spending shock (sudden spike in outflows)
            spending_shock = int(np.random.random() < (0.25 if in_stress else 0.05))

            # EMI obligation ratio (total EMI / monthly income)
            emi_obligation = round(float(loan["emi_amount"]) / income, 3)
            emi_obligation = np.clip(emi_obligation, 0.01, 0.90)

            # Avg monthly inflow
            base_inflow = income * np.random.uniform(0.85, 1.20)
            if income_shock:
                base_inflow *= np.random.uniform(0.5, 0.7)
            avg_monthly_inflow = int(base_inflow)

            # No. of active loan accounts
            active_accounts = np.random.choice([1,2,3,4,5], p=[0.50,0.28,0.13,0.06,0.03])

            rows.append({
                "signal_id":           f"SIG{len(rows):010d}",
                "loan_id":             lid,
                "customer_id":         cid,
                "month":               month,
                "cash_flow_consistency": round(cf_consistency, 4),
                "balance_volatility":  round(bal_volatility, 4),
                "income_shock_flag":   income_shock,
                "spending_shock_flag": spending_shock,
                "avg_monthly_inflow":  avg_monthly_inflow,
                "emi_obligation_ratio":round(emi_obligation, 3),
                "active_loan_accounts":active_accounts,
            })

    return pd.DataFrame(rows)

behavioral = generate_behavioral_signals(loans, customers, repayments)
print(f"   ✓ {len(behavioral):,} behavioral signal records")

# ─────────────────────────────────────────────
# 7. OUTCOMES TABLE
# ─────────────────────────────────────────────
print("\n[5/6] Generating outcomes & LTV proxies...")

def generate_outcomes(loans, repayments, customers):
    np.random.seed(33)

    cust_lookup = customers.set_index("customer_id")[["monthly_income","risk_tier"]]

    # Aggregate repayments per loan
    rep_agg = repayments.groupby("loan_id").agg(
        total_paid        = ("amount_paid", "sum"),
        installments_paid = ("installment_no", "count"),
        max_dpd           = ("dpd", "max"),
        is_defaulted      = ("is_defaulted", "max"),
        final_status      = ("loan_status", "last"),
    ).reset_index()

    outcome_rows = []
    for _, loan in loans.iterrows():
        lid = loan["loan_id"]
        if loan["cooling_off_exit"] == 1:
            outcome_rows.append({
                "loan_id":           lid,
                "customer_id":       loan["customer_id"],
                "final_loan_status": "Cooling-Off-Exit",
                "default_flag":      0,
                "max_dpd":           0,
                "total_amount_paid": 0,
                "recovery_amount":   0,
                "loss_given_default":0,
                "customer_ltv":      0,
                "risk_adjusted_return": 0,
                "is_profitable":     0,
            })
            continue

        agg = rep_agg[rep_agg["loan_id"] == lid]
        if agg.empty:
            continue
        agg = agg.iloc[0]

        cust  = cust_lookup.loc[loan["customer_id"]]
        income = int(cust["monthly_income"])
        ticket = int(loan["ticket_size"])
        apr    = float(loan["apr"])
        tenure = int(loan["tenure_months"])
        cac    = int(loan["cost_of_acquisition"])

        is_def   = int(agg["is_defaulted"])
        max_dpd  = int(agg["max_dpd"])
        paid     = float(agg["total_paid"])
        status   = str(agg["final_status"])

        # Recovery (30–60% for defaults in India digital lending)
        recovery = 0
        lgd      = 0
        if is_def:
            recovery_rate = np.random.uniform(0.25, 0.65)
            recovery      = int((ticket - paid) * recovery_rate)
            lgd           = max(0, ticket - paid - recovery)

        # LTV  (revenue over lifetime of customer)
        gross_interest = ticket * apr * (tenure / 12)
        processing_rev = int(loan["processing_fee"])
        total_revenue  = gross_interest + processing_rev
        total_cost     = cac + ticket * 0.02   # ops cost ~2%
        customer_ltv   = round(total_revenue - total_cost - lgd, 2)

        # Risk-adjusted return
        expected_loss = ticket * 0.50 * is_def  # simplified EL
        rar = round((total_revenue - total_cost - expected_loss) / ticket, 4)

        outcome_rows.append({
            "loan_id":            lid,
            "customer_id":        loan["customer_id"],
            "final_loan_status":  status,
            "default_flag":       is_def,
            "max_dpd":            max_dpd,
            "total_amount_paid":  round(paid, 2),
            "recovery_amount":    recovery,
            "loss_given_default": round(lgd, 2),
            "customer_ltv":       customer_ltv,
            "risk_adjusted_return": rar,
            "is_profitable":      int(customer_ltv > 0),
        })

    return pd.DataFrame(outcome_rows)

outcomes = generate_outcomes(loans, repayments, customers)
print(f"   ✓ {len(outcomes):,} outcome records")
print(f"   ✓ Default rate: {outcomes.default_flag.mean()*100:.2f}%")
print(f"   ✓ Profitable loans: {outcomes.is_profitable.mean()*100:.1f}%")

# ─────────────────────────────────────────────
# 8. SAVE ALL TABLES
# ─────────────────────────────────────────────
print("\n[6/6] Saving datasets...")

import os
OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)

customers.to_csv(f"{OUT}/01_customers.csv", index=False)
loans.to_csv(f"{OUT}/02_loans.csv", index=False)
repayments.to_csv(f"{OUT}/03_repayments.csv", index=False)
behavioral.to_csv(f"{OUT}/04_behavioral_signals.csv", index=False)
outcomes.to_csv(f"{OUT}/05_outcomes.csv", index=False)

# ─── MASTER JOINED TABLE (for quick analysis) ─────────────
print("   Building master joined table...")
master = (
    loans
    .merge(customers, on="customer_id", how="left")
    .merge(outcomes[["loan_id","final_loan_status","default_flag",
                      "max_dpd","customer_ltv","risk_adjusted_return",
                      "is_profitable","loss_given_default"]], on="loan_id", how="left")
)
master.to_csv(f"{OUT}/00_master_loans.csv", index=False)

# ─────────────────────────────────────────────
# 9. DATA DICTIONARY
# ─────────────────────────────────────────────
dd = """FILE: 01_customers.csv
customer_id               Unique customer identifier (CUST0000001)
risk_tier                 Prime / Near-Prime / Subprime / Thin-File
city                      City of residence
geo_tier                  Metro / Tier-2 / Tier-3
state                     Indian state
age                       Age in years (18–58)
gender                    Male / Female / Other
employment_type           Salaried-Govt, Salaried-Private, Self-Employed-Business,
                          Self-Employed-Prof, Gig-Worker, Daily-Wage, Student, Unemployed
employment_stability_score  1–10 (10 = most stable)
monthly_income            Gross monthly income in INR
bureau_score              CIBIL-equivalent score (300–900; 0 = no bureau history)
new_to_credit_flag        1 if no bureau history (Thin-File)
existing_customer         1 if previously onboarded with lender
kyc_type                  Aadhaar-OTP / Video-KYC / Physical

FILE: 02_loans.csv
loan_id                   Unique loan identifier (LN00000001)
customer_id               FK → customers
product_type              Personal-Loan / SME-Working-Capital / BNPL /
                          Two-Wheeler-Loan / Consumer-Durable / Education-Loan
ticket_size               Loan principal in INR
tenure_months             Loan tenure in months
apr                       Annual Percentage Rate (RBI-compliant, all-inclusive)
processing_fee            Upfront processing fee in INR
emi_amount                Monthly EMI in INR (reducing balance method)
origination_date          Date of loan disbursal
origination_risk_grade    A+ / A / A- / B+ / B / B- / C+ / C / D
acquisition_channel       Organic-App / Referral / Paid-Digital / DSA-Agent /
                          Bank-Partnership / Corporate-Tie-Up / NBFC-Embedded
cost_of_acquisition       CAC in INR
approval_turnaround_days  Days from application to disbursal
cooling_off_exit          1 if borrower exited during RBI-mandated cooling-off period

FILE: 03_repayments.csv
repayment_id              Unique repayment event ID
loan_id                   FK → loans
customer_id               FK → customers
installment_no            Month number (1 = first EMI)
due_date                  EMI due date
payment_date              Actual payment date
emi_due                   Amount due
amount_paid               Amount actually paid
payment_status            Paid-On-Time / Late-1-30 / Late-31-60 / Partial / Missed
dpd                       Days Past Due
dpd_bucket                Current / DPD_1-30 / DPD_31-60 / DPD_61-90 / DPD_90+
loan_status               Standard / SMA-1 / SMA-2 / NPA / Defaulted
is_defaulted              1 if this installment row marks a default event

FILE: 04_behavioral_signals.csv
signal_id                 Unique signal record ID
loan_id                   FK → loans
customer_id               FK → customers
month                     Month number in loan lifecycle
cash_flow_consistency     0–1 (1 = perfectly consistent cash flows)
balance_volatility        Ratio of balance std-dev to mean (higher = riskier)
income_shock_flag         1 if sudden ≥30% drop in inflows detected
spending_shock_flag       1 if sudden spike in outflows detected
avg_monthly_inflow        Average monthly bank inflow in INR
emi_obligation_ratio      EMI / monthly income (FOIR proxy)
active_loan_accounts      Total open loan accounts (including this one)

FILE: 05_outcomes.csv
loan_id                   FK → loans
customer_id               FK → customers
final_loan_status         Closed / Defaulted / Active / Cooling-Off-Exit
default_flag              1 if loan defaulted
max_dpd                   Maximum DPD observed in loan lifetime
total_amount_paid         Total INR repaid
recovery_amount           Post-default recovery in INR
loss_given_default        Irrecoverable loss in INR
customer_ltv              Estimated customer lifetime value in INR
risk_adjusted_return      (Revenue - Cost - ExpectedLoss) / Principal
is_profitable             1 if customer_ltv > 0

FILE: 00_master_loans.csv
  Wide table joining loans + customers + outcomes for quick analysis.
"""

with open(f"{OUT}/DATA_DICTIONARY.txt", "w") as f:
    f.write(dd)

# ─────────────────────────────────────────────
# 10. QUICK VALIDATION REPORT
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  DATASET SUMMARY")
print("="*60)

print(f"\n{'Table':<30} {'Rows':>10} {'Columns':>10}")
print("-"*52)
for name, df in [
    ("00_master_loans",      master),
    ("01_customers",         customers),
    ("02_loans",             loans),
    ("03_repayments",        repayments),
    ("04_behavioral_signals",behavioral),
    ("05_outcomes",          outcomes),
]:
    print(f"{name:<30} {len(df):>10,} {len(df.columns):>10}")

print(f"\n── Customer Mix ──")
print(customers["risk_tier"].value_counts(normalize=True).mul(100).round(1).to_string())

print(f"\n── Product Mix ──")
print(loans["product_type"].value_counts().to_string())

print(f"\n── Acquisition Channel Mix ──")
print(loans["acquisition_channel"].value_counts().to_string())

print(f"\n── Default Rate by Risk Grade ──")
dr = master.groupby("origination_risk_grade")["default_flag"].mean().mul(100).round(2)
print(dr.sort_values(ascending=False).to_string())

print(f"\n── Geography Mix ──")
print(customers["geo_tier"].value_counts(normalize=True).mul(100).round(1).to_string())

print("\n✅ All files saved to /mnt/user-data/outputs/")
print("="*60)
