"""
BSF AI Banking Intelligence System — Core Engine
All AI logic: Segmentation, NBA, Behavior, Credit Decision
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# ── Constants ─────────────────────────────────────────────────
SEGMENT_ORDER  = ['Standard', 'Silver', 'Gold', 'Platinum', 'VIP']
RISK_ORDER     = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
SAIBOR_3M      = 5.50

SEGMENT_COLORS = {
    'Standard':  '#7F8C8D',
    'Silver':    '#BDC3C7',
    'Gold':      '#F1C40F',
    'Platinum':  '#8E44AD',
    'VIP':       '#E74C3C',
}

RISK_COLORS = {
    'Very Low':  '#27AE60',
    'Low':       '#2ECC71',
    'Medium':    '#F39C12',
    'High':      '#E67E22',
    'Very High': '#C0392B',
}

# ── Product Catalog (Real BSF Products) ──────────────────────
PRODUCT_CATALOG = {
    'CC_TRAVEL': {
        'name': 'BSF World Travel Card',
        'category': 'Credit Card', 'tier': 3,
        'min_income': 15000, 'min_score': 700, 'margin': 0.26,
        'highlights': 'Airport lounge · Travel insurance · Reward points',
    },
    'PL_PERSONAL': {
        'name': 'BSF Personal Finance',
        'category': 'Loan', 'tier': 2,
        'min_income': 5000, 'min_score': 620, 'margin': 0.15,
        'highlights': 'Up to SAR 500K · Flexible repayment',
    },
    'CC_CASHBACK': {
        'name': 'BSF Cashback Card',
        'category': 'Credit Card', 'tier': 2,
        'min_income': 8000, 'min_score': 640, 'margin': 0.22,
        'highlights': 'Cashback on all purchases · No minimum spend',
    },
    'PL_HOME': {
        'name': 'BSF Home Finance',
        'category': 'Loan', 'tier': 3,
        'min_income': 12000, 'min_score': 700, 'margin': 0.12,
        'highlights': 'Up to SAR 6M · 25 years tenure',
    },
    'SA_SAVING': {
        'name': 'BSF Savings Account',
        'category': 'Savings', 'tier': 1,
        'min_income': 0, 'min_score': 0, 'margin': 0.05,
        'highlights': 'No minimum balance · Competitive profit rate',
    },
}

# ── Channel Priority ──────────────────────────────────────────
CHANNEL_PRIORITY = {
    'Standard':  ['SMS', 'WhatsApp', 'Branch Agent', 'Push Notification', 'RM'],
    'Silver':    ['WhatsApp', 'Push Notification', 'SMS', 'Branch Agent', 'RM'],
    'Gold':      ['Push Notification', 'WhatsApp', 'Branch Agent', 'RM', 'SMS'],
    'Platinum':  ['RM', 'Push Notification', 'WhatsApp', 'Branch Agent', 'SMS'],
    'VIP':       ['RM', 'Branch Agent', 'Push Notification', 'WhatsApp', 'SMS'],
}

PRODUCT_CHANNEL_OVERRIDE = {
    'PL_HOME':     'RM',
    'CC_TRAVEL':   'Push Notification',
    'CC_CASHBACK': 'Push Notification',
    'PL_PERSONAL': 'WhatsApp',
    'SA_SAVING':   'SMS',
}

# ── Credit Rules ──────────────────────────────────────────────
CREDIT_LIMIT_RULES = {
    'Very Low':  {'multiplier': 4.0, 'card_tier': 'Elite/Platinum', 'max_cap': 200000},
    'Low':       {'multiplier': 3.0, 'card_tier': 'Gold/Privilege',  'max_cap': 120000},
    'Medium':    {'multiplier': 2.0, 'card_tier': 'Advance',         'max_cap': 60000},
    'High':      {'multiplier': 1.0, 'card_tier': 'Classic',         'max_cap': 25000},
    'Very High': {'multiplier': 0.5, 'card_tier': 'Entry-Level',     'max_cap': 10000},
}

LOAN_RULES = {
    'Very Low':  {'income_multiple': 24, 'max_tenure_months': 60, 'max_cap': 500000},
    'Low':       {'income_multiple': 18, 'max_tenure_months': 48, 'max_cap': 350000},
    'Medium':    {'income_multiple': 12, 'max_tenure_months': 36, 'max_cap': 200000},
    'High':      {'income_multiple':  6, 'max_tenure_months': 24, 'max_cap': 80000},
    'Very High': {'income_multiple':  3, 'max_tenure_months': 12, 'max_cap': 30000},
}

RATE_SPREAD = {
    'Very Low':  {'spread': 1.50, 'cc_apr': 18.0},
    'Low':       {'spread': 2.00, 'cc_apr': 21.0},
    'Medium':    {'spread': 3.00, 'cc_apr': 24.0},
    'High':      {'spread': 4.50, 'cc_apr': 27.0},
    'Very High': {'spread': 6.00, 'cc_apr': 30.0},
}

# ─────────────────────────────────────────────────────────────
# MODULE 1: SEGMENTATION
# ─────────────────────────────────────────────────────────────

def compute_segment(income: float, credit_score: float, balance: float,
                    tenure: int, products_owned: list) -> str:
    score = 0
    score += min(40, income / 1000)
    score += (credit_score - 300) / 600 * 25
    score += min(15, balance / 20000)
    score += min(10, tenure / 18)
    score += len(products_owned) * 2

    if score >= 80:  return 'VIP'
    elif score >= 60: return 'Platinum'
    elif score >= 40: return 'Gold'
    elif score >= 22: return 'Silver'
    else:             return 'Standard'


def compute_engagement_score(app_logins: float, txn_frequency: float,
                              num_products: int) -> float:
    return round(app_logins * 0.3 + txn_frequency / 12 * 0.4 + num_products * 2 * 0.3, 2)

# ─────────────────────────────────────────────────────────────
# MODULE 1: NBA ENGINE
# ─────────────────────────────────────────────────────────────

def compute_reason_codes(customer: dict, product_id: str, product_info: dict) -> list:
    specific, profile = [], []
    income   = customer['monthly_income']
    score    = customer['credit_score']
    eng      = customer.get('engagement_score', 5)
    spend    = customer.get('total_spend_12m', 0)
    n_prod   = customer.get('num_products_owned', 1)
    has_cc   = customer.get('has_credit_card', 0)
    has_loan = customer.get('has_loan', 0)
    tenure   = customer.get('tenure_months', 0)
    late     = customer.get('late_payment_count', 0)
    channel  = customer.get('preferred_channel', '')
    category = product_info['category']

    if product_id == 'CC_TRAVEL':
        specific.append('Frequent high-value spending — ideal travel card candidate')
        specific.append('No existing credit card — strong upsell opportunity')
        if spend >= 100000:
            specific.append(f'Annual spend SAR {spend:,.0f} qualifies for premium travel rewards')
        if income >= 15000:
            specific.append('Income exceeds minimum for World Travel Card eligibility')
    elif product_id == 'PL_PERSONAL':
        specific.append('No existing personal loan — low liability exposure')
        specific.append(f'Eligible for up to SAR {int(income * 15):,} based on salary profile')
        specific.append('High loan approval probability — strong repayment capacity')
    elif product_id == 'CC_CASHBACK':
        specific.append('Regular spending pattern — cashback maximizes daily purchase value')
        specific.append('No existing credit card — immediate cashback benefit available')
        if spend >= 50000:
            specific.append(f'Estimated annual cashback: SAR {int(spend * 0.02):,}')
    elif product_id == 'PL_HOME':
        specific.append('No existing home finance — homeownership opportunity identified')
        specific.append(f'Income of SAR {income:,}/month supports home finance eligibility')
        if tenure >= 24:
            specific.append('Existing BSF relationship qualifies for preferential rate')
    elif product_id == 'SA_SAVING':
        specific.append('Balance indicates savings potential not yet optimized')
        specific.append('Savings account unlocks higher profit rate on existing deposits')

    if income >= 20000:   profile.append('Strong income profile')
    elif income >= 10000: profile.append('Solid salary eligibility')
    if score >= 750:      profile.append('Excellent credit standing')
    elif score >= 700:    profile.append('Good credit standing')
    if late == 0:         profile.append('Clean repayment history')
    if tenure >= 60:      profile.append(f'Long-standing BSF customer ({tenure // 12} years)')
    if eng >= 15:         profile.append('High digital engagement')
    if n_prod <= 2:       profile.append('Low product penetration — cross-sell opportunity')

    return (specific + profile)[:5]


def compute_match_score(customer: dict, product_id: str, product_info: dict) -> float:
    income  = customer['monthly_income']
    score   = customer['credit_score']
    segment = customer.get('segment', 'Standard')
    eng     = customer.get('engagement_score', 5)
    has_cc  = customer.get('has_credit_card', 0)
    has_loan= customer.get('has_loan', 0)
    spend   = customer.get('total_spend_12m', 0)
    tenure  = customer.get('tenure_months', 0)

    if income < product_info['min_income'] or score < product_info['min_score']:
        return 0.0

    min_inc = product_info['min_income']
    min_cs  = product_info['min_score']
    income_score = min(1.0, (income - min_inc) / (min_inc + 1)) if min_inc > 0 else 1.0
    cs_score     = min(1.0, (score - min_cs) / (900 - min_cs + 1)) if min_cs > 0 else score / 900
    beh_score    = min(1.0, eng / 20.0)
    seg_tier     = {'Standard': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4, 'VIP': 5}
    seg_fit      = 1.0 if product_info['tier'] <= seg_tier.get(segment, 1) else \
                   max(0.2, 1.0 - (product_info['tier'] - seg_tier.get(segment, 1)) * 0.2)
    base = 0.30 * income_score + 0.25 * cs_score + 0.25 * beh_score + 0.20 * seg_fit

    boost = 0.0
    category = product_info['category']
    if category == 'Credit Card' and not has_cc:
        boost += 0.15
        if spend >= 100000: boost += 0.05
    if product_id == 'CC_TRAVEL' and income >= 15000 and spend >= 100000:
        boost += 0.10
    if category == 'Loan' and not has_loan and income >= 8000:
        boost += 0.12
    if product_id == 'PL_HOME' and income >= 12000 and tenure >= 24:
        boost += 0.08
    if category in ['Loyalty', 'FX/Transfer'] and not has_cc and not has_loan:
        boost -= 0.15

    noise = np.random.uniform(-0.02, 0.02)
    raw   = min(0.99, max(0.30, base + boost + noise))
    return round(raw * 100, 1)


def get_product_limit(customer: dict, product_id: str) -> str:
    income, score = customer['monthly_income'], customer['credit_score']
    if 'CC' in product_id:
        limit = int(income * np.random.uniform(1.5, 4.0))
        tier  = 'Elite/Platinum' if score > 750 else ('Gold/Privilege' if score > 700 else 'Classic')
        return f'SAR {limit:,} | {tier}'
    elif 'PL' in product_id:
        return f'SAR {int(income * np.random.uniform(10, 24)):,}'
    elif 'SA' in product_id:
        return f'{round(np.random.uniform(3.5, 6.5), 2)}% p.a.'
    return 'Available'


def nba_engine(customer: dict, top_n: int = 2) -> list:
    owned  = set(str(customer.get('products_owned', '')).split('|'))
    scores = []
    for pid, pinfo in PRODUCT_CATALOG.items():
        if pid in owned: continue
        ms = compute_match_score(customer, pid, pinfo)
        if ms > 0:
            scores.append({
                'product_id':        pid,
                'product_name':      pinfo['name'],
                'category':          pinfo['category'],
                'match_score':       ms,
                'recommended_limit': get_product_limit(customer, pid),
                'reason_codes':      compute_reason_codes(customer, pid, pinfo),
            })
    return sorted(scores, key=lambda x: x['match_score'], reverse=True)[:top_n]


# ── ML-POWERED NBA (LightGBM per product) ─────────────────────
def nba_engine_ml(customer: dict, top_n: int = 3) -> list:
    """
    Next-Best-Action using trained LightGBM classifiers.
    Returns top_n products ranked by ML buy-probability,
    enriched with catalog metadata and reason codes.
    """
    from utils.ml_nba import predict_products
    ml_recs = predict_products(customer)
    owned   = set(str(customer.get('products_owned', '')).split('|'))
    results = []
    for rec in ml_recs:
        pid   = rec['product_id']
        pinfo = PRODUCT_CATALOG.get(pid, {})
        if not pinfo or pid in owned:
            continue
        results.append({
            'product_id':        pid,
            'product_name':      rec['product_name'],
            'category':          rec['category'],
            'match_score':       rec['confidence_pct'],
            'ml_probability':    rec['ml_probability'],
            'confidence_pct':    rec['confidence_pct'],
            'recommended_limit': get_product_limit(customer, pid),
            'reason_codes':      compute_reason_codes(customer, pid, pinfo),
            'highlights':        pinfo.get('highlights', ''),
            'scored_by':         'LightGBM',
        })
        if len(results) >= top_n:
            break
    return results


def select_channel(customer: dict, top_product_id: str) -> dict:
    segment = str(customer.get('segment', 'Standard'))
    eng     = customer.get('engagement_score', 0)
    pref    = customer.get('preferred_channel', '')
    if top_product_id in PRODUCT_CHANNEL_OVERRIDE:
        primary  = PRODUCT_CHANNEL_OVERRIDE[top_product_id]
        priority = CHANNEL_PRIORITY.get(segment, CHANNEL_PRIORITY['Standard'])
        backup   = next((c for c in priority if c != primary), 'SMS')
        return {'primary': primary, 'backup': backup, 'reason': 'Product-specific override'}
    if eng > 15 and pref == 'Mobile App':
        return {'primary': 'Push Notification', 'backup': 'WhatsApp', 'reason': 'High digital engagement'}
    priority = CHANNEL_PRIORITY.get(segment, CHANNEL_PRIORITY['Standard'])
    return {'primary': priority[0], 'backup': priority[1], 'reason': f'Segment-{segment} priority'}

# ─────────────────────────────────────────────────────────────
# MODULE 2: BEHAVIOR ANALYSIS
# ─────────────────────────────────────────────────────────────

def compute_behavior_score(app_logins: float, txn_rate: float,
                            tenure: int, num_products: int) -> float:
    score = (
        min(20, app_logins / 30 * 20) +
        txn_rate * 25 +
        min(15, tenure / 120 * 15) +
        min(20, num_products / 5 * 20) +
        20  # base
    )
    return round(min(100, max(0, score)), 1)


def categorize_behavior(score: float) -> str:
    if score >= 75:  return 'Highly Active'
    elif score >= 50: return 'Active'
    elif score >= 30: return 'Moderate'
    else:             return 'Dormant'


def detect_anomaly(late_payments: int, credit_score: float,
                   monthly_income: float, spend: float) -> dict:
    flags = []
    if late_payments >= 3:
        flags.append(f'{late_payments} late payments detected')
    if credit_score < 580 and spend > monthly_income * 18:
        flags.append('High spend relative to income with low credit score')
    if spend > monthly_income * 24:
        flags.append('Spending exceeds 2× annual income')
    return {
        'flag': len(flags) > 0,
        'reasons': flags,
        'type': 'Churn Risk' if late_payments >= 3 else ('Fraud Signal' if len(flags) > 1 else 'Behavioral Outlier'),
        'severity': 'High' if len(flags) >= 2 else ('Medium' if len(flags) == 1 else 'Normal'),
    }


def detect_drift(behavior_score: float, segment: str) -> str:
    seg_mean = {'Standard': 30, 'Silver': 45, 'Gold': 58, 'Platinum': 70, 'VIP': 82}.get(segment, 50)
    if behavior_score > seg_mean + 12: return 'Upgrade Candidate'
    elif behavior_score < seg_mean - 12: return 'Downgrade Risk'
    else: return 'Stable'


def generate_agent_insight(behavior_score: float, behavior_cat: str,
                            anomaly: dict, drift: str, segment: str,
                            app_logins: float) -> list:
    insights = []
    if behavior_cat == 'Highly Active':
        insights.append('✅ Highly active customer — prioritize for premium offer delivery')
    elif behavior_cat == 'Dormant':
        insights.append('⚠️ Dormant customer — trigger re-engagement campaign')
    if app_logins >= 15:
        insights.append('📱 High digital engagement — Push Notification preferred')
    if drift == 'Upgrade Candidate':
        insights.append(f'⬆️ Behavior exceeds {segment} norms — segment upgrade candidate')
    elif drift == 'Downgrade Risk':
        insights.append(f'⬇️ Behavior below {segment} norms — retention strategy needed')
    if anomaly['flag']:
        for r in anomaly['reasons']:
            insights.append(f'🚨 Anomaly: {r}')
    if not insights:
        insights.append('✅ Normal behavioral patterns — continue monitoring')
    return insights

# ─────────────────────────────────────────────────────────────
# MODULE 3: CREDIT DECISION
# ─────────────────────────────────────────────────────────────

def compute_risk_score(credit_score: float, late_payments: int, tenure: int,
                       balance: float, income: float, behavior_score: float,
                       anomaly_flag: bool) -> float:
    credit_norm  = (credit_score - 300) / 600
    pay_disc     = 1 - (late_payments / (late_payments + 2))
    inc_stab     = min(1, tenure / 180)
    bal_adq      = min(1, balance / (income * 10 + 1))
    beh_norm     = behavior_score / 100
    raw = (credit_norm * 40 + pay_disc * 25 + inc_stab * 15 +
           bal_adq * 10 + beh_norm * 10 -
           (8 if anomaly_flag else 0) - late_payments * 3)
    return round(min(100, max(0, raw)), 1)


def classify_risk(risk_score: float) -> str:
    if risk_score >= 80:   return 'Very Low'
    elif risk_score >= 65: return 'Low'
    elif risk_score >= 50: return 'Medium'
    elif risk_score >= 35: return 'High'
    else:                  return 'Very High'


def recommend_credit(income: float, credit_score: float, risk_category: str) -> dict:
    cc_rules   = CREDIT_LIMIT_RULES.get(risk_category, CREDIT_LIMIT_RULES['Medium'])
    loan_rules = LOAN_RULES.get(risk_category, LOAN_RULES['Medium'])
    rate_rules = RATE_SPREAD.get(risk_category, RATE_SPREAD['Medium'])

    cc_limit   = min(int(income * cc_rules['multiplier']), cc_rules['max_cap'])
    cc_limit   = max(5000, round(cc_limit / 1000) * 1000)
    loan_amt   = min(int(income * loan_rules['income_multiple']), loan_rules['max_cap'])
    loan_amt   = max(10000, round(loan_amt / 5000) * 5000)
    rate       = round(SAIBOR_3M + rate_rules['spread'] + (750 - credit_score) / 450 * 0.5, 2)
    rate       = max(3.0, min(15.0, rate))
    tenure     = loan_rules['max_tenure_months']
    monthly_r  = rate / 100 / 12
    installment = int(loan_amt * (monthly_r * (1 + monthly_r) ** tenure) /
                      ((1 + monthly_r) ** tenure - 1)) if monthly_r > 0 else int(loan_amt / tenure)
    dbr         = installment / income if income > 0 else 1

    cc_eligible   = income >= 3000 and credit_score >= 580
    loan_eligible = income >= 5000 and credit_score >= 620 and dbr <= 0.33

    return {
        'cc_limit':           cc_limit if cc_eligible else 0,
        'cc_card_tier':       cc_rules['card_tier'],
        'cc_eligible':        cc_eligible,
        'cc_apr':             rate_rules['cc_apr'],
        'loan_amount':        loan_amt if loan_eligible else 0,
        'loan_tenure_months': tenure,
        'loan_installment':   installment if loan_eligible else 0,
        'loan_dbr':           round(dbr, 3),
        'loan_eligible':      loan_eligible,
        'interest_rate':      rate,
    }


def get_approval_status(risk_category: str, anomaly_flag: bool) -> dict:
    if risk_category in ['Very Low', 'Low'] and not anomaly_flag:
        return {'status': 'Auto-Approve Eligible',    'level': 'auto',   'icon': '✅',
                'desc': 'Standard verification only — proceed with offer delivery.'}
    elif risk_category == 'Medium' or anomaly_flag:
        return {'status': 'Manual Review Required',   'level': 'manual', 'icon': '📋',
                'desc': 'Credit officer review + document verification required.'}
    else:
        return {'status': 'Senior Review Required',   'level': 'senior', 'icon': '⚠️',
                'desc': 'Senior credit officer + risk committee approval required.'}


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def run_full_pipeline(customer_input: dict) -> dict:
    """Run complete BSF AI pipeline for a single customer."""
    income   = float(customer_input.get('monthly_income', 0))
    balance  = float(customer_input.get('account_balance', 0))
    score    = float(customer_input.get('credit_score', 600))
    tenure   = int(customer_input.get('tenure_months', 0))
    late     = int(customer_input.get('late_payment_count', 0))
    spend    = float(customer_input.get('total_spend_12m', 0))
    logins   = float(customer_input.get('app_logins_monthly', 0))
    products = str(customer_input.get('products_owned', 'SA_SAVING'))
    channel  = customer_input.get('preferred_channel', 'Mobile App')
    owned    = [p.strip() for p in products.split('|') if p.strip()]
    freq     = float(customer_input.get('txn_frequency_12m', max(1, spend / 2000)))

    has_cc   = int(any('CC' in p for p in owned))
    has_loan = int(any('PL' in p for p in owned))
    n_prod   = len(owned)

    # ── Module 1 ──────────────────────────────────────────────
    segment    = compute_segment(income, score, balance, tenure, owned)
    eng_score  = compute_engagement_score(logins, freq, n_prod)

    customer_enriched = {
        **customer_input,
        'segment': segment, 'engagement_score': eng_score,
        'has_credit_card': has_cc, 'has_loan': has_loan,
        'num_products_owned': n_prod, 'products_owned': products,
        'total_spend_12m': spend, 'tenure_months': tenure,
        'late_payment_count': late,
    }
    nba_recs   = nba_engine(customer_enriched, top_n=2)
    top_pid    = nba_recs[0]['product_id'] if nba_recs else ''
    channel_rec = select_channel(customer_enriched, top_pid)

    # ── Module 2 ──────────────────────────────────────────────
    txn_rate   = min(0.9, spend / (income * 12 + 1) * 0.6 + logins / 60 * 0.4) if income > 0 else 0.1
    beh_score  = compute_behavior_score(logins, txn_rate, tenure, n_prod)
    beh_cat    = categorize_behavior(beh_score)
    anomaly    = detect_anomaly(late, score, income, spend)
    drift      = detect_drift(beh_score, segment)
    insights   = generate_agent_insight(beh_score, beh_cat, anomaly, drift, segment, logins)

    # ── Module 3 ──────────────────────────────────────────────
    risk_score  = compute_risk_score(score, late, tenure, balance, income, beh_score, anomaly['flag'])
    risk_cat    = classify_risk(risk_score)
    credit      = recommend_credit(income, score, risk_cat)
    approval    = get_approval_status(risk_cat, anomaly['flag'])

    return {
        # M1
        'segment':          segment,
        'engagement_score': eng_score,
        'nba_recommendations': nba_recs,
        'channel':          channel_rec,
        # M2
        'behavior_score':   beh_score,
        'behavior_category': beh_cat,
        'anomaly':          anomaly,
        'drift_status':     drift,
        'agent_insights':   insights,
        # M3
        'risk_score':       risk_score,
        'risk_category':    risk_cat,
        'credit':           credit,
        'approval':         approval,
    }


# ─────────────────────────────────────────────────────────────
# SYNTHETIC DASHBOARD DATA
# ─────────────────────────────────────────────────────────────

def generate_dashboard_stats() -> dict:
    """Pre-computed stats for overview dashboard."""
    return {
        'total_customers': 5000,
        'segment_dist': {'Standard': 1480, 'Silver': 1250, 'Gold': 1100, 'Platinum': 760, 'VIP': 410},
        'behavior_dist': {'Highly Active': 900, 'Active': 1600, 'Moderate': 1750, 'Dormant': 750},
        'risk_dist': {'Very Low': 1100, 'Low': 1500, 'Medium': 1400, 'High': 700, 'Very High': 300},
        'anomaly_count': 250,
        'upgrade_candidates': 600,
        'cc_eligible_pct': 78,
        'loan_eligible_pct': 65,
        'auto_approve_pct': 45,
        'model_accuracy': {'XGBoost': 82.2, 'Random Forest': 79.0, 'Gradient Boosting': 81.5, 'Ensemble': 84.1},
        'top_products': [
            {'name': 'BSF World Travel Card',  'count': 1820, 'avg_score': 91.2},
            {'name': 'BSF Personal Finance',   'count': 1540, 'avg_score': 85.7},
            {'name': 'BSF Cashback Card',      'count': 1210, 'avg_score': 79.3},
            {'name': 'BSF Home Finance',       'count': 890,  'avg_score': 83.1},
            {'name': 'BSF Savings Account',    'count': 540,  'avg_score': 71.5},
        ],
        'anomaly_types': {
            'Churn Risk': 98, 'Activity Spike': 62,
            'Journey Friction': 45, 'Fraud Signal': 28, 'Outlier': 17,
        },
    }
