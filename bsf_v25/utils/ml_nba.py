"""
BSF AI Banking Intelligence System
ML-Powered NBA Engine — LightGBM Cross-sell Models
Developed by Accord Business Group (ABG)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────
PRODUCTS = ['CC_TRAVEL', 'PL_PERSONAL', 'CC_CASHBACK', 'PL_HOME', 'SA_SAVING']

PRODUCT_NAMES = {
    'CC_TRAVEL':   'BSF World Travel Card',
    'PL_PERSONAL': 'BSF Personal Finance',
    'CC_CASHBACK': 'BSF Cashback Card',
    'PL_HOME':     'BSF Home Finance',
    'SA_SAVING':   'BSF Savings Account',
}

PRODUCT_CATEGORIES = {
    'CC_TRAVEL':   'Credit Card',
    'PL_PERSONAL': 'Loan',
    'CC_CASHBACK': 'Credit Card',
    'PL_HOME':     'Loan',
    'SA_SAVING':   'Savings',
}

SEGMENT_MAP = {'Standard': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3, 'VIP': 4}

FEATURES = [
    'income', 'credit_score', 'balance', 'tenure',
    'num_products', 'engagement_score', 'annual_spend',
    'segment_enc', 'behavior_score', 'app_logins',
    'has_credit_card', 'has_loan',
]

FEATURE_LABELS = {
    'income':           'Monthly Income',
    'credit_score':     'Credit Score',
    'balance':          'Account Balance',
    'tenure':           'Tenure (months)',
    'num_products':     'Products Owned',
    'engagement_score': 'Engagement Score',
    'annual_spend':     'Annual Spend',
    'segment_enc':      'Customer Segment',
    'behavior_score':   'Behavior Score',
    'app_logins':       'App Logins',
    'has_credit_card':  'Has Credit Card',
    'has_loan':         'Has Loan',
}

# ── Global cache ───────────────────────────────────────────────
_MODELS  = {}
_METRICS = {}
_TRAINED = False


# ── Helpers ────────────────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def _sample_labels(probs, rng):
    return (rng.random(len(probs)) < probs).astype(int)


# ── Synthetic Training Data ────────────────────────────────────
def _generate_training_data(n=12000):
    rng = np.random.default_rng(42)

    seg_names = ['Standard', 'Silver', 'Gold', 'Platinum', 'VIP']
    seg_probs = [0.30, 0.25, 0.22, 0.15, 0.08]
    segments  = rng.choice(seg_names, size=n, p=seg_probs)

    income_ranges  = {'Standard':(3000,7000),  'Silver':(7000,12000),
                      'Gold':(12000,20000), 'Platinum':(20000,35000), 'VIP':(35000,80000)}
    balance_ranges = {'Standard':(5000,20000),  'Silver':(15000,50000),
                      'Gold':(40000,100000),'Platinum':(80000,200000),'VIP':(150000,600000)}
    score_ranges   = {'Standard':(540,680),     'Silver':(620,720),
                      'Gold':(680,780),     'Platinum':(720,820),     'VIP':(780,900)}

    income, balance, credit_score = [], [], []
    for seg in segments:
        lo, hi = income_ranges[seg];  income.append(rng.uniform(lo, hi))
        lo, hi = balance_ranges[seg]; balance.append(rng.uniform(lo, hi))
        lo, hi = score_ranges[seg];   credit_score.append(rng.uniform(lo, hi))

    income       = np.array(income)
    balance      = np.array(balance)
    credit_score = np.array(credit_score)
    segment_enc  = np.array([SEGMENT_MAP[s] for s in segments])
    tenure       = rng.integers(1, 181, n)
    num_products = rng.integers(1, 7, n)
    annual_spend = income * rng.uniform(0.4, 3.0, n)
    app_logins   = rng.integers(0, 120, n)
    has_cc       = ((credit_score > 640).astype(int) * rng.integers(0, 2, n))
    has_loan     = ((income > 6000).astype(int) * rng.integers(0, 2, n))
    engagement   = np.clip(app_logins * 0.30 + num_products * 8 + rng.normal(0, 5, n), 0, 100)
    behavior     = np.clip(engagement * 0.75 + rng.normal(0, 10, n), 0, 100)

    df = pd.DataFrame({
        'income': income, 'credit_score': credit_score, 'balance': balance,
        'tenure': tenure, 'num_products': num_products, 'annual_spend': annual_spend,
        'app_logins': app_logins, 'engagement_score': engagement,
        'behavior_score': behavior, 'segment_enc': segment_enc,
        'has_credit_card': has_cc, 'has_loan': has_loan,
    })

    # Labels
    df['label_CC_TRAVEL'] = _sample_labels(_sigmoid(
        0.0004*(income-15000) + 0.005*(credit_score-700) +
        0.000003*(annual_spend-80000) + 0.4*(segment_enc-2) + 0.5*(1-has_cc)
    ), rng)

    df['label_PL_PERSONAL'] = _sample_labels(_sigmoid(
        0.0002*(income-5000) + 0.004*(credit_score-620) +
        0.003*tenure + 0.6*(1-has_loan)
    ), rng)

    df['label_CC_CASHBACK'] = _sample_labels(_sigmoid(
        0.0003*(income-8000) + 0.003*(credit_score-640) +
        0.015*(engagement-40) + 0.4*(1-has_cc)
    ), rng)

    df['label_PL_HOME'] = _sample_labels(_sigmoid(
        0.00025*(income-12000) + 0.006*(credit_score-700) +
        0.004*(tenure-24) + 0.3*(segment_enc-1) + 0.5*(1-has_loan)
    ), rng)

    df['label_SA_SAVING'] = _sample_labels(_sigmoid(
        -0.1*(num_products-2) + 0.003*(behavior-30) - 0.000005*balance
    ), rng)

    return df


# ── Training ───────────────────────────────────────────────────
def train_all_models(force=False):
    global _MODELS, _METRICS, _TRAINED
    if _TRAINED and not force:
        return _MODELS, _METRICS

    df = _generate_training_data(n=12000)
    X  = df[FEATURES]

    for product in PRODUCTS:
        y = df['label_' + product]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        model = LGBMClassifier(
            n_estimators=300, learning_rate=0.04, max_depth=6,
            num_leaves=31, subsample=0.80, colsample_bytree=0.80,
            min_child_samples=20, class_weight='balanced',
            importance_type='gain',
            random_state=42, verbose=-1,
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        _MODELS[product] = model
        _METRICS[product] = {
            'auc':           round(roc_auc_score(y_test, y_prob), 3),
            'avg_precision': round(average_precision_score(y_test, y_prob), 3),
            'f1':            round(f1_score(y_test, y_pred, zero_division=0), 3),
            'positive_rate': round(float(y.mean()), 3),
            'n_train':       len(X_train),
            'n_test':        len(X_test),
        }

    _TRAINED = True
    return _MODELS, _METRICS


# ── Inference ──────────────────────────────────────────────────
def predict_products(customer):
    """Return ranked product list with ML buy-probabilities."""
    models, _ = train_all_models()
    seg = customer.get('segment', 'Standard')

    feat = {
        'income':           float(customer.get('monthly_income', 5000)),
        'credit_score':     float(customer.get('credit_score', 650)),
        'balance':          float(customer.get('balance', 10000)),
        'tenure':           float(customer.get('tenure_months', 12)),
        'num_products':     float(customer.get('num_products_owned', 1)),
        'engagement_score': float(customer.get('engagement_score', 30)),
        'annual_spend':     float(customer.get('total_spend_12m',
                                 customer.get('monthly_income', 5000) * 10)),
        'segment_enc':      float(SEGMENT_MAP.get(seg, 0)),
        'behavior_score':   float(customer.get('behavior_score', 40)),
        'app_logins':       float(customer.get('app_logins', 10)),
        'has_credit_card':  float(customer.get('has_credit_card', 0)),
        'has_loan':         float(customer.get('has_loan', 0)),
    }
    X     = pd.DataFrame([feat])
    owned = set(str(customer.get('products_owned', '')).split('|'))

    results = []
    for product in PRODUCTS:
        if product in owned:
            continue
        prob = float(models[product].predict_proba(X)[0][1])
        results.append({
            'product_id':     product,
            'product_name':   PRODUCT_NAMES[product],
            'category':       PRODUCT_CATEGORIES[product],
            'ml_probability': round(prob, 4),
            'confidence_pct': round(prob * 100, 1),
        })

    results.sort(key=lambda x: x['ml_probability'], reverse=True)
    return results


# ── Feature Importance ─────────────────────────────────────────
def get_top_features(product_id, top_n=6):
    models, _ = train_all_models()
    model = models.get(product_id)
    if not model:
        return []
    raw    = model.feature_importances_
    ranked = sorted(zip(FEATURES, raw), key=lambda x: x[1], reverse=True)[:top_n]
    top_sum = sum(imp for _, imp in ranked) or 1.0
    return [
        {'feature': f, 'label': FEATURE_LABELS.get(f, f),
         'importance': round(float(imp) / top_sum * 100, 1)}
        for f, imp in ranked
    ]


def get_all_metrics():
    _, metrics = train_all_models()
    return metrics
