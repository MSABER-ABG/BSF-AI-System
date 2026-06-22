"""
BSF AI Banking Intelligence System
Segmentation Visualization Engine
K-Means clustering + PCA + Silhouette on synthetic BSF data
Developed by Accord Business Group (ABG)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────
SEGMENT_ORDER  = ['Standard', 'Silver', 'Gold', 'Platinum', 'VIP']
SEGMENT_COLORS = {
    'Standard': '#7F8C8D', 'Silver':   '#95A5A6',
    'Gold':     '#F39C12', 'Platinum': '#8E44AD', 'VIP': '#E63946',
}
SEGMENT_SIZES = {'Standard': 1480, 'Silver': 1250, 'Gold': 1100, 'Platinum': 760, 'VIP': 410}
N_CUSTOMERS   = 5000
K_OPTIMAL     = 5

FEATURES = [
    'monthly_income', 'credit_score', 'account_balance',
    'tenure_months', 'num_products', 'annual_spend',
    'app_logins', 'engagement_score',
    'late_payment_count', 'txn_frequency',
    'savings_ratio', 'debt_to_income',
]

FEATURE_LABELS = {
    'monthly_income':    'Monthly Income (SAR)',
    'credit_score':      'Credit Score',
    'account_balance':   'Account Balance (SAR)',
    'tenure_months':     'Tenure (months)',
    'num_products':      'Products Owned',
    'annual_spend':      'Annual Spend (SAR)',
    'app_logins':        'App Logins / month',
    'engagement_score':  'Engagement Score',
    'late_payment_count':'Late Payments',
    'txn_frequency':     'Txn Frequency / month',
    'savings_ratio':     'Savings Ratio',
    'debt_to_income':    'Debt-to-Income Ratio',
}

# ── Global cache ───────────────────────────────────────────────
_cache = {}


# ── Data Generation ────────────────────────────────────────────
def _generate_customers(n=N_CUSTOMERS):
    rng  = np.random.default_rng(42)
    segs = list(SEGMENT_SIZES.keys())
    sizes = list(SEGMENT_SIZES.values())
    segment_labels = np.repeat(segs, sizes)[:n]

    income_rng  = {'Standard':(3000,7000),   'Silver':(7000,12000),
                   'Gold':(12000,20000), 'Platinum':(20000,35000), 'VIP':(35000,80000)}
    balance_rng = {'Standard':(5000,20000),  'Silver':(15000,50000),
                   'Gold':(40000,100000),'Platinum':(80000,200000),'VIP':(150000,600000)}
    score_rng   = {'Standard':(540,680),     'Silver':(620,720),
                   'Gold':(680,780),    'Platinum':(720,820),     'VIP':(780,900)}
    tenure_rng  = {'Standard':(1,36),   'Silver':(6,72),
                   'Gold':(12,120), 'Platinum':(24,168), 'VIP':(36,180)}
    prod_rng    = {'Standard':(1,2), 'Silver':(1,3), 'Gold':(2,4),
                   'Platinum':(3,5), 'VIP':(4,6)}
    login_rng   = {'Standard':(0,15), 'Silver':(5,25), 'Gold':(15,45),
                   'Platinum':(30,80), 'VIP':(60,120)}

    rows = []
    for seg in segment_labels:
        inc   = rng.uniform(*income_rng[seg])
        bal   = rng.uniform(*balance_rng[seg])
        cs    = rng.uniform(*score_rng[seg])
        ten   = int(rng.integers(*tenure_rng[seg]))
        npro  = int(rng.integers(*prod_rng[seg]))
        log   = int(rng.integers(*login_rng[seg]))
        spend = inc * rng.uniform(0.5, 2.8)
        txn   = int(rng.integers(2, 50))
        late  = max(0, int(rng.normal(0.5 if seg in ['Standard','Silver'] else 0.1, 0.8)))
        sav_r = float(np.clip(rng.normal(0.10 if seg == 'Standard' else 0.25, 0.08), 0, 0.6))
        dti   = float(np.clip(rng.normal(0.45 if seg == 'Standard' else 0.20, 0.10), 0, 0.7))
        eng   = float(np.clip(log * 0.3 + npro * 8 + rng.normal(0, 5), 0, 100))
        rows.append({
            'segment':           seg,
            'monthly_income':    round(inc, 0),
            'credit_score':      round(cs, 0),
            'account_balance':   round(bal, 0),
            'tenure_months':     ten,
            'num_products':      npro,
            'annual_spend':      round(spend, 0),
            'app_logins':        log,
            'engagement_score':  round(eng, 1),
            'late_payment_count': late,
            'txn_frequency':     txn,
            'savings_ratio':     round(sav_r, 3),
            'debt_to_income':    round(dti, 3),
        })
    return pd.DataFrame(rows)


def _label_to_seg(labels, df):
    """Map cluster IDs to BSF segment names by ascending median income."""
    median_inc = {c: df[df['cluster'] == c]['monthly_income'].median()
                  for c in range(K_OPTIMAL)}
    sorted_clusters = sorted(median_inc, key=median_inc.get)
    return {c: SEGMENT_ORDER[i] for i, c in enumerate(sorted_clusters)}


# ── Main entry point ───────────────────────────────────────────
def get_segmentation_data():
    """Run K-Means + PCA + Silhouette and return all results (cached)."""
    if _cache:
        return _cache

    df = _generate_customers()

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    km = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=20, max_iter=300)
    df['cluster'] = km.fit_predict(X_scaled)
    c2s = _label_to_seg(km.labels_, df)
    df['segment_pred'] = df['cluster'].map(c2s)

    # PCA 2D + 3D
    pca2     = PCA(n_components=2, random_state=42)
    pca3     = PCA(n_components=3, random_state=42)
    coords2  = pca2.fit_transform(X_scaled)
    coords3  = pca3.fit_transform(X_scaled)
    df['pca1']    = coords2[:, 0]
    df['pca2']    = coords2[:, 1]
    df['pca3d_1'] = coords3[:, 0]
    df['pca3d_2'] = coords3[:, 1]
    df['pca3d_3'] = coords3[:, 2]

    pca_explained = pca2.explained_variance_ratio_ * 100

    # PCA loadings (all components)
    pca_full = PCA(n_components=len(FEATURES), random_state=42)
    pca_full.fit(X_scaled)
    loadings = pd.DataFrame(
        pca_full.components_[:3].T,
        index=FEATURES,
        columns=['PC1', 'PC2', 'PC3'],
    )
    pca_all_evr = pca_full.explained_variance_ratio_ * 100

    # Silhouette
    sil_score_val = silhouette_score(X_scaled, km.labels_)
    df['silhouette'] = silhouette_samples(X_scaled, km.labels_)

    # Elbow + Silhouette curve
    k_range, inertias, sil_scores_k = list(range(2, 11)), [], []
    for k in k_range:
        km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl  = km_k.fit_predict(X_scaled)
        inertias.append(km_k.inertia_)
        sil_scores_k.append(silhouette_score(X_scaled, lbl))

    # Cluster centers
    centers_orig = scaler.inverse_transform(km.cluster_centers_)
    centers_df   = pd.DataFrame(centers_orig, columns=FEATURES)
    centers_df['segment'] = [c2s[i] for i in range(K_OPTIMAL)]
    centers_df = centers_df.set_index('segment').reindex(SEGMENT_ORDER)

    # Normalize centers 0-100 per feature
    centers_norm = centers_df.copy()
    for col in FEATURES:
        mn, mx = centers_df[col].min(), centers_df[col].max()
        rng_val = (mx - mn) if mx != mn else 1.0
        centers_norm[col] = (centers_df[col] - mn) / rng_val * 100

    _cache.update({
        'df':            df,
        'X_scaled':      X_scaled,
        'centers_df':    centers_df,
        'centers_norm':  centers_norm,
        'pca_explained': pca_explained,
        'pca_loadings':  loadings,
        'pca_all_evr':   pca_all_evr,
        'sil_score':     round(sil_score_val, 4),
        'k_range':       k_range,
        'inertias':      inertias,
        'sil_scores_k':  sil_scores_k,
        'n_iter':        int(km.n_iter_),
    })
    return _cache
