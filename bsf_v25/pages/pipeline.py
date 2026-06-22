import streamlit as st
import plotly.graph_objects as go
from utils.styling import *

PIPELINE_STEPS = [
    {
        "icon": "📦", "label": "Raw Data", "sub": "5K customers\n300K txns",
        "status": "done", "module": "Setup", "color": ABG_BLUE,
        "title": "Synthetic Data Generation",
        "why": "No real BSF data is available at this stage, so we generate a statistically representative synthetic dataset that mirrors real Saudi banking demographics, income distributions, and transaction behaviors. This allows the full pipeline to be built, tested, and demonstrated without privacy concerns.",
        "how": [
            "5,000 customer profiles generated using Saudi demographic distributions (age, city, employment, nationality)",
            "Income ranges calibrated to Saudi labor market: Government avg SAR 12K, Private SAR 9K, Self-Employed SAR 15K",
            "Transaction history: ~60 txns/customer/year across 6 types (Purchase, Transfer, Bill Payment, ATM, Online, Salary)",
            "Products assigned based on income + credit score eligibility gates per product",
            "Faker library with Arabic locale ensures realistic names, cities, and dates",
        ],
        "output": "5,000 customer rows + ~300,000 transaction records saved to Google Drive",
        "metrics": {"Customers": "5,000", "Transactions": "~300K", "Raw Features": "19", "Products": "5"},
    },
    {
        "icon": "⚙️", "label": "Feature Eng.", "sub": "23 features",
        "status": "done", "module": "Module 1", "color": ABG_BLUE,
        "title": "Feature Engineering — 23 Features Across 4 Groups",
        "why": "Raw customer data alone cannot capture behavioral patterns. Feature engineering transforms raw fields into meaningful signals that reveal how a customer truly behaves — not just who they are on paper. RFM is a proven framework used by top banks worldwide to quantify customer value and engagement.",
        "how": [
            "RFM (7 features): Recency = days since last txn, Frequency = total txns 12M, Monetary = total spend + avg/max/std + last-3M spend",
            "Financial (5 features): monthly_income, account_balance, credit_score, tenure_months, num_products_owned",
            "Product flags (3 features): has_credit_card, has_loan, has_investment — binary signals for cross-sell gaps",
            "Behavioral (3 features): app_logins_monthly, txn_type_diversity, engagement_score (weighted composite)",
            "Derived ratios (3 features): balance_to_income_ratio, spend_to_income_ratio, digital_ratio",
            "Engagement Score = app_logins×0.3 + txn_freq/12×0.4 + n_products×2×0.3",
        ],
        "output": "23-feature matrix per customer, ready for normalization and modeling",
        "metrics": {"RFM Features": "7", "Financial": "5", "Behavioral": "3", "Derived Ratios": "3"},
    },
    {
        "icon": "📏", "label": "Normalizer", "sub": "StandardScaler",
        "status": "done", "module": "Module 1", "color": ABG_BLUE,
        "title": "Feature Normalization — StandardScaler",
        "why": "K-Means computes Euclidean distance between points. Without normalization, large-scale features (account_balance in SAR 100,000s) dominate over small-scale ones (num_products 1–5). This would cause the algorithm to cluster purely by balance, ignoring behavioral signals entirely.",
        "how": [
            "StandardScaler formula: z = (x − μ) / σ applied to all 23 features",
            "Result: every feature has mean=0 and standard deviation=1 after transformation",
            "All 23 features contribute equally to K-Means distance calculations",
            "Scaler is fitted on training data only, then applied identically to new customers",
            "Saved as feature_scaler.pkl — loaded at inference time for real-time prediction",
        ],
        "output": "Normalized 23-feature matrix (mean=0, std=1) — consistent scale across all dimensions",
        "metrics": {"Formula": "z=(x−μ)/σ", "Mean After": "0.000", "Std After": "1.000", "Saved As": "scaler.pkl"},
    },
    {
        "icon": "🎯", "label": "K-Means", "sub": "K=5 segments",
        "status": "done", "module": "Module 1", "color": ABG_BLUE,
        "title": "Customer Segmentation — K-Means Clustering (K=5)",
        "why": "Segmentation is the foundation of personalized banking. Instead of treating all 5,000 customers identically, K-Means groups them into 5 natural clusters based on their true financial and behavioral profiles. This enables targeted product recommendations, channel strategies, and credit decisions — multiplying campaign ROI.",
        "how": [
            "K selected using 3 metrics: Elbow Method (inertia drop), Silhouette Score (separation quality), Davies-Bouldin Index (compactness)",
            "K-Means++ initialization: smart centroid seeding to avoid poor local minima",
            "n_init=20: runs 20 times with different seeds, picks the best result for stability",
            "Clusters mapped to segment labels by median income rank: lowest→Standard, highest→VIP",
            "Silhouette Score > 0.30 confirms clusters are well-separated and meaningful",
            "NOTE: Labels (Standard/Silver/Gold/Platinum/VIP) are placeholders — BSF business team must define real specs",
        ],
        "output": "5 segments: Standard (30%) · Silver (25%) · Gold (22%) · Platinum (15%) · VIP (8%)",
        "metrics": {"K": "5", "n_init": "20", "Silhouette": ">0.30", "Algorithm": "k-means++"},
    },
    {
        "icon": "🏆", "label": "NBA Engine", "sub": "Top-2 products",
        "status": "done", "module": "Module 1", "color": ABG_BLUE,
        "title": "Next-Best Action (NBA) Engine — Weighted Scoring + Affinity Boost",
        "why": "Generic product offers fail. The NBA engine ensures every customer receives only products they are genuinely eligible for, ranked by how well the product fits their specific profile. Reason codes make recommendations explainable — critical for customer trust and regulatory compliance in Saudi banking.",
        "how": [
            "Eligibility filter: income ≥ product minimum AND credit score ≥ product minimum — ineligible products excluded entirely",
            "4-factor weighted score: Income alignment 30% + Credit fit 25% + Behavioral propensity 25% + Segment fit 20%",
            "Affinity Boost: +15% if no credit card, +10% Travel Card with high spend, +12% loan with no existing loan, −15% penalty for loyalty/FX when core products missing",
            "Reason codes: product-specific first (e.g. 'Annual spend SAR 174K qualifies for travel rewards'), then profile context",
            "Returns top-2 products ranked by final match score 0–100%",
        ],
        "output": "Top-2 BSF product recommendations per customer with match score, limit, and 5 reason codes",
        "metrics": {"Products Scored": "5", "Top-N": "2", "Score Range": "30–99%", "Reason Codes": "5 per rec"},
    },
    {
        "icon": "📡", "label": "Channel", "sub": "Primary+Backup",
        "status": "done", "module": "Module 1", "color": ABG_BLUE,
        "title": "Best Channel Selector — 7 Channels, Segment-Based Priority",
        "why": "Reaching the right customer through the wrong channel wastes budget and reduces conversion. VIP customers expect RM contact; Standard customers respond better to SMS. Channel selection closes the loop between recommendation and delivery — ensuring the right product reaches the customer most effectively.",
        "how": [
            "7 channels: RM, Branch Agent, Push Notification, WhatsApp, SMS, Avatar, Call Center",
            "Segment priority queues: Standard→SMS, Silver→WhatsApp, Gold→Push, Platinum/VIP→RM",
            "Product-type overrides: Home Finance→RM, Travel Card→Push, Qassitha→SMS always",
            "Digital override: engagement_score > 15 AND preferred = Mobile App → Push Notification",
            "Primary and backup channels both assigned — ensures delivery even if primary fails",
        ],
        "output": "Primary + backup channel per customer with override reason logged",
        "metrics": {"Channels": "7", "Overrides": "5 products", "Eng. Threshold": ">15", "Coverage": "100%"},
    },
    {
        "icon": "📱", "label": "Clickstream", "sub": "Sessions·Clicks",
        "status": "done", "module": "Module 2", "color": ABG_GOLD,
        "title": "Digital Behavioral Data — Clickstream Generation",
        "why": "Transaction data tells us what customers do. Clickstream tells us what they intend to do. Page visits, click counts, session duration, and drop-off pages reveal intent, frustration, and engagement patterns invisible in financial data. This powers anomaly detection and drift analysis.",
        "how": [
            "Sessions scaled by segment: Standard×0.5, Silver×0.8, Gold×1.0, Platinum×1.3, VIP×1.6",
            "Per session: duration (log-normal), pages visited (Poisson), clicks, scrolls, device type (70% Mobile)",
            "Transaction completion probability = 0.3 × segment_multiplier per session",
            "Drop-off page recorded when session ends without transaction — journey friction analysis",
            "3 months of history generated: ~14–15 sessions/customer average",
        ],
        "output": "~72,000 session records with clicks, pages, duration, device, drop-off per customer",
        "metrics": {"Sessions": "~72K", "Avg/Customer": "~14", "Months": "3", "Features": "15"},
    },
    {
        "icon": "🚨", "label": "Anomaly", "sub": "IsoForest+Z-Score",
        "status": "done", "module": "Module 2", "color": ABG_GOLD,
        "title": "Anomaly Detection — Isolation Forest + Z-Score (Two-Layer)",
        "why": "A customer who suddenly increases sessions 200% or abandons 80% of sessions is exhibiting abnormal behavior — potentially churn risk, fraud, or high purchase intent. Two-layer detection catches different anomaly types; neither method alone is sufficient for comprehensive coverage.",
        "how": [
            "Layer 1 — Isolation Forest (multivariate): 200 random trees, anomalies isolated with fewer splits, contamination=5%",
            "Layer 2 — Z-Score (univariate): flags customers where any single feature |z| > 3.0 std deviations from mean",
            "Combined flag: customer flagged if EITHER method triggers — maximizes recall",
            "Anomaly typed: Churn Risk (sessions drop >50% MoM), Spike (sessions +100%), Friction (drop-off >80%), Fraud Signal",
            "All anomalies routed to Analysis Agent — no automatic action without admin approval",
        ],
        "output": "~250 flagged customers with anomaly type, severity, and recommended action pending approval",
        "metrics": {"Method 1": "Isolation Forest", "Method 2": "Z-Score |z|>3", "Flagged": "~5%", "Types": "5"},
    },
    {
        "icon": "🤖", "label": "Ensemble", "sub": "XGB+RF+GB",
        "status": "done", "module": "Module 3", "color": ABG_GREEN,
        "title": "Risk Classification — Ensemble (XGBoost + Random Forest + Gradient Boosting)",
        "why": "A single model can be biased. The ensemble combines three algorithms: XGBoost excels at non-linear patterns, Random Forest provides robustness through bagging, Gradient Boosting captures sequential patterns. Dynamic weights ensure the final prediction reflects the best-performing combination.",
        "how": [
            "XGBoost: n=300, depth=6, lr=0.05, subsample=0.8 — strong on tabular financial data",
            "Random Forest: n=300, depth=10, class_weight='balanced' — handles imbalanced risk classes",
            "Gradient Boosting: n=300, depth=6, lr=0.05 — captures complex feature interactions",
            "Dynamic weights: each model's weight = test_accuracy / sum_of_accuracies × 10",
            "Soft voting: averages predicted probabilities → smoother, more calibrated predictions",
            "5-fold cross-validation confirms ~84% accuracy is stable and not overfitted",
        ],
        "output": "Risk category (5 classes) + risk score 0–100 + confidence score per customer",
        "metrics": {"XGBoost": "82.2%", "Random Forest": "79.0%", "Grad. Boosting": "81.5%", "Ensemble": "84.1%"},
    },
    {
        "icon": "💳", "label": "Credit", "sub": "Limit+Rate+Loan",
        "status": "done", "module": "Module 3", "color": ABG_GREEN,
        "title": "Credit Decision Engine — Limit + Loan + SAIBOR-Linked Rate",
        "why": "Credit decisions must be both fair and profitable. Too generous → higher default risk. Too conservative → lost revenue. The engine links each decision directly to the ensemble risk score, ensuring Very Low risk customers get competitive rates while Very High risk customers get protected limits.",
        "how": [
            "CC Limit = monthly_income × risk_multiplier (4.0× Very Low → 0.5× Very High), capped at tier maximum",
            "Credit score fine-tune: ±20% adjustment based on distance from score 600",
            "Loan = income × income_multiple (24× → 3×), max SAR 500K, rounded to nearest SAR 5,000",
            "DBR check: monthly installment / income ≤ 33% (SAMA regulation) — loan rejected if DBR exceeded",
            "Interest rate = SAIBOR 3M (5.50%) + risk spread (1.5% → 6.0%) + credit score fine-tune (±0.5%)",
        ],
        "output": "CC limit, card tier, loan amount, monthly installment, DBR ratio, interest rate per customer",
        "metrics": {"SAIBOR Base": "5.50%", "Max CC Limit": "SAR 200K", "Max Loan": "SAR 500K", "DBR Cap": "33%"},
    },
    {
        "icon": "👤", "label": "Human Loop", "sub": "Approval queue",
        "status": "active", "module": "Module 3", "color": ABG_ORANGE,
        "title": "Human-in-the-Loop — Internal Approval Before Any Action",
        "why": "AI models make probabilistic decisions — they can be wrong. In banking, a wrong credit decision has serious financial and regulatory consequences. Human-in-the-Loop ensures no credit offer reaches a customer without review by a qualified credit officer, ensuring SAMA compliance.",
        "how": [
            "Auto-Approve: Very Low + Low risk with no anomaly → standard verification only",
            "Manual Review: Medium risk OR any anomaly flag → credit officer reviews income docs",
            "Senior Review: High + Very High risk → senior credit officer + risk committee sign-off",
            "Decline Recommended: extreme Very High risk → system suggests decline, human makes final call",
            "All decisions logged with timestamp, reviewer ID, and rationale for full audit trail",
        ],
        "output": "Approval queue sorted by priority — no offer sent to any customer until approved",
        "metrics": {"Auto-Approve": "~45%", "Manual": "~35%", "Senior": "~15%", "Decline": "~5%"},
    },
    {
        "icon": "💬", "label": "Chatbot", "sub": "Module 4",
        "status": "pending", "module": "Module 4", "color": ABG_MUTED,
        "title": "BSF AI Chatbot — Personalized Customer Communication (Coming Next)",
        "why": "The pipeline produces perfect recommendations but they only create value when the customer understands and acts on them. The Chatbot is the final delivery layer: it transforms NBA output, reason codes, and credit decisions into a personalized conversational message that explains recommendations in simple terms.",
        "how": [
            "Welcome message generated dynamically from: name + segment + top-2 NBA products + reason codes",
            "RAG (Retrieval-Augmented Generation): BSF product knowledge base queried for detailed answers",
            "FAQ handling: 'Why was this recommended?', 'What are the benefits?', 'How do I apply?'",
            "Multi-channel: Push Notification opener → WhatsApp conversation → Branch handoff",
            "Human escalation: complex queries escalated to RM with full context summary",
        ],
        "output": "Personalized welcome message + FAQ responses + product comparison + application link",
        "metrics": {"Status": "Coming Next", "Delivery": "Push/WhatsApp/Branch", "RAG": "BSF Product KB", "Languages": "AR + EN"},
    },
]

def render():
    # ── Pipeline buttons: use Streamlit 1.58 baseButton-secondary data-testid ──
    st.markdown("""
<style>
/* Pipeline buttons: opacity:0 — card overlays via pointer-events:none */
div[data-testid="stHorizontalBlock"] button[data-testid="baseButton-secondary"] {
    opacity: 0 !important;
    height: 94px !important;
    min-height: 94px !important;
    width: 100% !important;
    cursor: pointer !important;
    border-radius: 10px !important;
    display: block !important;
}
</style>
""", unsafe_allow_html=True)

    # Session state for selected step
    if 'selected_step' not in st.session_state:
        st.session_state.selected_step = 0

    st.markdown(page_header(
        "Full AI Pipeline",
        "End-to-end flow: Raw Data → Segmentation → NBA → Behavior → Credit Decision → Chatbot",
        "🔄"
    ), unsafe_allow_html=True)

    # ── Pipeline Overview Cards ───────────────────────────────
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 4px;">'
        f'Pipeline Steps <span style="font-size:12px;font-weight:400;color:{ABG_MUTED};">'
        f'— click any card to explore</span></h3>',
        unsafe_allow_html=True
    )

    # ── Row 1: Setup + Module 1 (steps 0-5) ─────────────────
    band_r1 = st.columns([1, 5])
    for col, (name, color, label) in zip(band_r1, [
        ("Setup",    ABG_BLUE, "Data Prep"),
        ("Module 1", ABG_BLUE, "Segmentation & NBA"),
    ]):
        col.markdown(f"""
        <div style="text-align:center;padding:5px 4px;border-radius:6px;
        background:{color}12;border:1px solid {color}30;margin-bottom:8px;">
          <div style="font-size:10px;font-weight:700;color:{color};">{name}</div>
          <div style="font-size:9px;color:{ABG_MUTED};">{label}</div>
        </div>""", unsafe_allow_html=True)

    row1_cols = st.columns(6)
    for i in range(6):
        step   = PIPELINE_STEPS[i]
        color  = step['color']; status = step['status']
        is_sel = st.session_state.selected_step == i
        border = f"border:2.5px solid {color};" if is_sel else (
                 f"border:1.5px solid {color}66;" if status in ['done','active'] else
                 f"border:1.5px solid {ABG_BORDER};")
        bg     = f"background:{color}15;" if is_sel else (
                 f"background:{color}07;" if status in ['done','active'] else
                 f"background:{ABG_WHITE};")
        shadow = f"box-shadow:0 4px 14px {color}30;" if is_sel else ""
        badge  = "✅" if status=='done' else ("⏳" if status=='active' else "🔜")
        sub_html = '<br>'.join([
            f'<span style="font-size:9px;color:{ABG_MUTED};line-height:1.1;">{l}</span>'
            for l in step['sub'].split('\n')
        ])
        with row1_cols[i]:
            st.markdown(f"""
            <div style="{border}{bg}{shadow}border-radius:10px;padding:10px 5px;
            text-align:center;height:100px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;gap:2px;margin-bottom:4px;">
              <div style="font-size:22px;line-height:1;">{step['icon']}</div>
              <div style="font-size:11px;font-weight:{'800' if is_sel else '600'};
              color:{color if is_sel else ABG_DARK};line-height:1.2;margin-top:2px;">
                {step['label']}</div>
              <div style="line-height:1.2;">{sub_html}</div>
              <div style="font-size:10px;margin-top:2px;">{badge}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # ── Row 2: Module 2 + Module 3 + Module 4 (steps 6-11) ──
    band_r2 = st.columns([2, 3, 1])
    for col, (name, color, label) in zip(band_r2, [
        ("Module 2", ABG_GOLD,  "Behavior"),
        ("Module 3", ABG_GREEN, "Credit"),
        ("Module 4", ABG_MUTED, "Chatbot"),
    ]):
        col.markdown(f"""
        <div style="text-align:center;padding:5px 4px;border-radius:6px;
        background:{color}12;border:1px solid {color}30;margin-bottom:8px;">
          <div style="font-size:10px;font-weight:700;color:{color};">{name}</div>
          <div style="font-size:9px;color:{ABG_MUTED};">{label}</div>
        </div>""", unsafe_allow_html=True)

    row2_cols = st.columns(6)
    for i in range(6, 12):
        step   = PIPELINE_STEPS[i]
        color  = step['color']; status = step['status']
        is_sel = st.session_state.selected_step == i
        border = f"border:2.5px solid {color};" if is_sel else (
                 f"border:1.5px solid {color}66;" if status in ['done','active'] else
                 f"border:1.5px solid {ABG_BORDER};")
        bg     = f"background:{color}15;" if is_sel else (
                 f"background:{color}07;" if status in ['done','active'] else
                 f"background:{ABG_WHITE};")
        shadow = f"box-shadow:0 4px 14px {color}30;" if is_sel else ""
        badge  = "✅" if status=='done' else ("⏳" if status=='active' else "🔜")
        sub_html = '<br>'.join([
            f'<span style="font-size:9px;color:{ABG_MUTED};line-height:1.1;">{l}</span>'
            for l in step['sub'].split('\n')
        ])
        with row2_cols[i - 6]:
            st.markdown(f"""
            <div style="{border}{bg}{shadow}border-radius:10px;padding:10px 5px;
            text-align:center;height:100px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;gap:2px;margin-bottom:4px;">
              <div style="font-size:22px;line-height:1;">{step['icon']}</div>
              <div style="font-size:11px;font-weight:{'800' if is_sel else '600'};
              color:{color if is_sel else ABG_DARK};line-height:1.2;margin-top:2px;">
                {step['label']}</div>
              <div style="line-height:1.2;">{sub_html}</div>
              <div style="font-size:10px;margin-top:2px;">{badge}</div>
            </div>""", unsafe_allow_html=True)

    # Step selector — clean pills row (no blue buttons)
    step_labels = [f"{s['icon']} {s['label']}" for s in PIPELINE_STEPS]
    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 4px;">
    {"".join([
        f'<div style="padding:5px 14px;border-radius:20px;font-size:11px;font-weight:{"700" if j==st.session_state.selected_step else "500"};cursor:default;'
        f'background:{""+PIPELINE_STEPS[j]["color"]+"20" if j==st.session_state.selected_step else "#F0F0F6"};'
        f'color:{""+PIPELINE_STEPS[j]["color"] if j==st.session_state.selected_step else ABG_MUTED};'
        f'border:1.5px solid {""+PIPELINE_STEPS[j]["color"]+"44" if j==st.session_state.selected_step else "transparent"};'
        f'">{ step_labels[j] }</div>'
        for j in range(len(PIPELINE_STEPS))
    ])}
    </div>""", unsafe_allow_html=True)

    sel_idx = st.selectbox(
        "Select pipeline step:",
        options=list(range(len(PIPELINE_STEPS))),
        format_func=lambda x: f"{PIPELINE_STEPS[x]['icon']} {PIPELINE_STEPS[x]['label']}",
        index=st.session_state.selected_step,
        key="pipeline_step_select",
        label_visibility="collapsed",
    )
    if sel_idx != st.session_state.selected_step:
        st.session_state.selected_step = sel_idx
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Step Detail ───────────────────────────────────────────
    idx   = st.session_state.selected_step
    step  = PIPELINE_STEPS[idx]
    color = step['color']

    status_labels = {'done': '✅ Completed', 'active': '⏳ In Progress', 'pending': '🔜 Coming Next'}
    status_colors = {'done': ABG_GREEN, 'active': ABG_ORANGE, 'pending': ABG_MUTED}

    # Step nav pills
    nav_html = ''.join([
        f'<span style="display:inline-flex;align-items:center;gap:4px;padding:4px 10px;'
        f'border-radius:20px;font-size:10px;font-weight:{"700" if j==idx else "500"};'
        f'background:{PIPELINE_STEPS[j]["color"]}{"20" if j==idx else "08"};'
        f'color:{PIPELINE_STEPS[j]["color"] if j==idx else ABG_MUTED};'
        f'border:1px solid {PIPELINE_STEPS[j]["color"]}{"44" if j==idx else "20"};'
        f'margin:2px;">'
        f'{PIPELINE_STEPS[j]["icon"]} {PIPELINE_STEPS[j]["label"]}</span>'
        for j in range(len(PIPELINE_STEPS))
    ])
    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:16px;
    padding:12px;background:{ABG_WHITE};border:1px solid {ABG_BORDER};border-radius:10px;">
    {nav_html}
    </div>""", unsafe_allow_html=True)

    # Step header card
    st.markdown(f"""
    <div style="background:{ABG_WHITE};border:1.5px solid {color}44;border-radius:14px;
    padding:22px 26px;border-left:5px solid {color};margin-bottom:20px;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;">
        <div>
          <div style="font-size:10px;font-weight:700;color:{color};text-transform:uppercase;
          letter-spacing:0.09em;margin-bottom:5px;">{step['module']}</div>
          <div style="font-size:20px;font-weight:800;color:{ABG_DARK};">
            {step['icon']} {step['title']}
          </div>
        </div>
        <span style="padding:5px 16px;border-radius:20px;font-size:12px;font-weight:700;
        background:{status_colors[step['status']]}20;color:{status_colors[step['status']]};">
          {status_labels[step['status']]}
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Three columns
    dc1, dc2, dc3 = st.columns([2, 3, 2])

    with dc1:
        st.markdown(f"""
        <div style="background:{color}08;border:1.5px solid {color}30;
        border-radius:12px;padding:20px;">
          <div style="font-size:13px;font-weight:800;color:{color};margin-bottom:12px;">
            💡 Why This Step Matters
          </div>
          <div style="font-size:13px;color:{ABG_DARK};line-height:1.8;">
            {step['why']}
          </div>
        </div>""", unsafe_allow_html=True)

    with dc2:
        how_items = ''.join([
            f'<div style="display:flex;gap:10px;padding:10px 0;border-bottom:1px solid {ABG_BORDER};">'
            f'<div style="min-width:22px;height:22px;border-radius:50%;background:{color};'
            f'color:white;font-size:11px;font-weight:700;display:flex;align-items:center;'
            f'justify-content:center;flex-shrink:0;margin-top:2px;">{j+1}</div>'
            f'<div style="font-size:12px;color:{ABG_DARK};line-height:1.7;">{h}</div>'
            f'</div>'
            for j, h in enumerate(step['how'])
        ])
        st.markdown(f"""
        <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
        border-radius:12px;padding:20px;">
          <div style="font-size:13px;font-weight:800;color:{ABG_DARK};margin-bottom:12px;">
            ⚙️ How It Works
          </div>
          {how_items}
          <div style="margin-top:14px;padding:10px 14px;border-radius:8px;
          background:{color}0d;border-left:3px solid {color};font-size:12px;color:{ABG_DARK};">
            <strong>Output:</strong> {step['output']}
          </div>
        </div>""", unsafe_allow_html=True)

    with dc3:
        metrics_rows = ''.join([
            f'<div style="padding:12px 14px;border-bottom:1px solid {ABG_BORDER};'
            f'display:flex;justify-content:space-between;align-items:center;">'
            f'<div style="font-size:11px;color:{ABG_MUTED};font-weight:600;">{k}</div>'
            f'<div style="font-size:13px;font-weight:800;color:{color};">{v}</div>'
            f'</div>'
            for k, v in step['metrics'].items()
        ])
        st.markdown(f"""
        <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
        border-radius:12px;overflow:hidden;">
          <div style="padding:14px 16px;background:{color}0d;border-bottom:1px solid {color}22;">
            <div style="font-size:13px;font-weight:800;color:{color};">📊 Key Metrics</div>
          </div>
          {metrics_rows}
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 16px;">'
        f'📊 Pipeline Performance</h3>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        models     = ['XGBoost', 'Random Forest', 'Grad. Boosting', 'Ensemble ✅']
        accs       = [82.2, 79.0, 81.5, 84.1]
        bar_colors = [ABG_BLUE, ABG_GOLD, ABG_GREEN, ABG_PURPLE]
        fig = go.Figure(go.Bar(
            x=models, y=accs,
            marker_color=bar_colors, marker_line_width=0,
            text=[f"{v:.1f}%" for v in accs], textposition='outside',
        ))
        fig.update_layout(
            title="Module 3 — Model Accuracy Comparison",
            paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
            font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
            margin=dict(t=40, b=20, l=10, r=10), height=280,
            showlegend=False,
            yaxis=dict(range=[74, 88], gridcolor=ABG_BORDER, ticksuffix="%"),
            xaxis=dict(gridcolor=ABG_BORDER),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        segs       = ['Standard', 'Silver', 'Gold', 'Platinum', 'VIP']
        counts     = [1480, 1250, 1100, 760, 410]
        seg_colors = ['#7F8C8D', '#BDC3C7', '#F1C40F', '#8E44AD', '#E74C3C']
        fig2 = go.Figure(go.Pie(
            labels=segs, values=counts,
            marker=dict(colors=seg_colors, line=dict(color='white', width=2)),
            hole=0.5, textfont=dict(size=11),
        ))
        fig2.update_layout(
            title="Module 1 — Segment Distribution",
            paper_bgcolor=ABG_WHITE,
            font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
            margin=dict(t=40, b=10, l=10, r=10), height=280,
            legend=dict(font=dict(size=11)),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Feature importance chart
    features = [
        'credit_score_norm', 'payment_discipline', 'income_stability',
        'balance_adequacy', 'behavior_score_norm', 'txn_consistency',
        'debt_to_income', 'digital_stability', 'anomaly_risk',
        'churn_risk', 'product_depth', 'late_payment_count',
        'income_tier', 'credit_utilization', 'session_engagement',
    ]
    importances = [0.22, 0.18, 0.13, 0.10, 0.09, 0.07,
                   0.06, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01]
    feat_colors = [
        ABG_GREEN  if i < 5  else
        ABG_BLUE   if i < 9  else
        ABG_ORANGE
        for i in range(len(features))
    ]
    fig3 = go.Figure(go.Bar(
        x=importances, y=features, orientation='h',
        marker_color=feat_colors, marker_line_width=0,
        text=[f"{v:.0%}" for v in importances], textposition='outside',
    ))
    fig3.update_layout(
        title="Module 3 — Ensemble Feature Importance (15 Risk Features)",
        paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
        font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
        margin=dict(t=40, b=20, l=180, r=80), height=420,
        showlegend=False,
        xaxis=dict(gridcolor=ABG_BORDER, tickformat=".0%"),
        yaxis=dict(gridcolor=ABG_BORDER, tickfont=dict(size=11)),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Tech Stack ────────────────────────────────────────────
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 16px;">'
        f'🛠️ Technology Stack</h3>',
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)
    stacks = [
        (ABG_BLUE, "Module 1 Models", [
            ("K-Means (K=5)",    "k-means++ · n_init=20 · max_iter=500"),
            ("PCA (2D)",         "Visualization only · explained variance"),
            ("StandardScaler",   "mean=0, std=1 · saved as .pkl"),
            ("Weighted Engine",  "NBA: 30+25+25+20% + Affinity Boost"),
            ("Channel Selector", "7 channels · segment + product override"),
        ]),
        (ABG_GOLD, "Module 2 Models", [
            ("Isolation Forest", "n_estimators=200 · contamination=5%"),
            ("Z-Score",          "scipy.stats · threshold |z| > 3.0"),
            ("MinMaxScaler",     "Behavior score 0–100 · 7 features"),
            ("Drift Detector",   "Z-score vs segment mean · ±1.5σ"),
            ("Analysis Agent",   "Rule-based · admin approval required"),
        ]),
        (ABG_GREEN, "Module 3 Models", [
            ("XGBoost",          "n=300 · depth=6 · lr=0.05 · weight≈3"),
            ("Random Forest",    "n=300 · depth=10 · balanced · weight≈2"),
            ("Grad. Boosting",   "n=300 · depth=6 · lr=0.05 · weight≈2"),
            ("Soft Voting",      "Dynamic weights from CV accuracy"),
            ("Human-in-Loop",    "Auto / Manual / Senior Review routing"),
        ]),
    ]
    for col, (color, title, items) in zip([c1, c2, c3], stacks):
        rows = ''.join([
            f'<div style="padding:10px 0;border-bottom:1px solid {ABG_BORDER};">'
            f'<div style="font-size:12px;font-weight:700;color:{ABG_DARK};">{name}</div>'
            f'<div style="font-size:11px;color:{ABG_MUTED};margin-top:2px;">{detail}</div>'
            f'</div>'
            for name, detail in items
        ])
        with col:
            col.markdown(f"""
            <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
            border-radius:12px;overflow:hidden;">
              <div style="padding:14px 16px;border-bottom:1px solid {ABG_BORDER};
              font-size:13px;font-weight:800;color:{color};background:{color}08;">
                {title}
              </div>
              <div style="padding:0 16px;">{rows}</div>
            </div>""", unsafe_allow_html=True)
