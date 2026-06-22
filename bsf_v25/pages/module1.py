"""
Module 1 — Segmentation & AI-Powered NBA Engine
LightGBM cross-sell models + segment overview
"""
import streamlit as st
import plotly.graph_objects as go
from utils.styling import (
    ABG_BLUE, ABG_DARK, ABG_WHITE, ABG_BORDER, ABG_MUTED,
    ABG_RED, ABG_GREEN, ABG_GOLD, ABG_PURPLE, ABG_ORANGE,
    page_header, info_box, metric_card,
)
from utils.engine import (
    SEGMENT_COLORS as SEG_C,
    generate_dashboard_stats,
    nba_engine_ml,
    PRODUCT_CATALOG,
)
from utils.ml_nba import (
    train_all_models, get_all_metrics, get_top_features,
    PRODUCTS, PRODUCT_NAMES, PRODUCT_CATEGORIES,
)

CAT_COLORS = {'Credit Card': ABG_BLUE, 'Loan': ABG_GREEN, 'Savings': ABG_GOLD}

def _auc_color(auc):
    if auc >= 0.90: return ABG_GREEN
    if auc >= 0.80: return ABG_BLUE
    if auc >= 0.70: return ABG_GOLD
    return ABG_RED

def render():
    st.markdown(page_header(
        "Module 1 — Segmentation & AI NBA Engine",
        "K-Means clustering · LightGBM cross-sell models · Live NBA recommendations",
        "👥",
    ), unsafe_allow_html=True)

    stats = generate_dashboard_stats()
    seg   = stats['segment_dist']

    # ── Segment KPI Cards ─────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, (s, n) in zip([c1,c2,c3,c4,c5], seg.items()):
        pct = n / 5000 * 100
        col.markdown(metric_card(s, str(n), f"{pct:.1f}% of customers",
            SEG_C[s], s, SEG_C[s]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Segment Charts ────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=list(seg.keys()), y=list(seg.values()),
            marker_color=[SEG_C[s] for s in seg.keys()],
            marker_line_width=0,
            text=[f"{v:,}" for v in seg.values()],
            textposition='outside',
        ))
        fig.update_layout(
            title="Segment Distribution", paper_bgcolor=ABG_WHITE,
            plot_bgcolor='#FAFAFA', margin=dict(t=40,b=20,l=10,r=10),
            height=280, font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
            showlegend=False,
            xaxis=dict(gridcolor=ABG_BORDER), yaxis=dict(gridcolor=ABG_BORDER),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cats = ['Income','Balance','Credit Score','Products','Annual Spend','Engagement']
        segs_data = {
            'Gold':     [58, 55, 62, 60, 65, 58],
            'Platinum': [75, 78, 80, 75, 78, 72],
            'VIP':      [95, 97, 92, 88, 90, 95],
        }
        _fill = {
            'Gold':     'rgba(243,156,18,0.13)',
            'Platinum': 'rgba(142,68,173,0.13)',
            'VIP':      'rgba(230,57,70,0.13)',
        }
        fig2 = go.Figure()
        for sn, vals in segs_data.items():
            fig2.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=cats+[cats[0]],
                fill='toself', name=sn,
                line=dict(color=SEG_C[sn], width=2),
                fillcolor=_fill[sn],
            ))
        fig2.update_layout(
            title="Segment Profile Radar",
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100],
                    tickfont=dict(size=9, color=ABG_MUTED)),
                angularaxis=dict(tickfont=dict(size=11, color=ABG_DARK)),
            ),
            paper_bgcolor=ABG_WHITE, margin=dict(t=40,b=20,l=10,r=10),
            height=280, font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
            legend=dict(font=dict(size=11)),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── NBA Score Formula (original) ──────────────────────────
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;'
                f'margin:8px 0 16px;">NBA Match Score Formula</h3>',
                unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    factors = [
        ("30%","Income Alignment","Distance above product's minimum income threshold",ABG_BLUE),
        ("25%","Credit Score Fit","Distance above product's minimum score",ABG_GOLD),
        ("25%","Behavioral Propensity","Engagement score normalized 0–1",ABG_GREEN),
        ("20%","Segment Fit","Customer tier vs product required tier",ABG_PURPLE),
    ]
    for col,(pct,title,desc,color) in zip([c1,c2,c3,c4],factors):
        col.markdown(f"""
        <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
        border-radius:10px;padding:18px;text-align:center;border-top:3px solid {color};">
          <div style="font-size:32px;font-weight:800;color:{color};">{pct}</div>
          <div style="font-size:12px;font-weight:700;color:{ABG_DARK};margin:6px 0 4px;">{title}</div>
          <div style="font-size:11px;color:{ABG_MUTED};">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(info_box(
        "<strong>+ Affinity Boost Layer:</strong> &nbsp;&nbsp;"
        "No credit card → <strong>+15%</strong> for CC products &nbsp;|&nbsp; "
        "Travel card + high spend → <strong>+10%</strong> &nbsp;|&nbsp; "
        "No loan + stable income → <strong>+12%</strong> for loans &nbsp;|&nbsp; "
        "Loyalty/FX when core products missing → <strong>−15%</strong> penalty",
        ABG_BLUE,
    ), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # AI MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 4px;">'
        f'🤖 AI Recommendation: Cross-Sell NBA — [By LightGBM Model]</h3>'
        f'<p style="font-size:12px;color:{ABG_MUTED};margin:0 0 16px;">'
        f'One binary classifier per product · 12,000 synthetic BSF customers · 80/20 train-test split</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Training LightGBM models…"):
        metrics = get_all_metrics()

    prod_icons = {
        'CC_TRAVEL':'✈️','PL_PERSONAL':'💰','CC_CASHBACK':'💳',
        'PL_HOME':'🏠','SA_SAVING':'📈',
    }
    prod_short = {
        'CC_TRAVEL':'World Travel Card','PL_PERSONAL':'Personal Finance',
        'CC_CASHBACK':'Cashback Card','PL_HOME':'Home Finance','SA_SAVING':'Savings Account',
    }

    cols5 = st.columns(5)
    for col, pid in zip(cols5, PRODUCTS):
        m   = metrics[pid]
        auc = m['auc']
        ac  = _auc_color(auc)
        cat = PRODUCT_CATEGORIES[pid]
        cc  = CAT_COLORS.get(cat, ABG_MUTED)
        col.markdown(f"""
        <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
        border-radius:12px;padding:16px 14px;border-top:3px solid {ac};">
          <div style="font-size:18px;margin-bottom:5px;">{prod_icons[pid]}</div>
          <div style="font-size:11px;font-weight:800;color:{ABG_DARK};
          line-height:1.3;margin-bottom:10px;">{prod_short[pid]}</div>
          <div style="font-size:10px;color:{ABG_MUTED};margin-bottom:2px;">ROC-AUC</div>
          <div style="font-size:24px;font-weight:800;color:{ac};">{auc:.3f}</div>
          <div style="background:#F0F0F6;border-radius:3px;height:5px;
          margin:5px 0 10px;overflow:hidden;">
            <div style="height:100%;width:{int(auc*100)}%;background:{ac};border-radius:3px;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
            <span style="font-size:10px;color:{ABG_MUTED};">Avg Precision</span>
            <span style="font-size:10px;font-weight:700;color:{ABG_DARK};">{m['avg_precision']:.3f}</span>
          </div>
          <div style="display:flex;justify-content:space-between;">
            <span style="font-size:10px;color:{ABG_MUTED};">F1 Score</span>
            <span style="font-size:10px;font-weight:700;color:{ABG_DARK};">{m['f1']:.3f}</span>
          </div>
          <div style="margin-top:8px;background:{cc}15;border-radius:4px;
          padding:2px 7px;display:inline-block;">
            <span style="font-size:10px;font-weight:600;color:{cc};">{cat}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # AUC bar + adoption rate
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        aucs   = [metrics[p]['auc'] for p in PRODUCTS]
        labels = [prod_short[p] for p in PRODUCTS]
        fig_a  = go.Figure(go.Bar(
            x=aucs, y=labels, orientation='h',
            marker_color=[_auc_color(a) for a in aucs],
            marker_line_width=0,
            text=[f"{a:.3f}" for a in aucs], textposition='outside',
        ))
        fig_a.update_layout(
            title="ROC-AUC by Product Model", paper_bgcolor=ABG_WHITE,
            plot_bgcolor='#FAFAFA', margin=dict(t=40,b=10,l=10,r=60),
            height=260, font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
            xaxis=dict(range=[0,1.05], gridcolor=ABG_BORDER),
            yaxis=dict(gridcolor=ABG_BORDER),
        )
        st.plotly_chart(fig_a, use_container_width=True)

    with c2:
        pos_rates = [metrics[p]['positive_rate']*100 for p in PRODUCTS]
        fig_p = go.Figure(go.Bar(
            x=[prod_short[p] for p in PRODUCTS], y=pos_rates,
            marker_color=[CAT_COLORS.get(PRODUCT_CATEGORIES[p], ABG_MUTED) for p in PRODUCTS],
            marker_line_width=0,
            text=[f"{r:.1f}%" for r in pos_rates], textposition='outside',
        ))
        fig_p.update_layout(
            title="Product Adoption Rate in Training Data", paper_bgcolor=ABG_WHITE,
            plot_bgcolor='#FAFAFA', margin=dict(t=40,b=10,l=10,r=10),
            height=260, font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
            showlegend=False,
            xaxis=dict(gridcolor=ABG_BORDER, tickangle=-20),
            yaxis=dict(gridcolor=ABG_BORDER, title='Adoption %'),
        )
        st.plotly_chart(fig_p, use_container_width=True)

    # Feature Importance
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;'
                f'margin:0 0 14px;">📊 Feature Importance by Product Model</h3>',
                unsafe_allow_html=True)

    prod_sel = st.selectbox(
        "Select product:",
        options=PRODUCTS,
        format_func=lambda p: f"{prod_icons[p]} {PRODUCT_NAMES[p]}",
        key="m1_feat_select",
    )
    top_feats = get_top_features(prod_sel, top_n=6)
    cc_fi     = CAT_COLORS.get(PRODUCT_CATEGORIES[prod_sel], ABG_BLUE)
    fig_fi    = go.Figure(go.Bar(
        x=[f['importance'] for f in top_feats],
        y=[f['label'] for f in top_feats],
        orientation='h',
        marker_color=cc_fi, marker_line_width=0,
        text=[f"{f['importance']:.1f}%" for f in top_feats],
        textposition='outside',
    ))
    fig_fi.update_layout(
        title=f"Top Features — {PRODUCT_NAMES[prod_sel]}",
        paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
        margin=dict(t=40,b=10,l=10,r=70), height=280,
        font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
        xaxis=dict(range=[0,55], gridcolor=ABG_BORDER, title='Importance (%)'),
        yaxis=dict(gridcolor=ABG_BORDER),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── Live Segmentation Tester ──────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;'
        f'margin:0 0 4px;">👥 Live Segmentation Tester</h3>'
        f'<p style="font-size:12px;color:{ABG_MUTED};margin:0 0 14px;">'
        f'Predict the customer segment in real-time using the K-Means scoring rules</p>',
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ Customer inputs", expanded=True):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            seg_income = st.slider("Monthly Income (SAR)", 1000, 80000, 10000, 500,
                                   key="seg_income")
            seg_score  = st.slider("Credit Score", 300, 900, 680, 10,
                                   key="seg_score")
        with sc2:
            seg_balance = st.slider("Account Balance (SAR)", 1000, 600000, 40000, 1000,
                                    key="seg_balance")
            seg_tenure  = st.slider("Tenure (months)", 1, 180, 24, 1,
                                    key="seg_tenure")
        with sc3:
            seg_products = st.multiselect(
                "Products Owned",
                ['SA_SAVING', 'CC_CASHBACK', 'CC_TRAVEL', 'PL_PERSONAL', 'PL_HOME'],
                default=['SA_SAVING'],
                key="seg_products",
            )

    # Compute segment live
    from utils.engine import compute_segment as _compute_seg
    _seg_result  = _compute_seg(seg_income, seg_score, seg_balance, seg_tenure, seg_products)
    _seg_color   = SEG_C.get(_seg_result, ABG_MUTED)

    seg_icons = {'Standard':'👤','Silver':'🥈','Gold':'🥇','Platinum':'💎','VIP':'👑'}
    seg_descs = {
        'Standard': 'Entry-level customers — SMS & branch-based outreach',
        'Silver':   'Growing customers — WhatsApp & push engagement',
        'Gold':     'Core valuable segment — digital-first approach',
        'Platinum': 'High-value customers — dedicated RM assignment',
        'VIP':      'Top-tier customers — priority concierge service',
    }
    # Score breakdown
    _score_income   = round(min(40, seg_income / 1000), 1)
    _score_cs       = round((seg_score - 300) / 600 * 25, 1)
    _score_balance  = round(min(15, seg_balance / 20000), 1)
    _score_tenure   = round(min(10, seg_tenure / 18), 1)
    _score_products = round(len(seg_products) * 2, 1)
    _total_score    = round(_score_income + _score_cs + _score_balance + _score_tenure + _score_products, 1)

    rs1, rs2, rs3 = st.columns([1.2, 1, 1.6])
    with rs1:
        st.markdown(f"""
        <div style="background:{ABG_WHITE};border:2px solid {_seg_color};
        border-radius:14px;padding:22px;text-align:center;
        box-shadow:0 4px 16px rgba({','.join(str(int(h,16)) for h in [_seg_color[1:3],_seg_color[3:5],_seg_color[5:7]])},0.15);">
          <div style="font-size:36px;margin-bottom:8px;">{seg_icons.get(_seg_result,'👤')}</div>
          <div style="font-size:11px;font-weight:700;color:{ABG_MUTED};
          text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Predicted Segment</div>
          <div style="font-size:28px;font-weight:800;color:{_seg_color};">{_seg_result}</div>
          <div style="font-size:11px;color:{ABG_MUTED};margin-top:6px;line-height:1.4;">
            {seg_descs.get(_seg_result,'')}</div>
          <div style="margin-top:12px;background:{_seg_color}15;border-radius:8px;padding:8px;">
            <div style="font-size:10px;color:{ABG_MUTED};">Composite Score</div>
            <div style="font-size:20px;font-weight:800;color:{_seg_color};">{_total_score:.1f}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    with rs2:
        thresholds = [
            ("VIP",      "≥ 80", ABG_RED    if _seg_result == 'VIP'      else ABG_MUTED),
            ("Platinum", "≥ 60", ABG_PURPLE if _seg_result == 'Platinum' else ABG_MUTED),
            ("Gold",     "≥ 40", ABG_GOLD   if _seg_result == 'Gold'     else ABG_MUTED),
            ("Silver",   "≥ 22", '#95A5A6'  if _seg_result == 'Silver'   else ABG_MUTED),
            ("Standard", "< 22", '#7F8C8D'  if _seg_result == 'Standard' else ABG_MUTED),
        ]
        st.markdown(
            f'<div style="font-size:12px;font-weight:700;color:{ABG_DARK};margin-bottom:10px;">'
            f'Score Thresholds</div>', unsafe_allow_html=True)
        for seg_name, thresh, color in thresholds:
            active = seg_name == _seg_result
            st.markdown(f"""
            <div style="background:{''+color+'12' if active else 'transparent'};
            border:1.5px solid {''+color+'44' if active else ABG_BORDER};
            border-radius:8px;padding:7px 12px;margin-bottom:4px;
            display:flex;align-items:center;justify-content:space-between;">
              <span style="font-size:12px;font-weight:{'700' if active else '500'};
              color:{''+color if active else '#888'};">{seg_icons.get(seg_name,'')} {seg_name}</span>
              <span style="font-size:11px;font-weight:700;color:{color};">{thresh}</span>
              {'<span style="font-size:10px;color:'+color+';font-weight:700;">← YOU</span>' if active else ''}
            </div>""", unsafe_allow_html=True)

    with rs3:
        st.markdown(
            f'<div style="font-size:12px;font-weight:700;color:{ABG_DARK};margin-bottom:10px;">'
            f'Score Breakdown</div>', unsafe_allow_html=True)
        breakdown = [
            ("Monthly Income",  _score_income,   40,  ABG_BLUE),
            ("Credit Score",    _score_cs,        25,  ABG_GOLD),
            ("Account Balance", _score_balance,   15,  ABG_GREEN),
            ("Tenure",          _score_tenure,    10,  ABG_PURPLE),
            ("Products Owned",  _score_products,  10,  ABG_ORANGE),
        ]
        for label, val, max_val, color in breakdown:
            pct = (val / max_val * 100) if max_val > 0 else 0
            st.markdown(f"""
            <div style="margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:11px;color:{ABG_MUTED};">{label}</span>
                <span style="font-size:11px;font-weight:700;color:{color};">
                  {val:.1f} / {max_val}</span>
              </div>
              <div style="background:#F0F0F6;border-radius:4px;height:6px;overflow:hidden;">
                <div style="height:100%;width:{pct:.0f}%;background:{color};
                border-radius:4px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{_seg_color}12;border:1.5px solid {_seg_color}33;
        border-radius:8px;padding:10px 12px;margin-top:6px;">
          <div style="display:flex;justify-content:space-between;">
            <span style="font-size:12px;font-weight:700;color:{ABG_DARK};">Total Score</span>
            <span style="font-size:14px;font-weight:800;color:{_seg_color};">
              {_total_score:.1f} pts</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Live NBA Demo ─────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;'
                f'margin:0 0 14px;">🎯 Live NBA — AI Recommendation</h3>',
                unsafe_allow_html=True)

    with st.expander("⚙️ Configure customer profile", expanded=True):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            demo_income  = st.slider("Monthly Income (SAR)", 3000, 80000, 15000, 1000)
            demo_score   = st.slider("Credit Score", 500, 900, 720, 10)
        with r1c2:
            demo_balance = st.slider("Account Balance (SAR)", 5000, 500000, 60000, 5000)
            demo_tenure  = st.slider("Tenure (months)", 1, 180, 36, 1)
        with r1c3:
            demo_segment  = st.selectbox("Segment",
                ['Standard','Silver','Gold','Platinum','VIP'], index=2)
            demo_owned    = st.multiselect("Products Owned",
                ['CC_TRAVEL','PL_PERSONAL','CC_CASHBACK','PL_HOME','SA_SAVING'],
                default=['SA_SAVING'])

    demo_customer = {
        'monthly_income':    demo_income,
        'credit_score':      demo_score,
        'balance':           demo_balance,
        'tenure_months':     demo_tenure,
        'segment':           demo_segment,
        'products_owned':    '|'.join(demo_owned) if demo_owned else '',
        'num_products_owned': len(demo_owned),
        'engagement_score':  20.0,
        'behavior_score':    50.0,
        'app_logins':        30,
        'total_spend_12m':   demo_income * 10,
        'has_credit_card':   int(any('CC_' in p for p in demo_owned)),
        'has_loan':          int(any('PL_' in p for p in demo_owned)),
        'late_payment_count': 0,
        'preferred_channel': 'Mobile App',
    }

    recs = nba_engine_ml(demo_customer, top_n=3)

    if not recs:
        st.info("Customer already owns all available products.")
    else:
        rec_cols = st.columns(len(recs))
        for col, rec in zip(rec_cols, recs):
            prob   = rec['ml_probability']
            score  = rec['confidence_pct']
            color  = CAT_COLORS.get(rec['category'], ABG_BLUE)
            reasons_html = "".join([
                f'<div style="font-size:10px;color:{ABG_MUTED};padding:3px 0;'
                f'border-bottom:1px solid {ABG_BORDER};">· {r}</div>'
                for r in rec['reason_codes'][:3]
            ])
            col.markdown(f"""
            <div style="background:{ABG_WHITE};border:1.5px solid {color}44;
            border-radius:14px;padding:20px;border-top:4px solid {color};">
              <div style="font-size:10px;font-weight:700;color:{color};
              text-transform:uppercase;letter-spacing:0.08em;
              margin-bottom:6px;">{rec['category']}</div>
              <div style="font-size:14px;font-weight:800;color:{ABG_DARK};
              margin-bottom:14px;line-height:1.3;">{rec['product_name']}</div>
              <div style="font-size:10px;color:{ABG_MUTED};margin-bottom:2px;">
                AI Buy Probability</div>
              <div style="font-size:28px;font-weight:800;color:{color};">{score:.1f}%</div>
              <div style="background:#F0F0F6;border-radius:4px;height:7px;
              margin:6px 0 14px;overflow:hidden;">
                <div style="height:100%;width:{int(prob*100)}%;
                background:{color};border-radius:4px;"></div>
              </div>
              <div style="font-size:10px;font-weight:700;color:{ABG_DARK};margin-bottom:5px;">
                Why this product:</div>
              {reasons_html}
              <div style="margin-top:12px;font-size:10px;color:{ABG_MUTED};">Limit / Rate</div>
              <div style="font-size:12px;font-weight:700;color:{ABG_DARK};">
                {rec['recommended_limit']}</div>
              <div style="margin-top:10px;background:{color}12;border-radius:4px;
              padding:3px 8px;display:inline-block;">
                <span style="font-size:10px;font-weight:600;color:{color};">
                🤖 Scored by LightGBM</span>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Segment Profiles (original) ───────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;'
                f'margin:0 0 16px;">Segment Profiles & Channel Strategy</h3>',
                unsafe_allow_html=True)
    profiles = [
        ("Standard","~30%","SAR 4,800","SAR 12,000","SMS","#7F8C8D"),
        ("Silver","~25%","SAR 8,500","SAR 28,000","WhatsApp","#95A5A6"),
        ("Gold","~22%","SAR 14,000","SAR 65,000","Push Notification","#F39C12"),
        ("Platinum","~15%","SAR 24,000","SAR 140,000","RM","#8E44AD"),
        ("VIP","~8%","SAR 48,000","SAR 380,000","RM / Branch","#E63946"),
    ]
    pcols = st.columns(5)
    for col,(sn,pct,inc,bal,ch,color) in zip(pcols,profiles):
        col.markdown(f"""
        <div style="background:{ABG_WHITE};border:1.5px solid {color}33;
        border-radius:10px;padding:16px;border-top:3px solid {color};">
          <div style="font-size:14px;font-weight:800;color:{color};">{sn}</div>
          <div style="font-size:11px;color:{ABG_MUTED};margin-bottom:10px;">{pct} of base</div>
          <div style="font-size:10px;color:{ABG_MUTED};">Avg Income</div>
          <div style="font-size:13px;font-weight:700;color:{ABG_DARK};">{inc}</div>
          <div style="font-size:10px;color:{ABG_MUTED};margin-top:6px;">Avg Balance</div>
          <div style="font-size:13px;font-weight:700;color:{ABG_DARK};">{bal}</div>
          <div style="font-size:10px;color:{ABG_MUTED};margin-top:6px;">Top Channel</div>
          <div style="font-size:12px;font-weight:600;color:{color};">{ch}</div>
        </div>""", unsafe_allow_html=True)
