import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from utils.styling import *
from utils.engine import generate_dashboard_stats, SEGMENT_COLORS as SEG_C, RISK_COLORS as RISK_C

def render():
    st.markdown(page_header(
        "BSF AI Banking Intelligence System",
        "End-to-end AI pipeline: Segmentation → Cross-sell → Best Channel → Credit Decision · Developed by ABG",
        "🏦"
    ), unsafe_allow_html=True)

    stats = generate_dashboard_stats()

    # ── KPI Row ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(metric_card("Total Customers", "5,000", "Synthetic BSF profiles", ABG_BLUE, "Module 1", ABG_BLUE), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Sessions Analyzed", "~72K", "Clickstream sessions", ABG_GOLD, "Module 2", ABG_GOLD), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Credit Decisions", "5,000", "With Human-in-Loop", ABG_GREEN, "Module 3", ABG_GREEN), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("Model Accuracy", "84.1%", "Ensemble CV 5-fold", ABG_PURPLE, "Ensemble", ABG_PURPLE), unsafe_allow_html=True)
    with c5: st.markdown(metric_card("Anomalies Flagged", f"{stats['anomaly_count']}", "IsoForest + Z-Score", ABG_RED, "Alert", ABG_RED), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row ────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        seg = stats['segment_dist']
        fig = go.Figure(go.Bar(
            x=list(seg.keys()), y=list(seg.values()),
            marker_color=[SEG_C[s] for s in seg.keys()],
            marker_line_width=0,
        ))
        fig.update_layout(
            title=dict(text="Customer Segments", font=dict(size=14, color=ABG_DARK, family="Plus Jakarta Sans")),
            paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
            font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
            margin=dict(t=40, b=20, l=10, r=10), height=260,
            showlegend=False,
            xaxis=dict(tickfont=dict(size=11), gridcolor=ABG_BORDER),
            yaxis=dict(tickfont=dict(size=11), gridcolor=ABG_BORDER),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        beh = stats['behavior_dist']
        beh_colors = [ABG_GREEN, ABG_BLUE, ABG_GOLD, ABG_MUTED]
        fig2 = go.Figure(go.Pie(
            labels=list(beh.keys()), values=list(beh.values()),
            marker=dict(colors=beh_colors, line=dict(color='white', width=2)),
            hole=0.55,
            textfont=dict(size=11, family="Plus Jakarta Sans"),
        ))
        fig2.update_layout(
            title=dict(text="Behavior Categories", font=dict(size=14, color=ABG_DARK, family="Plus Jakarta Sans")),
            paper_bgcolor=ABG_WHITE, margin=dict(t=40, b=10, l=10, r=10), height=260,
            legend=dict(font=dict(size=11, family="Plus Jakarta Sans")),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        risk = stats['risk_dist']
        fig3 = go.Figure(go.Pie(
            labels=list(risk.keys()), values=list(risk.values()),
            marker=dict(colors=[RISK_C[r] for r in risk.keys()], line=dict(color='white', width=2)),
            hole=0.55,
            textfont=dict(size=11, family="Plus Jakarta Sans"),
        ))
        fig3.update_layout(
            title=dict(text="Risk Categories", font=dict(size=14, color=ABG_DARK, family="Plus Jakarta Sans")),
            paper_bgcolor=ABG_WHITE, margin=dict(t=40, b=10, l=10, r=10), height=260,
            legend=dict(font=dict(size=11, family="Plus Jakarta Sans")),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Module Cards ──────────────────────────────────────────
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:8px 0 16px;">System Modules</h3>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    modules = [
        ("👥", "Module 1 — Segmentation & NBA", ABG_BLUE,
         "K-Means clustering segments 5,000 customers into 5 tiers. NBA engine recommends top-2 BSF products per customer with match scores, affinity boost & product-specific reason codes.",
         ["K-Means (K=5)", "PCA 2D", "Weighted Scoring", "Affinity Boost", "Channel Selector"]),
        ("📊", "Module 2 — Behavior Analysis", ABG_GOLD,
         "Analyzes clickstream data (clicks, page visits, scrolls) to compute behavior scores, detect multivariate anomalies, and identify segment drift candidates.",
         ["Isolation Forest", "Z-Score", "Behavior Scoring", "Drift Detection", "Analysis Agent"]),
        ("💳", "Module 3 — Credit Decision", ABG_GREEN,
         "Ensemble model (XGBoost + Random Forest + Gradient Boosting) classifies risk and recommends CC limits, loan amounts, and SAIBOR-linked interest rates with human approval.",
         ["XGBoost", "Random Forest", "Gradient Boosting", "Ensemble Voting", "Human-in-Loop"]),
        ("🎯", "Live Customer Demo", ABG_PURPLE,
         "Enter any customer profile and run the full AI pipeline in real-time. See segmentation, NBA recommendations, behavior score, credit decision, and chatbot welcome message instantly.",
         ["Interactive", "Full Pipeline", "Real-time", "Ahmed Al-Harbi", "Sara · Khalid"]),
    ]
    for i, (icon, title, color, desc, tags) in enumerate(modules):
        col = c1 if i % 2 == 0 else c2
        tags_html = ''.join([f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;background:{color}15;color:{color};border:1px solid {color}30;margin:2px;">{t}</span>' for t in tags])
        with col:
            st.markdown(f"""
            <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};border-radius:12px;padding:22px;
            border-top:3px solid {color};margin-bottom:16px;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
              <div style="font-size:28px;margin-bottom:10px;">{icon}</div>
              <div style="font-size:14px;font-weight:800;color:{ABG_DARK};margin-bottom:8px;">{title}</div>
              <div style="font-size:12px;color:{ABG_MUTED};line-height:1.6;margin-bottom:14px;">{desc}</div>
              <div>{tags_html}</div>
            </div>""", unsafe_allow_html=True)

    # ── Top Products ──────────────────────────────────────────
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:8px 0 16px;">Top Recommended Products (NBA Engine)</h3>', unsafe_allow_html=True)
    prods = stats['top_products']
    prod_names = [p['name'].replace('BSF ', '') for p in prods]
    prod_counts = [p['count'] for p in prods]
    prod_scores = [p['avg_score'] for p in prods]
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name='Times Recommended', x=prod_names, y=prod_counts,
        marker_color=ABG_BLUE, marker_line_width=0, yaxis='y'))
    fig4.add_trace(go.Scatter(name='Avg Match Score %', x=prod_names, y=prod_scores,
        mode='lines+markers', line=dict(color=ABG_GOLD, width=2),
        marker=dict(size=8, color=ABG_GOLD), yaxis='y2'))
    fig4.update_layout(
        paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
        font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
        margin=dict(t=20, b=20, l=10, r=60), height=280,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=11)),
        yaxis=dict(title='Count', gridcolor=ABG_BORDER),
        yaxis2=dict(title='Match Score %', overlaying='y', side='right', range=[60, 100]),
        xaxis=dict(tickfont=dict(size=11), gridcolor=ABG_BORDER),
    )
    st.plotly_chart(fig4, use_container_width=True)
