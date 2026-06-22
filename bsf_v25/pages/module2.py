import streamlit as st
import plotly.graph_objects as go
from utils.styling import *
from utils.engine import generate_dashboard_stats

def render():
    st.markdown(page_header("Module 2 — Customer Behavior Analysis",
        "Clickstream analysis · Anomaly detection · Segment drift · Analysis agent", "📊"), unsafe_allow_html=True)

    stats = generate_dashboard_stats()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(metric_card("Highly Active", "~18%", "Score ≥ 75/100", ABG_GREEN, "Top Tier", ABG_GREEN), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Active", "~32%", "Score 50–74", ABG_BLUE, "Normal", ABG_BLUE), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Anomalies", "250", "IsoForest + Z-Score", ABG_RED, "5.0%", ABG_RED), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("Upgrade Candidates", "~600", "Above segment norm +1.5σ", ABG_PURPLE, "12%", ABG_PURPLE), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        beh_colors = [ABG_GREEN, ABG_BLUE, ABG_GOLD, ABG_MUTED]
        labels = ['Highly Active', 'Active', 'Moderate', 'Dormant']
        values = [900, 1600, 1750, 750]
        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=beh_colors, line=dict(color='white', width=2)),
            hole=0.55,
        ))
        fig.update_layout(title="Behavior Category Distribution", paper_bgcolor=ABG_WHITE,
            font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
            margin=dict(t=40,b=10,l=10,r=10), height=280,
            legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        anom = stats['anomaly_types']
        fig2 = go.Figure(go.Bar(
            x=list(anom.values()), y=list(anom.keys()), orientation='h',
            marker_color=[ABG_RED, ABG_ORANGE, ABG_GOLD, '#C0392B', ABG_PURPLE],
            marker_line_width=0,
        ))
        fig2.update_layout(title="Anomaly Types Detected", paper_bgcolor=ABG_WHITE,
            plot_bgcolor='#FAFAFA', font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
            margin=dict(t=40,b=10,l=10,r=10), height=280,
            xaxis=dict(gridcolor=ABG_BORDER), yaxis=dict(gridcolor=ABG_BORDER))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Detection Methods ─────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 16px;">Detection Methods</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};border-radius:12px;padding:22px;">
          <div style="font-size:14px;font-weight:800;color:{ABG_BLUE};margin-bottom:12px;">🌲 Isolation Forest (Multivariate)</div>
          <p style="font-size:12px;color:{ABG_MUTED};line-height:1.7;">
          Detects outliers by isolating anomalies in 8-dimensional behavioral space.
          Anomalies are isolated faster — shorter paths in random trees.<br><br>
          <strong style="color:{ABG_DARK};">Features:</strong> sessions, duration, clicks, scrolls,
          transaction rate, drop-off rate, MoM session change, MoM click change
          </p>
          <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;">
            <span style="background:{ABG_BLUE}15;color:{ABG_BLUE};padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;">n_estimators=200</span>
            <span style="background:{ABG_BLUE}15;color:{ABG_BLUE};padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;">contamination=5%</span>
            <span style="background:{ABG_BLUE}15;color:{ABG_BLUE};padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;">max_samples=auto</span>
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};border-radius:12px;padding:22px;">
          <div style="font-size:14px;font-weight:800;color:{ABG_GOLD};margin-bottom:12px;">📊 Z-Score (Univariate)</div>
          <p style="font-size:12px;color:{ABG_MUTED};line-height:1.7;">
          Flags customers whose individual feature values deviate more than 3 standard
          deviations from the population mean. Complements Isolation Forest.<br><br>
          <strong style="color:{ABG_DARK};">Formula:</strong> z = (x − μ) / σ · Flag if |z| > 3.0<br>
          <strong style="color:{ABG_DARK};">Features:</strong> clicks, session duration, drop-off rate, MoM session change
          </p>
          <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;">
            <span style="background:{ABG_GOLD}15;color:{ABG_GOLD};padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;">Threshold |z| > 3.0</span>
            <span style="background:{ABG_GOLD}15;color:{ABG_GOLD};padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;">4 key features</span>
            <span style="background:{ABG_GOLD}15;color:{ABG_GOLD};padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;">scipy.stats</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Drift Detection ───────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 16px;">Segment Drift Detection</h3>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    drifts = [
        ("⬆️", "Upgrade Candidate", ABG_GREEN, "~12%",
         "Behavior score > +1.5σ above segment mean", "Review for segment upgrade"),
        ("➡️", "Stable", ABG_BLUE, "~76%",
         "Within ±1.5σ of segment mean", "Continue monitoring"),
        ("⬇️", "Downgrade Risk", ABG_RED, "~12%",
         "Behavior score < −1.5σ below segment mean", "Apply retention strategy"),
    ]
    for col, (icon, title, color, pct, desc, action) in zip([c1,c2,c3], drifts):
        with col:
            col.markdown(f"""
            <div style="background:{color}08;border:1.5px solid {color}33;border-radius:12px;padding:20px;text-align:center;">
              <div style="font-size:32px;">{icon}</div>
              <div style="font-size:14px;font-weight:800;color:{color};margin:8px 0 4px;">{title}</div>
              <div style="font-size:22px;font-weight:800;color:{ABG_DARK};">{pct}</div>
              <div style="font-size:11px;color:{ABG_MUTED};margin:6px 0 10px;">{desc}</div>
              <div style="font-size:11px;font-weight:600;color:{color};">→ {action}</div>
            </div>""", unsafe_allow_html=True)

    # ── Behavior Score Weights ─────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 16px;">Behavior Score Composition (0–100)</h3>', unsafe_allow_html=True)

    weights = [
        ("Total Sessions", "20%", ABG_BLUE),
        ("Avg Session Duration", "15%", ABG_BLUE),
        ("Avg Clicks/Session", "15%", ABG_GOLD),
        ("Avg Pages/Session", "15%", ABG_GOLD),
        ("Transaction Rate", "20%", ABG_GREEN),
        ("Engagement Depth", "10%", ABG_PURPLE),
        ("Mobile Ratio", "5%", ABG_MUTED),
    ]
    st.markdown(f"""
    <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};border-radius:12px;padding:20px;">
    {"".join([f'''
    <div style="margin-bottom:12px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span style="font-size:12px;font-weight:600;color:{ABG_DARK};">{feat}</span>
        <span style="font-size:12px;font-weight:700;color:{color};">{w}</span>
      </div>
      <div style="background:#F0F0F6;border-radius:4px;height:6px;overflow:hidden;">
        <div style="height:100%;width:{int(w.rstrip("%"))*4}%;background:{color};border-radius:4px;"></div>
      </div>
    </div>''' for feat, w, color in weights])}
    </div>""", unsafe_allow_html=True)
