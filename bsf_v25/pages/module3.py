import streamlit as st
import plotly.graph_objects as go
from utils.styling import *
from utils.engine import generate_dashboard_stats, RISK_COLORS as RC, SAIBOR_3M

def render():
    st.markdown(page_header("Module 3 — Dynamic Credit Decision Engine",
        "Ensemble ML · SAIBOR-linked rates · Human-in-the-Loop approval", "💳"), unsafe_allow_html=True)

    stats = generate_dashboard_stats()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(metric_card("Ensemble Accuracy", "84.1%", "5-fold CV · Dynamic weights", ABG_GREEN, "Best Model", ABG_GREEN), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("CC Eligible", "~78%", "Score ≥ 580 · Income ≥ 3K", ABG_BLUE, "Eligible", ABG_BLUE), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Loan Eligible", "~65%", "DBR ≤ 33% · Score ≥ 620", ABG_GOLD, "Eligible", ABG_GOLD), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("Auto-Approve", "~45%", "Very Low / Low risk only", ABG_PURPLE, "No Review", ABG_PURPLE), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        acc = stats['model_accuracy']
        colors = [ABG_BLUE, ABG_GOLD, ABG_GREEN, ABG_PURPLE]
        is_best = ['Ensemble ✅' if k == 'Ensemble' else k for k in acc.keys()]
        fig = go.Figure(go.Bar(
            x=is_best, y=list(acc.values()),
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.1f}%" for v in acc.values()], textposition='outside',
        ))
        fig.update_layout(title="Model Performance Comparison", paper_bgcolor=ABG_WHITE,
            plot_bgcolor='#FAFAFA', font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
            margin=dict(t=40,b=20,l=10,r=10), height=280, showlegend=False,
            yaxis=dict(range=[75, 88], gridcolor=ABG_BORDER),
            xaxis=dict(gridcolor=ABG_BORDER))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        risk = stats['risk_dist']
        fig2 = go.Figure(go.Pie(
            labels=list(risk.keys()), values=list(risk.values()),
            marker=dict(colors=[RC[r] for r in risk.keys()], line=dict(color='white', width=2)),
            hole=0.55,
        ))
        fig2.update_layout(title="Risk Category Distribution", paper_bgcolor=ABG_WHITE,
            font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
            margin=dict(t=40,b=10,l=10,r=10), height=280,
            legend=dict(font=dict(size=11)))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Credit Rules Table ────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 16px;">Credit Decision Rules by Risk Category</h3>', unsafe_allow_html=True)

    rules = [
        ("Very Low",  ABG_GREEN,  "4.0×", "Elite/Platinum", "24×", "60m", f"SAIBOR + 1.5%", "Auto-Approve",     ABG_GREEN),
        ("Low",       "#2ECC71",  "3.0×", "Gold/Privilege", "18×", "48m", f"SAIBOR + 2.0%", "Auto-Approve",     ABG_GREEN),
        ("Medium",    ABG_GOLD,   "2.0×", "Advance",        "12×", "36m", f"SAIBOR + 3.0%", "Manual Review",    ABG_ORANGE),
        ("High",      ABG_ORANGE, "1.0×", "Classic",        "6×",  "24m", f"SAIBOR + 4.5%", "Senior Review",    ABG_RED),
        ("Very High", ABG_RED,    "0.5×", "Entry-Level",    "3×",  "12m", f"SAIBOR + 6.0%", "Decline",          '#6B6B8A'),
    ]

    rows = ''.join([f"""
    <tr style="border-bottom:1px solid {ABG_BORDER};">
      <td style="padding:12px 14px;"><span style="display:inline-block;padding:3px 12px;border-radius:5px;font-size:12px;font-weight:700;background:{rc}22;color:{rc};">{risk}</span></td>
      <td style="padding:12px 14px;font-size:13px;font-weight:600;color:{ABG_DARK};">{cm}</td>
      <td style="padding:12px 14px;font-size:12px;color:{ABG_MUTED};">{tier}</td>
      <td style="padding:12px 14px;font-size:13px;font-weight:600;color:{ABG_DARK};">{lm}</td>
      <td style="padding:12px 14px;font-size:12px;color:{ABG_MUTED};">{ten}</td>
      <td style="padding:12px 14px;font-size:12px;color:{ABG_MUTED};">{rate}</td>
      <td style="padding:12px 14px;"><span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;background:{ac}22;color:{ac};">{appr}</span></td>
    </tr>""" for risk, rc, cm, tier, lm, ten, rate, appr, ac in rules])

    st.markdown(f"""
    <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};border-radius:12px;overflow:hidden;">
      <table style="width:100%;border-collapse:collapse;">
        <thead><tr style="background:#FAFAFA;border-bottom:2px solid {ABG_BORDER};">
          <th style="padding:12px 14px;text-align:left;font-size:11px;color:{ABG_MUTED};text-transform:uppercase;letter-spacing:0.06em;">Risk</th>
          <th style="padding:12px 14px;font-size:11px;color:{ABG_MUTED};text-transform:uppercase;">CC Multiplier</th>
          <th style="padding:12px 14px;font-size:11px;color:{ABG_MUTED};text-transform:uppercase;">Card Tier</th>
          <th style="padding:12px 14px;font-size:11px;color:{ABG_MUTED};text-transform:uppercase;">Loan Multiple</th>
          <th style="padding:12px 14px;font-size:11px;color:{ABG_MUTED};text-transform:uppercase;">Max Tenure</th>
          <th style="padding:12px 14px;font-size:11px;color:{ABG_MUTED};text-transform:uppercase;">Interest Rate</th>
          <th style="padding:12px 14px;font-size:11px;color:{ABG_MUTED};text-transform:uppercase;">Approval</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div style="margin-top:10px;padding:10px 14px;background:{ABG_GOLD}0d;border-radius:6px;border-left:3px solid {ABG_GOLD};font-size:12px;color:{ABG_DARK};">
      <strong>SAIBOR 3M Base Rate: {SAIBOR_3M}%</strong> (June 2025) · All rates = SAIBOR + Risk Spread + Credit Score Fine-tune
    </div>""", unsafe_allow_html=True)

    # ── Ensemble Architecture ─────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 16px;">Ensemble Architecture</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    models = [
        ("XGBoost", "82.2%", "Weight: 3", ABG_BLUE,   "n=300 · depth=6 · lr=0.05"),
        ("Random Forest", "79.0%", "Weight: 2", ABG_GOLD, "n=300 · depth=10 · balanced"),
        ("Gradient Boosting", "81.5%", "Weight: 2", ABG_GREEN, "n=300 · depth=6 · lr=0.05"),
    ]
    for col, (name, acc_v, weight, color, params) in zip([c1,c2,c3], models):
        with col:
            col.markdown(f"""
            <div style="background:{ABG_WHITE};border:1.5px solid {color}33;border-radius:10px;
            padding:16px;border-top:3px solid {color};text-align:center;">
              <div style="font-size:13px;font-weight:800;color:{color};">{name}</div>
              <div style="font-size:24px;font-weight:800;color:{ABG_DARK};margin:6px 0;">{acc_v}</div>
              <div style="font-size:11px;font-weight:600;color:{color};margin-bottom:8px;">{weight}</div>
              <div style="font-size:10px;color:{ABG_MUTED};">{params}</div>
            </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div style="background:{ABG_PURPLE}0a;border:2px solid {ABG_PURPLE};border-radius:10px;
        padding:16px;text-align:center;">
          <div style="font-size:13px;font-weight:800;color:{ABG_PURPLE};">✅ Ensemble</div>
          <div style="font-size:24px;font-weight:800;color:{ABG_DARK};margin:6px 0;">84.1%</div>
          <div style="font-size:11px;font-weight:600;color:{ABG_PURPLE};margin-bottom:8px;">Soft Voting · Final</div>
          <div style="font-size:10px;color:{ABG_MUTED};">Dynamic weights from CV accuracy</div>
        </div>""", unsafe_allow_html=True)

    # ── Human in the Loop ─────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 16px;">👤 Human-in-the-Loop Workflow</h3>', unsafe_allow_html=True)
    steps = [
        ("🤖", "AI Risk Score", "Ensemble model\n0–100 scale", ABG_BLUE, "Automated"),
        ("📋", "Decision Package", "CC Limit + Loan\n+ Rate + Tenure", ABG_BLUE, "Generated"),
        ("👁️", "Internal Review", "Credit Officer\nDocument check", ABG_ORANGE, "Human Review"),
        ("✅", "Approve / Reject", "Final decision\nby team", ABG_GREEN, "Awaiting"),
        ("📱", "Customer Contact", "Via best channel\nPersonalized msg", ABG_MUTED, "Pending"),
    ]
    cols = st.columns(5)
    for col, (icon, title, desc, color, status) in zip(cols, steps):
        with col:
            col.markdown(f"""
            <div style="background:{ABG_WHITE};border:1.5px solid {color}33;border-radius:10px;
            padding:14px;text-align:center;border-top:3px solid {color};">
              <div style="font-size:24px;">{icon}</div>
              <div style="font-size:12px;font-weight:700;color:{ABG_DARK};margin:6px 0 4px;">{title}</div>
              <div style="font-size:10px;color:{ABG_MUTED};white-space:pre-line;">{desc}</div>
              <div style="margin-top:8px;font-size:10px;font-weight:600;color:{color};">{status}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(info_box(
        "⚠️ <strong>Important:</strong> No credit decision is applied until reviewed and approved by the internal BSF credit team. "
        "Auto-Approve is only available for Very Low and Low risk customers with no anomaly flags.",
        ABG_ORANGE
    ), unsafe_allow_html=True)
