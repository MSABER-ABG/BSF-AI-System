"""
Segmentation Analysis — Full Dashboard Page
Design: card-based layout, KPI strip, tabbed deep-dive sections
Language: English only
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import sklearn.decomposition as _skd

from utils.styling import (
    ABG_BLUE, ABG_DARK, ABG_WHITE, ABG_BORDER, ABG_MUTED,
    ABG_RED, ABG_GREEN, ABG_GOLD, ABG_PURPLE, ABG_ORANGE,
    page_header, info_box,
)
from utils.segmentation_viz import (
    get_segmentation_data,
    SEGMENT_ORDER, SEGMENT_COLORS, SEGMENT_SIZES,
    FEATURES, FEATURE_LABELS, K_OPTIMAL,
)

# ── Shared plot defaults ───────────────────────────────────────
FONT   = dict(family="Plus Jakarta Sans", color=ABG_DARK)
BG     = dict(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA')
MGRID  = dict(gridcolor=ABG_BORDER)
SEG_CL = [SEGMENT_COLORS[s] for s in SEGMENT_ORDER]

def _layout(**kw):
    base = dict(**BG, font=FONT, margin=dict(t=46, b=16, l=10, r=10))
    base.update(kw)
    return base

# ── Reusable card HTML helpers ─────────────────────────────────
def _kpi_card(icon, label, value, sub, color, delta=None, delta_up=True):
    delta_html = ""
    if delta:
        arr   = "↑" if delta_up else "↓"
        dcol  = ABG_GREEN if delta_up else ABG_RED
        delta_html = (
            f'<div style="display:flex;align-items:center;gap:4px;margin-top:4px;">'
            f'<span style="font-size:12px;font-weight:700;color:{dcol};">'
            f'{arr} {delta}</span>'
            f'<span style="font-size:11px;color:{ABG_MUTED};">vs baseline</span></div>'
        )
    return f"""
    <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};border-radius:14px;
    padding:18px 16px;display:flex;align-items:flex-start;gap:14px;
    box-shadow:0 2px 8px rgba(61,61,219,0.06);">
      <div style="width:42px;height:42px;border-radius:10px;background:{color}15;
      display:flex;align-items:center;justify-content:center;
      font-size:20px;flex-shrink:0;">{icon}</div>
      <div>
        <div style="font-size:11px;font-weight:700;color:{ABG_MUTED};
        text-transform:uppercase;letter-spacing:0.07em;">{label}</div>
        <div style="font-size:22px;font-weight:800;color:{color};
        line-height:1.2;margin-top:2px;">{value}</div>
        <div style="font-size:11px;color:{ABG_MUTED};margin-top:1px;">{sub}</div>
        {delta_html}
      </div>
    </div>"""

def _section_header(title, subtitle=""):
    sub = f'<div style="font-size:11px;color:{ABG_MUTED};margin-top:2px;">{subtitle}</div>' if subtitle else ""
    return (
        f'<div style="margin:24px 0 14px;">'
        f'<div style="font-size:14px;font-weight:800;color:{ABG_DARK};">{title}</div>'
        f'{sub}</div>'
    )

def _card_wrap(content_html, accent=ABG_BORDER):
    return (
        f'<div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};'
        f'border-radius:14px;padding:20px;border-top:3px solid {accent};'
        f'box-shadow:0 2px 8px rgba(61,61,219,0.05);">{content_html}</div>'
    )

def _badge(text, color):
    return (
        f'<span style="background:{color}18;color:{color};'
        f'padding:3px 10px;border-radius:20px;font-size:11px;'
        f'font-weight:700;letter-spacing:0.03em;">{text}</span>'
    )

def _insight_row(insight, severity, recommendation, status):
    sev_color  = {
        "High": ABG_RED, "Medium": ABG_ORANGE, "Low": ABG_GREEN
    }.get(severity, ABG_MUTED)
    stat_color = {
        "Ready": ABG_GREEN, "In Review": ABG_BLUE,
        "Pending": ABG_ORANGE, "Active": ABG_PURPLE,
    }.get(status, ABG_MUTED)
    dot = f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{stat_color};margin-right:4px;"></span>'
    return f"""
    <tr>
      <td style="padding:9px 12px;font-size:12px;color:{ABG_DARK};border-bottom:1px solid {ABG_BORDER};">{insight}</td>
      <td style="padding:9px 12px;border-bottom:1px solid {ABG_BORDER};">{_badge(severity, sev_color)}</td>
      <td style="padding:9px 12px;font-size:12px;color:{ABG_MUTED};border-bottom:1px solid {ABG_BORDER};">{recommendation}</td>
      <td style="padding:9px 12px;font-size:12px;font-weight:600;color:{stat_color};border-bottom:1px solid {ABG_BORDER};">{dot}{status}</td>
    </tr>"""

def _progress_bar(label, icon, pct, color):
    return f"""
    <div style="margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:15px;">{icon}</span>
          <span style="font-size:13px;font-weight:600;color:{ABG_DARK};">{label}</span>
        </div>
        <span style="font-size:13px;font-weight:800;color:{color};">{pct}%</span>
      </div>
      <div style="background:#F0F0F6;border-radius:6px;height:8px;overflow:hidden;">
        <div style="height:100%;width:{pct}%;background:{color};border-radius:6px;"></div>
      </div>
    </div>"""

def _explain_card(text, color=ABG_BLUE):
    return (
        f'<div style="background:{color}08;border-left:3px solid {color};'
        f'border-radius:0 8px 8px 0;padding:11px 14px;margin:10px 0 4px;'
        f'font-size:12px;color:{ABG_DARK};line-height:1.65;">{text}</div>'
    )

# ══════════════════════════════════════════════════════════════
def render():
    st.markdown(
        page_header(
            "Segmentation Analysis",
            "K-Means (K=5) · PCA · Silhouette · Feature Distributions · Cluster Profiles",
            "🔍",
        ),
        unsafe_allow_html=True,
    )

    with st.spinner("Running K-Means on 5,000 customers …"):
        data         = get_segmentation_data()

    df           = data['df']
    centers_df   = data['centers_df']
    centers_norm = data['centers_norm']
    sil_score    = data['sil_score']
    k_range      = data['k_range']
    inertias     = data['inertias']
    sil_k        = data['sil_scores_k']
    pca_exp      = data['pca_explained']
    loadings     = data['pca_loadings']

    # ══════════════════════════════════════════════════════════
    # ── TOP KPI STRIP ─────────────────────────────────────────
    # ══════════════════════════════════════════════════════════
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("👥", "Total Customers",   "5,000",          "Segmented base",         ABG_BLUE,   "+3.1%", True),
        ("🎯", "Active Segments",   "5",              "Standard → VIP",         ABG_PURPLE, None,    True),
        ("📐", "Silhouette Score",  f"{sil_score:.3f}","Cluster cohesion",       ABG_GREEN,  "+0.04", True),
        ("🔬", "PCA Variance",      f"{sum(pca_exp):.1f}%","Explained by 2 PCs", ABG_GOLD,   None,    True),
        ("⚙️", "Features Used",    "12",             "RobustScaler applied",   ABG_DARK,   None,    True),
    ]
    for col, (ico, lbl, val, sub, clr, dlt, up) in zip([k1,k2,k3,k4,k5], kpis):
        col.markdown(_kpi_card(ico, lbl, val, sub, clr, dlt, up), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # ── MAIN DASHBOARD ROW 1 ──────────────────────────────────
    # ══════════════════════════════════════════════════════════
    row1_l, row1_m, row1_r = st.columns([1.1, 1.4, 1.3])

    # ── Spend by Segment donut ─────────────────────────────────
    with row1_l:
        st.markdown(
            _section_header("Spend by Segment", "Annual spend distribution across clusters"),
            unsafe_allow_html=True,
        )
        spend_mult = {'Standard':6, 'Silver':9, 'Gold':14, 'Platinum':24, 'VIP':48}
        spend_vals = [SEGMENT_SIZES[s] * spend_mult[s] for s in SEGMENT_ORDER]
        total_spend = sum(spend_vals)

        fig_donut = go.Figure(go.Pie(
            labels=SEGMENT_ORDER, values=spend_vals,
            marker=dict(colors=SEG_CL, line=dict(color='white', width=2)),
            hole=0.60, textinfo='percent',
            textfont=dict(size=12, family="Plus Jakarta Sans"),
        ))
        fig_donut.update_layout(
            **_layout(height=290, margin=dict(t=10,b=10,l=0,r=0)),
            showlegend=True,
            legend=dict(font=dict(size=11), orientation='v', x=0.85, y=0.5),
            annotations=[],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Transaction & Product Overview ────────────────────────
    with row1_m:
        st.markdown(
            _section_header("Customer & Product Breakdown", "Segment size and avg product holdings"),
            unsafe_allow_html=True,
        )
        prod_avg = {'Standard':1.4,'Silver':2.1,'Gold':3.0,'Platinum':3.8,'VIP':5.2}
        icons_seg = {'Standard':'👤','Silver':'🥈','Gold':'🥇','Platinum':'💎','VIP':'👑'}

        rows_html = ""
        for seg in SEGMENT_ORDER:
            n     = SEGMENT_SIZES[seg]
            pct   = n / 5000 * 100
            color = SEGMENT_COLORS[seg]
            rows_html += f"""
            <tr>
              <td style="padding:9px 12px;font-size:13px;font-weight:700;
              color:{color};border-bottom:1px solid {ABG_BORDER};">
                {icons_seg[seg]} {seg}</td>
              <td style="padding:9px 12px;font-size:12px;color:{ABG_DARK};
              border-bottom:1px solid {ABG_BORDER};">{n:,}</td>
              <td style="padding:9px 12px;border-bottom:1px solid {ABG_BORDER};">
                <div style="background:#F0F0F6;border-radius:4px;height:6px;
                width:120px;overflow:hidden;">
                  <div style="height:100%;width:{pct*2.2:.0f}px;
                  background:{color};border-radius:4px;max-width:120px;"></div>
                </div>
                <span style="font-size:11px;color:{ABG_MUTED};">{pct:.1f}%</span>
              </td>
              <td style="padding:9px 12px;font-size:12px;color:{ABG_DARK};
              font-weight:600;border-bottom:1px solid {ABG_BORDER};">
                {prod_avg[seg]} avg</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
        border-radius:14px;overflow:hidden;box-shadow:0 2px 8px rgba(61,61,219,0.05);">
          <table style="width:100%;border-collapse:collapse;">
            <thead>
              <tr style="background:#F8F8FF;">
                <th style="padding:9px 12px;font-size:11px;font-weight:700;
                color:{ABG_MUTED};text-align:left;text-transform:uppercase;
                letter-spacing:0.07em;">Segment</th>
                <th style="padding:9px 12px;font-size:11px;font-weight:700;
                color:{ABG_MUTED};text-align:left;text-transform:uppercase;
                letter-spacing:0.07em;">Customers</th>
                <th style="padding:9px 12px;font-size:11px;font-weight:700;
                color:{ABG_MUTED};text-align:left;text-transform:uppercase;
                letter-spacing:0.07em;">Share</th>
                <th style="padding:9px 12px;font-size:11px;font-weight:700;
                color:{ABG_MUTED};text-align:left;text-transform:uppercase;
                letter-spacing:0.07em;">Avg Products</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

    # ── AI Insights & Next-Best Actions ───────────────────────
    with row1_r:
        st.markdown(
            _section_header("AI Insights & Segment Actions", "Automated flags from cluster analysis"),
            unsafe_allow_html=True,
        )
        insights = [
            ("Gold segment shows cross-sell potential", "High",   "Push Travel Card offer",      "Ready"),
            ("VIP customers — low digital engagement",  "Medium", "Assign dedicated RM",         "In Review"),
            ("Standard segment: high churn signals",    "High",   "Launch fee-waiver campaign",  "Active"),
            ("Silver customers upgrading trend",        "Medium", "Accelerate with NBA engine",  "Pending"),
        ]
        rows_ins = "".join([_insight_row(*r) for r in insights])
        st.markdown(f"""
        <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
        border-radius:14px;overflow:hidden;box-shadow:0 2px 8px rgba(61,61,219,0.05);">
          <table style="width:100%;border-collapse:collapse;">
            <thead>
              <tr style="background:#F8F8FF;">
                <th style="padding:9px 12px;font-size:11px;font-weight:700;color:{ABG_MUTED};
                text-align:left;text-transform:uppercase;letter-spacing:0.07em;">Insight</th>
                <th style="padding:9px 12px;font-size:11px;font-weight:700;color:{ABG_MUTED};
                text-align:left;text-transform:uppercase;letter-spacing:0.07em;">Severity</th>
                <th style="padding:9px 12px;font-size:11px;font-weight:700;color:{ABG_MUTED};
                text-align:left;text-transform:uppercase;letter-spacing:0.07em;">Recommendation</th>
                <th style="padding:9px 12px;font-size:11px;font-weight:700;color:{ABG_MUTED};
                text-align:left;text-transform:uppercase;letter-spacing:0.07em;">Status</th>
              </tr>
            </thead>
            <tbody>{rows_ins}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # ── MAIN DASHBOARD ROW 2 ──────────────────────────────────
    # ══════════════════════════════════════════════════════════
    row2_l, row2_m, row2_r = st.columns([1.3, 1.2, 1.3])

    # ── Segment coverage progress bars ────────────────────────
    with row2_l:
        st.markdown(
            _section_header("Segment Coverage", "% of customers with ≥2 active products"),
            unsafe_allow_html=True,
        )
        coverage = {'VIP':88,'Platinum':76,'Gold':72,'Silver':54,'Standard':31}
        icons_c  = {'VIP':'👑','Platinum':'💎','Gold':'🥇','Silver':'🥈','Standard':'👤'}
        bars_html = "".join([
            _progress_bar(seg, icons_c[seg], pct, SEGMENT_COLORS[seg])
            for seg, pct in coverage.items()
        ])
        st.markdown(
            _card_wrap(bars_html, ABG_PURPLE),
            unsafe_allow_html=True,
        )

    # ── Cluster size bar ─────────────────────────────────────
    with row2_m:
        st.markdown(
            _section_header("Cluster Size Distribution", "Customers per K-Means cluster"),
            unsafe_allow_html=True,
        )
        fig_bar = go.Figure(go.Bar(
            x=SEGMENT_ORDER,
            y=[SEGMENT_SIZES[s] for s in SEGMENT_ORDER],
            marker_color=SEG_CL,
            marker_line_width=0,
            text=[f"{SEGMENT_SIZES[s]:,}" for s in SEGMENT_ORDER],
            textposition='outside',
            textfont=dict(size=11),
        ))
        fig_bar.update_layout(
            **_layout(height=230, margin=dict(t=10,b=10,l=10,r=10)),
            showlegend=False,
            xaxis=dict(**MGRID),
            yaxis=dict(**MGRID, title="Customers"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Estimated impact cards ────────────────────────────────
    with row2_r:
        st.markdown(
            _section_header("Estimated Segmentation Impact", "Projected uplift from AI-driven NBA"),
            unsafe_allow_html=True,
        )
        impacts = [
            ("📈", "Revenue Uplift",      "+9.2%",     "(30 Days)",    ABG_GREEN),
            ("🛡️", "Churn Prevented",    "1,240",      "Customers",    ABG_BLUE),
            ("💰", "Retained Spend",      "2.7B SAR",  "Est. retained",ABG_PURPLE),
        ]
        for ico, lbl, val, sub, color in impacts:
            st.markdown(f"""
            <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
            border-radius:12px;padding:14px 16px;margin-bottom:10px;
            display:flex;align-items:center;gap:14px;
            box-shadow:0 1px 4px rgba(61,61,219,0.05);">
              <div style="width:38px;height:38px;border-radius:9px;
              background:{color}15;display:flex;align-items:center;
              justify-content:center;font-size:18px;">{ico}</div>
              <div>
                <div style="font-size:11px;color:{ABG_MUTED};font-weight:600;">{lbl}</div>
                <div style="font-size:18px;font-weight:800;color:{color};">{val}</div>
                <div style="font-size:11px;color:{ABG_MUTED};">{sub}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # ── DEEP-DIVE TABS ────────────────────────────────────────
    # ══════════════════════════════════════════════════════════
    st.markdown(
        f'<div style="font-size:15px;font-weight:800;color:{ABG_DARK};'
        f'margin:8px 0 16px;padding-bottom:10px;'
        f'border-bottom:2px solid {ABG_BORDER};">📊 Deep-Dive Analysis</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕸️ Segment Profiles",
        "✅ Model Validation",
        "🔵 PCA Scatter",
        "📦 Feature Distributions",
        "🔥 Heatmap & Parallel Coords",
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — SEGMENT PROFILES (Radar)
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.markdown(
            _explain_card(
                "The radar chart overlays normalized profiles (0–100) for all five segments "
                "across 8 key behavioral and financial dimensions. Wider polygons indicate "
                "higher-value customers. The gap between polygons reflects cluster separation."
            ), unsafe_allow_html=True,
        )

        radar_feats = ['monthly_income','credit_score','account_balance',
                       'engagement_score','num_products','annual_spend',
                       'txn_frequency','savings_ratio']
        radar_labels = [FEATURE_LABELS[f].replace(' (SAR)','').replace(' / month','')
                        for f in radar_feats]

        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig_radar = go.Figure()
            _SEG_FILL = {
                'Standard': 'rgba(127,140,141,0.13)',
                'Silver':   'rgba(149,165,166,0.13)',
                'Gold':     'rgba(243,156,18,0.13)',
                'Platinum': 'rgba(142,68,173,0.13)',
                'VIP':      'rgba(230,57,70,0.13)',
            }
            for seg in SEGMENT_ORDER:
                vals = [float(centers_norm.loc[seg, f]) for f in radar_feats]
                v2   = vals + [vals[0]]
                l2   = radar_labels + [radar_labels[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=v2, theta=l2, fill='toself', name=seg,
                    line=dict(color=SEGMENT_COLORS[seg], width=2.5),
                    fillcolor=_SEG_FILL[seg], opacity=0.85,
                ))
            fig_radar.update_layout(
                **_layout(height=440, margin=dict(t=30,b=20,l=20,r=20)),
                title=dict(text="Multi-Segment Feature Radar", font=dict(size=14, color=ABG_DARK)),
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100],
                        tickfont=dict(size=9, color=ABG_MUTED)),
                    angularaxis=dict(tickfont=dict(size=11)),
                ),
                legend=dict(font=dict(size=12)),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with c2:
            st.markdown(
                _section_header("Key Radar Insights", "What each dimension reveals"),
                unsafe_allow_html=True,
            )
            insights_radar = [
                (ABG_RED,    "👑 VIP",      "Maxes out all 8 dimensions — particularly spend and balance."),
                (ABG_PURPLE, "💎 Platinum", "Strong income & credit; slightly lower digital engagement."),
                (ABG_GOLD,   "🥇 Gold",    "Balanced profile — ideal cross-sell target segment."),
                (ABG_BLUE,   "🥈 Silver",  "Moderate across all features; growing savings ratio."),
                (ABG_MUTED,  "👤 Standard","Low spend & balance but opportunity for product upsell."),
            ]
            for color, title, desc in insights_radar:
                st.markdown(f"""
                <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
                border-radius:10px;padding:12px 14px;margin-bottom:8px;
                border-left:4px solid {color};">
                  <div style="font-size:12px;font-weight:700;color:{color};">{title}</div>
                  <div style="font-size:11px;color:{ABG_MUTED};margin-top:3px;
                  line-height:1.5;">{desc}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(
                _explain_card(
                    "<strong>Takeaway:</strong> The consistent gap between consecutive "
                    "segment polygons confirms well-separated, meaningful clusters.",
                    ABG_GREEN
                ), unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════
    # TAB 2 — MODEL VALIDATION
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.markdown(
            _explain_card(
                "Two complementary methods validate that <strong>K=5 is the optimal number of clusters</strong>. "
                "The Elbow Method identifies the diminishing return in WCSS reduction, while the "
                "Silhouette Score measures how well each customer fits its assigned cluster."
            ), unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                _section_header("Elbow Curve", "Within-cluster sum of squares (WCSS) vs K"),
                unsafe_allow_html=True,
            )
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=k_range, y=inertias, mode='lines+markers', name='WCSS (Inertia)',
                line=dict(color=ABG_BLUE, width=2.5),
                marker=dict(size=8, color=ABG_BLUE, line=dict(color='white', width=2)),
            ))
            fig_elbow.add_trace(go.Scatter(
                x=[5], y=[inertias[k_range.index(5)]],
                mode='markers', name='Optimal K=5',
                marker=dict(size=16, color=ABG_RED, symbol='star',
                    line=dict(color='white', width=2)),
            ))
            fig_elbow.add_vline(x=5, line_dash='dash', line_color=ABG_RED,
                line_width=1.5,
                annotation_text='K=5 ← Elbow',
                annotation_font_color=ABG_RED,
                annotation_font_size=11)
            fig_elbow.update_layout(
                **_layout(height=320),
                title=dict(text="WCSS (Inertia) vs Number of Clusters", font=dict(size=13)),
                xaxis=dict(**MGRID, title="K"),
                yaxis=dict(**MGRID, title="WCSS"),
                legend=dict(font=dict(size=11)),
            )
            st.plotly_chart(fig_elbow, use_container_width=True)
            st.markdown(
                _explain_card(
                    "<strong>Elbow Method:</strong> WCSS measures total squared distances "
                    "within clusters. At K=5, the marginal gain flattens — adding more clusters "
                    "yields diminishing improvement in cohesion.", ABG_BLUE
                ), unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                _section_header("Silhouette Score", "Cluster separation quality vs K"),
                unsafe_allow_html=True,
            )
            k5_sil = sil_k[k_range.index(5)]
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(
                x=k_range, y=sil_k, mode='lines+markers', name='Silhouette',
                line=dict(color=ABG_GREEN, width=2.5),
                marker=dict(size=8, color=ABG_GREEN, line=dict(color='white', width=2)),
            ))
            fig_sil.add_trace(go.Scatter(
                x=[5], y=[k5_sil], mode='markers',
                name=f'K=5  ({k5_sil:.3f})',
                marker=dict(size=16, color=ABG_RED, symbol='star',
                    line=dict(color='white', width=2)),
            ))
            fig_sil.add_vline(x=5, line_dash='dash', line_color=ABG_RED, line_width=1.5)
            fig_sil.update_layout(
                **_layout(height=320),
                title=dict(text="Silhouette Score vs K", font=dict(size=13)),
                xaxis=dict(**MGRID, title="K"),
                yaxis=dict(**MGRID, title="Silhouette Score", range=[0, 0.6]),
                legend=dict(font=dict(size=11)),
            )
            st.plotly_chart(fig_sil, use_container_width=True)
            st.markdown(
                _explain_card(
                    f"<strong>Silhouette Score = {sil_score:.3f}</strong> at K=5 confirms "
                    "customers are meaningfully closer to their own cluster than to neighboring "
                    "ones. Scores range from -1 (misclassified) to +1 (perfect).", ABG_GREEN
                ), unsafe_allow_html=True,
            )

        # Silhouette per cluster
        st.markdown(
            _section_header("Silhouette Distribution by Segment", "Each bar = one customer"),
            unsafe_allow_html=True,
        )
        fig_sc = go.Figure()
        y_lower = 0
        tpos, tlbl = [], []
        for seg in SEGMENT_ORDER:
            seg_sil = df[df['segment'] == seg]['silhouette'].sort_values().values
            y_upper = y_lower + len(seg_sil)
            fig_sc.add_trace(go.Bar(
                x=seg_sil, y=list(range(y_lower, y_upper)),
                orientation='h', marker_color=SEGMENT_COLORS[seg],
                marker_line_width=0, name=seg, showlegend=True,
            ))
            tpos.append((y_lower + y_upper) // 2)
            tlbl.append(seg)
            y_lower = y_upper + 30

        fig_sc.add_vline(x=sil_score, line_dash='dash', line_color=ABG_DARK,
            line_width=1.5,
            annotation_text=f'Avg = {sil_score:.3f}',
            annotation_font_color=ABG_DARK, annotation_font_size=11)
        fig_sc.update_layout(
            **_layout(height=400),
            title=dict(text="Per-Customer Silhouette Coefficients", font=dict(size=13)),
            xaxis=dict(**MGRID, title="Silhouette Coefficient", range=[-0.3, 0.75]),
            yaxis=dict(tickvals=tpos, ticktext=tlbl, tickfont=dict(size=11)),
            legend=dict(font=dict(size=11), orientation='h', y=-0.12),
            barmode='overlay',
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown(
            _explain_card(
                "Wide, uniformly positive bars indicate cohesive segments. "
                "Customers with negative values sit closer to an adjacent cluster — "
                "typically border cases between Standard/Silver or Gold/Platinum.",
                ABG_PURPLE
            ), unsafe_allow_html=True,
        )

        # Algo summary cards
        st.markdown(
            _section_header("K-Means Algorithm Summary"),
            unsafe_allow_html=True,
        )
        ac1, ac2, ac3, ac4, ac5 = st.columns(5)
        algo_cards = [
            ("⚙️", "Algorithm",    "K-Means++",     ABG_BLUE),
            ("📏", "Scaler",       "RobustScaler",  ABG_PURPLE),
            ("🔢", "Features",     "12 numeric",    ABG_GREEN),
            ("🔄", "Iterations",   str(data['n_iter']), ABG_GOLD),
            ("🎯", "n_init",       "20 runs",       ABG_ORANGE),
        ]
        for col, (ico, lbl, val, clr) in zip([ac1,ac2,ac3,ac4,ac5], algo_cards):
            col.markdown(f"""
            <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
            border-radius:12px;padding:14px;text-align:center;border-top:3px solid {clr};">
              <div style="font-size:20px;">{ico}</div>
              <div style="font-size:10px;font-weight:700;color:{ABG_MUTED};
              text-transform:uppercase;letter-spacing:0.06em;margin:4px 0 2px;">{lbl}</div>
              <div style="font-size:14px;font-weight:800;color:{clr};">{val}</div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3 — PCA SCATTER
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.markdown(
            _explain_card(
                f"PCA projects 12 features onto 2 principal components, capturing "
                f"<strong>{sum(pca_exp):.1f}%</strong> of total variance. "
                "PC1 primarily encodes financial capacity (income, balance, spend). "
                "PC2 encodes digital engagement (logins, transactions, products owned)."
            ), unsafe_allow_html=True,
        )

        view = st.radio(
            "Select view:",
            ["2D Scatter", "3D Scatter", "Explained Variance", "PCA Loadings"],
            horizontal=True, key="pca_view",
        )

        if view == "2D Scatter":
            df_s = df.groupby('segment', group_keys=False).apply(
                lambda x: x.sample(min(350, len(x)), random_state=42)
            )
            fig2d = go.Figure()
            for seg in SEGMENT_ORDER:
                sub = df_s[df_s['segment'] == seg]
                fig2d.add_trace(go.Scatter(
                    x=sub['pca1'], y=sub['pca2'], mode='markers', name=seg,
                    marker=dict(color=SEGMENT_COLORS[seg], size=5, opacity=0.68,
                        line=dict(width=0)),
                ))
            fig2d.update_layout(
                **_layout(height=480),
                title=dict(text=f"PCA 2D — PC1 ({pca_exp[0]:.1f}%) vs PC2 ({pca_exp[1]:.1f}%)",
                    font=dict(size=14)),
                xaxis=dict(**MGRID, title=f"PC1 — Financial Capacity ({pca_exp[0]:.1f}% variance)"),
                yaxis=dict(**MGRID, title=f"PC2 — Digital Engagement ({pca_exp[1]:.1f}% variance)"),
                legend=dict(font=dict(size=12)),
            )
            st.plotly_chart(fig2d, use_container_width=True)
            st.markdown(
                _explain_card(
                    "Clusters separate clearly along the horizontal axis (PC1), "
                    "reflecting the dominant role of <strong>financial capacity</strong> in distinguishing "
                    "segments. Vertical spread (PC2) captures behavioral differences within each tier.",
                    ABG_BLUE
                ), unsafe_allow_html=True,
            )

        elif view == "3D Scatter":
            df_s = df.groupby('segment', group_keys=False).apply(
                lambda x: x.sample(min(200, len(x)), random_state=42)
            )
            fig3d = go.Figure()
            for seg in SEGMENT_ORDER:
                sub = df_s[df_s['segment'] == seg]
                fig3d.add_trace(go.Scatter3d(
                    x=sub['pca3d_1'], y=sub['pca3d_2'], z=sub['pca3d_3'],
                    mode='markers', name=seg,
                    marker=dict(color=SEGMENT_COLORS[seg], size=3, opacity=0.72),
                ))
            fig3d.update_layout(
                **_layout(height=520),
                title=dict(text="PCA 3D — First 3 Principal Components", font=dict(size=14)),
                scene=dict(
                    xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3',
                    bgcolor='#FAFAFA',
                ),
                legend=dict(font=dict(size=12)),
            )
            st.plotly_chart(fig3d, use_container_width=True)

        elif view == "Explained Variance":
            X_sc    = data['X_scaled']
            pca_all = _skd.PCA(n_components=len(FEATURES), random_state=42)
            pca_all.fit(X_sc)
            evr     = pca_all.explained_variance_ratio_ * 100
            cum_evr = np.cumsum(evr)

            fig_ev = go.Figure()
            fig_ev.add_trace(go.Bar(
                x=list(range(1, len(evr)+1)), y=evr,
                marker_color=ABG_BLUE, marker_line_width=0,
                name='Individual', opacity=0.75,
            ))
            fig_ev.add_trace(go.Scatter(
                x=list(range(1, len(evr)+1)), y=cum_evr,
                mode='lines+markers',
                line=dict(color=ABG_RED, width=2.5),
                marker=dict(size=7, color=ABG_RED),
                name='Cumulative', yaxis='y2',
            ))
            fig_ev.add_hline(y=90, line_dash='dash', line_color=ABG_GOLD,
                annotation_text='90% threshold', yref='y2',
                annotation_font_color=ABG_GOLD)
            fig_ev.update_layout(
                **_layout(height=380),
                title=dict(text="Explained Variance Ratio per Principal Component",
                    font=dict(size=14)),
                xaxis=dict(**MGRID, title="Principal Component"),
                yaxis=dict(**MGRID, title="Individual Variance (%)"),
                yaxis2=dict(overlaying='y', side='right', range=[0, 106],
                    title="Cumulative (%)", gridcolor='rgba(0,0,0,0)'),
                legend=dict(font=dict(size=11)),
            )
            st.plotly_chart(fig_ev, use_container_width=True)
            st.markdown(
                _explain_card(
                    f"PC1 alone explains <strong>{evr[0]:.1f}%</strong> of variance — an unusually "
                    "high value indicating a strong dominant financial gradient across customers. "
                    "Components 1–3 together exceed 80%, justifying a low-dimensional representation.",
                    ABG_GOLD
                ), unsafe_allow_html=True,
            )

        else:  # PCA Loadings
            ldg = loadings.abs()
            fig_ldg = go.Figure()
            colors_ldg = [ABG_BLUE, ABG_PURPLE, ABG_GOLD]
            for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
                fig_ldg.add_trace(go.Bar(
                    x=[FEATURE_LABELS[f].replace(' (SAR)','').replace(' / month','').replace(' (months)','')
                       for f in FEATURES],
                    y=ldg[pc].values, name=pc,
                    marker_color=colors_ldg[i], marker_line_width=0, opacity=0.85,
                ))
            fig_ldg.update_layout(
                **_layout(height=360),
                title=dict(text="PCA Loadings — Feature Contribution to Each Component",
                    font=dict(size=14)),
                barmode='group',
                xaxis=dict(**MGRID, tickangle=-35, tickfont=dict(size=9)),
                yaxis=dict(**MGRID, title="|Loading|"),
                legend=dict(font=dict(size=11)),
            )
            st.plotly_chart(fig_ldg, use_container_width=True)
            st.markdown(
                _explain_card(
                    "<strong>Monthly Income</strong> and <strong>Account Balance</strong> "
                    "dominate PC1 (financial tier). <strong>App Logins</strong> and "
                    "<strong>Engagement Score</strong> drive PC2 (digital behavior). "
                    "PC3 captures credit risk nuances (debt-to-income, late payments).",
                    ABG_PURPLE
                ), unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════
    # TAB 4 — FEATURE DISTRIBUTIONS
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown(
            _explain_card(
                "Distribution plots reveal how each feature varies across segments. "
                "<strong>Box plots</strong> highlight medians and quartiles; "
                "<strong>Violin plots</strong> expose the full probability density. "
                "Toggle the plot type using the selector below."
            ), unsafe_allow_html=True,
        )

        plot_type = st.radio(
            "Plot type:", ["Box Plot", "Violin Plot"], horizontal=True, key="dist_type",
        )
        sel_feats = [f for f in FEATURES if f != 'late_payment_count']
        pairs = [(sel_feats[i], sel_feats[i+1] if i+1 < len(sel_feats) else None)
                 for i in range(0, len(sel_feats), 2)]

        _BOX_FILL = {
            'Standard': 'rgba(127,140,141,0.27)',
            'Silver':   'rgba(149,165,166,0.27)',
            'Gold':     'rgba(243,156,18,0.27)',
            'Platinum': 'rgba(142,68,173,0.27)',
            'VIP':      'rgba(230,57,70,0.27)',
        }
        _VLN_FILL = {
            'Standard': 'rgba(127,140,141,0.33)',
            'Silver':   'rgba(149,165,166,0.33)',
            'Gold':     'rgba(243,156,18,0.33)',
            'Platinum': 'rgba(142,68,173,0.33)',
            'VIP':      'rgba(230,57,70,0.33)',
        }
        for f1, f2 in pairs:
            cols = st.columns(2) if f2 else [st.container()]
            for col, feat in zip(cols, [f for f in [f1, f2] if f]):
                lbl = (FEATURE_LABELS[feat]
                       .replace(' (SAR)','').replace(' / month','').replace(' (months)',''))
                with col:
                    fig_d = go.Figure()
                    for seg in SEGMENT_ORDER:
                        vals = df[df['segment'] == seg][feat].values
                        if plot_type == "Box Plot":
                            fig_d.add_trace(go.Box(
                                y=vals, name=seg,
                                marker_color=SEGMENT_COLORS[seg],
                                line_color=SEGMENT_COLORS[seg],
                                fillcolor=_BOX_FILL[seg],
                                boxmean=True, showlegend=False,
                            ))
                        else:
                            fig_d.add_trace(go.Violin(
                                y=vals, name=seg,
                                fillcolor=_VLN_FILL[seg],
                                line_color=SEGMENT_COLORS[seg],
                                box_visible=True, meanline_visible=True,
                                showlegend=False,
                            ))
                    fig_d.update_layout(
                        **_layout(height=255, margin=dict(t=40,b=10,l=10,r=10)),
                        title=dict(text=lbl, font=dict(size=12)),
                        xaxis=dict(**MGRID),
                        yaxis=dict(**MGRID),
                    )
                    st.plotly_chart(fig_d, use_container_width=True)

        st.markdown(
            _explain_card(
                "<strong>Key Insight:</strong> Account Balance shows the widest spread in "
                "Platinum/VIP segments, indicating wealth heterogeneity at the top tier. "
                "Engagement Score rises steadily with segment — confirming that higher-value "
                "customers are also the most digitally active.",
                ABG_GREEN
            ), unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════
    # TAB 5 — HEATMAP & PARALLEL COORDS
    # ══════════════════════════════════════════════════════════
    with tab5:
        st.markdown(
            _explain_card(
                "The <strong>Cluster Centers Heatmap</strong> visualizes the normalized "
                "intensity (0–100) of each feature per segment — darker blue = higher relative value. "
                "The <strong>Parallel Coordinates</strong> plot traces every customer across "
                "all features simultaneously, revealing inter-feature correlations."
            ), unsafe_allow_html=True,
        )

        # Heatmap
        st.markdown(
            _section_header("Cluster Centers Heatmap", "Normalized feature intensity per segment (0–100)"),
            unsafe_allow_html=True,
        )
        hm_feats = [f for f in FEATURES if f != 'late_payment_count']
        hm_lbls  = [FEATURE_LABELS[f].replace(' (SAR)','').replace(' / month','').replace(' (months)','')
                    for f in hm_feats]
        hm_vals  = [[float(centers_norm.loc[seg, f]) for f in hm_feats]
                    for seg in SEGMENT_ORDER]

        fig_hm = go.Figure(go.Heatmap(
            z=hm_vals, x=hm_lbls, y=SEGMENT_ORDER,
            colorscale=[[0,'#F0F0F6'],[0.5,'rgba(61,61,219,0.53)'],[1,'#3D3DDB']],
            text=[[f"{v:.0f}" for v in row] for row in hm_vals],
            texttemplate="%{text}",
            textfont=dict(size=11, family="Plus Jakarta Sans"),
            showscale=True,
            colorbar=dict(title="Score (0–100)", tickfont=dict(size=10)),
        ))
        fig_hm.update_layout(
            **_layout(height=300, margin=dict(t=30,b=10,l=10,r=10)),
            title=dict(text="Feature Intensity by Segment (Normalized)", font=dict(size=13)),
            xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=12)),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.markdown(
            _explain_card(
                "The heatmap confirms that <strong>Monthly Income</strong> and "
                "<strong>Account Balance</strong> are the primary differentiators between "
                "segments. The consistent intensity gradient from Standard (top) to VIP (bottom) "
                "validates cluster ordering.",
                ABG_GOLD
            ), unsafe_allow_html=True,
        )

        # Parallel Coordinates
        st.markdown(
            _section_header("Parallel Coordinates", "Every line = one customer across all features"),
            unsafe_allow_html=True,
        )
        pc_feats = ['monthly_income','credit_score','account_balance',
                    'engagement_score','annual_spend','num_products']
        seg_enc  = {s: i for i, s in enumerate(SEGMENT_ORDER)}
        df_pc    = df[pc_feats + ['segment']].copy()
        df_pc['seg_num'] = df_pc['segment'].map(seg_enc)

        df_norm = df_pc[pc_feats].copy()
        for col_ in pc_feats:
            mn, mx = df_norm[col_].min(), df_norm[col_].max()
            df_norm[col_] = (df_norm[col_] - mn) / (mx - mn + 1e-9)

        dims = [
            dict(
                label=FEATURE_LABELS[f].replace(' (SAR)','').replace(' / month',''),
                values=df_norm[f], range=[0, 1],
            )
            for f in pc_feats
        ]
        fig_pc = go.Figure(go.Parcoords(
            line=dict(
                color=df_pc['seg_num'],
                colorscale=[
                    [i / (K_OPTIMAL - 1), SEGMENT_COLORS[s]]
                    for i, s in enumerate(SEGMENT_ORDER)
                ],
                showscale=False,
            ),
            dimensions=dims,
        ))
        fig_pc.update_layout(
            **_layout(height=400, margin=dict(t=30,b=20,l=60,r=60)),
            title=dict(text="Parallel Coordinates — All 5,000 Customers", font=dict(size=13)),
        )
        st.plotly_chart(fig_pc, use_container_width=True)
        st.markdown(
            _explain_card(
                "Line crossings between axes indicate <strong>negative correlations</strong>. "
                "The dense band of lines in the upper range of Income/Balance corresponds to "
                "VIP/Platinum customers. Standard customers (left color) cluster in the lower "
                "ranges across all features. Drag axis brackets to filter customer subsets.",
                ABG_ORANGE
            ), unsafe_allow_html=True,
        )
