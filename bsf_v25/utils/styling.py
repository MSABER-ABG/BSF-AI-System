"""BSF AI — Streamlit Styling (matches BankGuard AI ABG theme)"""

def inject_css():
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}
.main { background: #F0F0F6 !important; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1300px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E8E8F0 !important;
}
[data-testid="stSidebar"] .block-container { padding: 1rem !important; }

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8E8F0;
    border-radius: 12px;
    padding: 16px 20px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stMetric"] label {
    color: #6B6B8A !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    color: #1A1A2E !important;
    font-size: 26px !important;
    font-weight: 800 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #3D3DDB !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #2D2DC4 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(61,61,219,0.25) !important;
}

/* ── Selectbox / Input ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input,
[data-testid="stTextInput"] > div > div > input {
    background: #FFFFFF !important;
    border: 1.5px solid #E8E8F0 !important;
    border-radius: 8px !important;
    color: #1A1A2E !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] > div > div > input:focus,
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #3D3DDB !important;
    box-shadow: 0 0 0 3px rgba(61,61,219,0.1) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    color: #6B6B8A !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 20px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #3D3DDB !important;
    border-bottom: 2px solid #3D3DDB !important;
}

/* ── Divider ── */
hr { border-color: #E8E8F0 !important; margin: 1rem 0 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E8E8F0 !important;
    border-radius: 10px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] .st-bo { background: #3D3DDB !important; }

/* ── Radio ── */
[data-testid="stRadio"] label { font-weight: 600 !important; color: #1A1A2E !important; }
</style>
"""

# ── Color constants ───────────────────────────────────────────
ABG_BLUE    = '#3D3DDB'
ABG_DARK    = '#1A1A2E'
ABG_BG      = '#F0F0F6'
ABG_WHITE   = '#FFFFFF'
ABG_BORDER  = '#E8E8F0'
ABG_MUTED   = '#6B6B8A'
ABG_RED     = '#E63946'
ABG_GREEN   = '#27AE60'
ABG_GOLD    = '#F39C12'
ABG_PURPLE  = '#8E44AD'
ABG_ORANGE  = '#E67E22'

SEGMENT_COLORS = {
    'Standard':  '#7F8C8D',
    'Silver':    '#95A5A6',
    'Gold':      '#F39C12',
    'Platinum':  '#8E44AD',
    'VIP':       '#E63946',
}

RISK_COLORS = {
    'Very Low':  '#27AE60',
    'Low':       '#2ECC71',
    'Medium':    '#F39C12',
    'High':      '#E67E22',
    'Very High': '#E63946',
}

# ── HTML Components ───────────────────────────────────────────

def card(content: str, border_color: str = '#E8E8F0', padding: str = '20px') -> str:
    return f"""
    <div style="background:#FFFFFF;border:1.5px solid {border_color};border-radius:12px;
    padding:{padding};box-shadow:0 1px 4px rgba(0,0,0,0.06);margin-bottom:12px;">
    {content}
    </div>"""

def metric_card(label: str, value: str, sub: str = '', color: str = ABG_BLUE,
                badge: str = '', badge_color: str = '') -> str:
    badge_html = f'<span style="display:inline-block;margin-top:8px;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:600;background:{badge_color}22;color:{badge_color};">{badge}</span>' if badge else ''
    return f"""
    <div style="background:#FFFFFF;border:1.5px solid #E8E8F0;border-radius:12px;
    padding:20px;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
      <div style="font-size:11px;font-weight:700;color:{ABG_MUTED};text-transform:uppercase;letter-spacing:0.07em;">{label}</div>
      <div style="font-size:28px;font-weight:800;color:{color};margin:6px 0 2px;">{value}</div>
      <div style="font-size:12px;color:{ABG_MUTED};">{sub}</div>
      {badge_html}
    </div>"""

def segment_pill(segment: str) -> str:
    color = SEGMENT_COLORS.get(segment, ABG_MUTED)
    icons = {'Standard': '●', 'Silver': '●', 'Gold': '★', 'Platinum': '◆', 'VIP': '♛'}
    icon  = icons.get(segment, '●')
    return f'<span style="display:inline-flex;align-items:center;gap:6px;padding:6px 16px;border-radius:20px;border:1.5px solid {color};color:{color};font-size:14px;font-weight:700;background:{color}11;">{icon} {segment}</span>'

def risk_pill(risk: str) -> str:
    color = RISK_COLORS.get(risk, ABG_MUTED)
    return f'<span style="display:inline-block;padding:4px 14px;border-radius:6px;font-size:13px;font-weight:700;background:{color}22;color:{color};">{risk}</span>'

def progress_bar(value: float, color: str = ABG_BLUE, height: str = '8px') -> str:
    pct = min(100, max(0, value))
    return f"""
    <div style="background:#F0F0F6;border-radius:4px;height:{height};overflow:hidden;margin-top:4px;">
      <div style="height:100%;width:{pct}%;background:{color};border-radius:4px;transition:width 0.8s ease;"></div>
    </div>"""

def channel_badge(channel: str) -> str:
    icons = {'Push Notification': '🔔', 'WhatsApp': '💬', 'SMS': '📱',
             'RM': '👤', 'Branch Agent': '🏦', 'Call Center': '📞', 'Avatar': '🤖'}
    icon = icons.get(channel, '📡')
    return f'<span style="display:inline-flex;align-items:center;gap:6px;padding:5px 14px;border-radius:6px;font-size:13px;font-weight:600;background:#3D3DDB15;border:1px solid #3D3DDB30;color:{ABG_BLUE};">{icon} {channel}</span>'

def section_header(title: str, subtitle: str = '') -> str:
    sub = f'<p style="color:{ABG_MUTED};font-size:13px;margin-top:4px;">{subtitle}</p>' if subtitle else ''
    return f'<h2 style="color:{ABG_DARK};font-size:20px;font-weight:800;margin:0;">{title}</h2>{sub}'

def info_box(text: str, color: str = ABG_BLUE) -> str:
    return f'<div style="padding:12px 16px;border-radius:8px;border-left:3px solid {color};background:{color}0d;font-size:13px;color:{ABG_DARK};margin:8px 0;">{text}</div>'

def approval_box(approval: dict) -> str:
    colors = {'auto': ABG_GREEN, 'manual': ABG_ORANGE, 'senior': ABG_RED}
    color  = colors.get(approval['level'], ABG_MUTED)
    return f"""
    <div style="border:1.5px solid {color};border-radius:10px;padding:18px;background:{color}0a;display:flex;align-items:center;gap:14px;margin-top:12px;">
      <div style="font-size:28px;">{approval['icon']}</div>
      <div>
        <div style="font-weight:700;font-size:15px;color:{ABG_DARK};">{approval['status']}</div>
        <div style="font-size:12px;color:{ABG_MUTED};margin-top:3px;">{approval['desc']}</div>
      </div>
    </div>"""

def nba_card(rec: dict, rank: int) -> str:
    colors = [ABG_BLUE, ABG_GOLD]
    color  = colors[rank - 1] if rank <= len(colors) else ABG_MUTED
    score  = rec['match_score']
    reasons_html = ''.join([
        f'<li style="display:flex;align-items:flex-start;gap:6px;font-size:12px;color:{ABG_MUTED};margin-bottom:4px;">'
        f'<span style="color:{color};font-size:14px;">›</span>{rc}</li>'
        for rc in rec['reason_codes']
    ])
    return f"""
    <div style="background:#FFFFFF;border:1.5px solid {color}33;border-radius:10px;padding:16px;
    border-top:3px solid {color};margin-bottom:12px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <div style="font-size:13px;font-weight:700;color:{ABG_DARK};">#{rank} {rec['product_name']}</div>
        <div style="font-size:20px;font-weight:800;color:{color};">{score}%</div>
      </div>
      <div style="background:#F0F0F6;border-radius:4px;height:6px;overflow:hidden;margin-bottom:12px;">
        <div style="height:100%;width:{score}%;background:{color};border-radius:4px;"></div>
      </div>
      <div style="font-size:11px;color:{ABG_MUTED};margin-bottom:4px;">Recommended Limit: <strong style="color:{ABG_DARK};">{rec['recommended_limit']}</strong></div>
      <ul style="list-style:none;padding:0;margin:8px 0 0;">{reasons_html}</ul>
    </div>"""

def sidebar_logo() -> str:
    return f"""
    <div style="padding:16px 8px 20px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
        <div style="width:40px;height:40px;border-radius:10px;background:linear-gradient(135deg,{ABG_BLUE},{ABG_PURPLE});
        display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:800;color:white;">BSF</div>
        <div>
          <div style="font-size:13px;font-weight:800;color:{ABG_DARK};">BSF AI System</div>
          <div style="font-size:10px;color:{ABG_MUTED};">by Accord Business Group</div>
        </div>
      </div>
      <div style="height:1px;background:#E8E8F0;margin-bottom:16px;"></div>
    </div>"""

def page_header(title: str, subtitle: str, icon: str = '🏦') -> str:
    return f"""
    <div style="background:linear-gradient(135deg,{ABG_BLUE}08,{ABG_PURPLE}05);border:1px solid {ABG_BORDER};
    border-radius:14px;padding:28px 32px;margin-bottom:24px;">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="font-size:36px;">{icon}</div>
        <div>
          <h1 style="font-size:22px;font-weight:800;color:{ABG_DARK};margin:0;">{title}</h1>
          <p style="font-size:13px;color:{ABG_MUTED};margin:4px 0 0;">{subtitle}</p>
        </div>
      </div>
    </div>"""
