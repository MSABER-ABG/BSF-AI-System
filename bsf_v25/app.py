"""
BSF AI Banking Intelligence System — v3.1
Developed by Accord Business Group (ABG)
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="BSF AI Banking Intelligence System",
    page_icon="🏦", layout="wide",
    initial_sidebar_state="expanded",
)

from utils.styling import (
    inject_css, ABG_BLUE, ABG_DARK, ABG_MUTED,
    ABG_BORDER, ABG_PURPLE, ABG_GOLD, ABG_GREEN
)

st.markdown(inject_css(), unsafe_allow_html=True)

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit auto page links */
[data-testid="stSidebarNav"] { display: none !important; }

/* Remove default sidebar top padding */
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* Radio group: remove gap, full width */
section[data-testid="stSidebar"] [data-testid="stRadio"] { width: 100%; }
section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 0 !important;
    width: 100%;
}

/* Each radio label looks like a nav item */
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent !important;
    border: 1.5px solid transparent !important;
    border-radius: 10px !important;
    padding: 10px 12px !important;
    margin: 1px 6px !important;
    cursor: pointer !important;
    color: #555 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    display: flex !important;
    align-items: center !important;
    width: calc(100% - 12px) !important;
    transition: background 0.15s !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(61,61,219,0.06) !important;
    color: #1A1A2E !important;
    border-color: rgba(61,61,219,0.15) !important;
}
/* Selected radio label */
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-selected="true"],
section[data-testid="stSidebar"] [data-testid="stRadio"] label[aria-checked="true"] {
    background: rgba(61,61,219,0.08) !important;
    border-color: rgba(61,61,219,0.30) !important;
    color: #1A1A2E !important;
    font-weight: 700 !important;
}
/* Hide the radio circle */
section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none !important;
}
/* Hide radio widget label */
section[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Navigation pages ───────────────────────────────────────────
NAV_OPTIONS = [
    ("Pipeline",     "⏩  Full Pipeline",     "12-step AI flow"),
    ("Overview",     "🏠  Overview",         "System dashboard"),
    ("Module1",      "👥  Segmentation Model",          "Segmentation & NBA"),
    ("Segmentation", "🔍  Segmentation Analysis",      "K-Means · PCA · Charts"),
    ("Module2",      "📊  Behavior Analysis",          "Behavior Analysis"),
    ("Module3",      "💳  Credit Decision",          "Credit Decision"),
    ("Demo",         "🎯  Live Demo",         "Full Pipeline Demo"),
]

PAGE_KEYS   = [o[0] for o in NAV_OPTIONS]
PAGE_LABELS = [o[1] for o in NAV_OPTIONS]

if 'active_page' not in st.session_state:
    st.session_state.active_page = 'Overview'

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:

    # Logo
    st.markdown(f"""
    <div style="padding:4px 12px 10px;">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:42px;height:42px;border-radius:10px;
        background:linear-gradient(135deg,{ABG_BLUE},{ABG_PURPLE});
        display:flex;align-items:center;justify-content:center;
        font-size:15px;font-weight:800;color:white;flex-shrink:0;">BSF</div>
        <div>
          <div style="font-size:13px;font-weight:800;color:{ABG_DARK};">BSF AI System</div>
          <div style="font-size:10px;color:{ABG_MUTED};">by Accord Business Group</div>
        </div>
      </div>
    </div>
    <div style="height:1px;background:#E8E8F0;margin:0 8px 10px;"></div>
    """, unsafe_allow_html=True)

    # Current index
    cur_idx = PAGE_KEYS.index(st.session_state.active_page) \
              if st.session_state.active_page in PAGE_KEYS else 0

    # Radio navigation — NO buttons, NO blue boxes
    selected_label = st.radio(
        "Navigation",
        options=PAGE_LABELS,
        index=cur_idx,
        label_visibility="collapsed",
        key="sidebar_nav",
    )

    # Update active page when selection changes
    selected_key = PAGE_KEYS[PAGE_LABELS.index(selected_label)]
    if selected_key != st.session_state.active_page:
        st.session_state.active_page = selected_key
        st.rerun()

    # Footer
    st.markdown(f"""
    <div style="height:1px;background:#E8E8F0;margin:10px 8px 8px;"></div>
    <div style="padding:10px 12px;background:#F8F8FF;border-radius:8px;
    border:1px solid {ABG_BORDER};margin:0 6px 8px;">
      <div style="font-size:10px;font-weight:700;color:{ABG_MUTED};
      text-transform:uppercase;letter-spacing:0.07em;margin-bottom:5px;">System Status</div>
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">
        <div style="width:6px;height:6px;border-radius:50%;background:#27AE60;"></div>
        <span style="font-size:11px;font-weight:600;color:{ABG_DARK};">All Modules Active</span>
      </div>
      <div style="font-size:10px;color:{ABG_MUTED};">5,000 customers · 3 modules</div>
    </div>
    <div style="text-align:center;font-size:10px;color:{ABG_MUTED};padding-bottom:8px;">
      v3.1 · June 2025 · ABG
    </div>
    """, unsafe_allow_html=True)


# ── Page Router ────────────────────────────────────────────────
page = st.session_state.active_page
if page == 'Overview':
    from pages import overview; overview.render()
elif page == 'Pipeline':
    from pages import pipeline; pipeline.render()
elif page == 'Module1':
    from pages import module1; module1.render()
elif page == 'Segmentation':
    from pages import segmentation; segmentation.render()
elif page == 'Module2':
    from pages import module2; module2.render()
elif page == 'Module3':
    from pages import module3; module3.render()
elif page == 'Demo':
    from pages import demo; demo.render()
