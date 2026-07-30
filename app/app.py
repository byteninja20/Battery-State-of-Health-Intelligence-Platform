import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import json
import shap
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='BatteryIQ — State of Health Platform',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── DESIGN SYSTEM — Theme-Aware CSS ──────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════
   TOKEN SYSTEM — adapts to Streamlit light + dark mode
   ═══════════════════════════════════════════════════════ */
:root {
    --brand-primary: #2563EB;
    --brand-dark:    #1D4ED8;
    --brand-light:   #DBEAFE;

    --status-green:  #16A34A;
    --status-yellow: #CA8A04;
    --status-orange: #EA580C;
    --status-red:    #DC2626;

    --status-green-bg:  #F0FDF4;
    --status-yellow-bg: #FEFCE8;
    --status-orange-bg: #FFF7ED;
    --status-red-bg:    #FEF2F2;

    --status-green-border:  #86EFAC;
    --status-yellow-border: #FDE047;
    --status-orange-border: #FDBA74;
    --status-red-border:    #FCA5A5;
}

/* Import Inter */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset Streamlit defaults ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.5rem 2rem 2rem !important;
    max-width: 1400px !important;
}

/* ── App header ── */
.biq-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 60%, #3B82F6 100%);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.biq-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
}
.biq-header::after {
    content: '';
    position: absolute;
    bottom: -60px; right: 80px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.biq-header-inner { position: relative; z-index: 1; }
.biq-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #93C5FD;
    margin-bottom: 0.4rem;
}
.biq-header h1 {
    font-size: 1.65rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 0.25rem;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.biq-header p {
    font-size: 0.82rem;
    color: #BFDBFE;
    margin: 0 0 0.75rem;
    font-weight: 400;
}
.biq-badge-row { display: flex; flex-wrap: wrap; gap: 6px; }
.biq-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.7rem;
    font-weight: 500;
    color: #DBEAFE;
    backdrop-filter: blur(4px);
}

/* ── Sidebar redesign ── */
[data-testid="stSidebar"] {
    background: var(--background-color, #0F1C2E) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] > div { padding: 1.25rem 1rem; }

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.sidebar-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #2563EB, #3B82F6);
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.sidebar-logo-text {
    font-size: 1rem;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.01em;
}
.sidebar-logo-sub {
    font-size: 0.68rem;
    color: #64748B;
    margin-top: 1px;
}
.sidebar-section {
    margin-bottom: 1.25rem;
}
.sidebar-section-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.6rem;
}
.sidebar-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.sidebar-row-label { font-size: 0.75rem; color: #94A3B8; }
.sidebar-row-value { font-size: 0.75rem; font-weight: 600; color: #E2E8F0; }
.sidebar-target-item {
    font-size: 0.72rem;
    color: #64748B;
    padding: 3px 0;
    display: flex;
    align-items: center;
    gap: 5px;
}
.sidebar-target-item::before {
    content: '';
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #3B82F6;
    flex-shrink: 0;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1rem 0;
}
.sidebar-dataset {
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(37, 99, 235, 0.18);
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
}
.sidebar-dataset-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: #93C5FD;
    margin-bottom: 4px;
}
.sidebar-dataset-text {
    font-size: 0.68rem;
    color: #64748B;
    line-height: 1.5;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid rgba(148,163,184,0.15) !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    padding: 0.6rem 1.25rem !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    background: transparent !important;
    transition: all 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #2563EB !important;
    border-bottom-color: #93C5FD !important;
}
.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    border-bottom-color: #2563EB !important;
    font-weight: 600 !important;
    background: transparent !important;
}

/* ── Section labels ── */
.biq-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 0.75rem;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid rgba(148,163,184,0.15);
    display: flex;
    align-items: center;
    gap: 7px;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.25rem;
}
.kpi-card {
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    background: var(--kpi-bg, rgba(255,255,255,0.05));
    border: 1px solid var(--kpi-border, rgba(148,163,184,0.12));
    position: relative;
    overflow: hidden;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.kpi-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.kpi-accent-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-label {
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-sub {
    font-size: 0.7rem;
    color: #94A3B8;
    font-weight: 400;
}
.kpi-green  .kpi-value { color: #16A34A; }
.kpi-yellow .kpi-value { color: #CA8A04; }
.kpi-orange .kpi-value { color: #EA580C; }
.kpi-red    .kpi-value { color: #DC2626; }
.kpi-blue   .kpi-value { color: #2563EB; }
.kpi-green  .kpi-accent-bar { background: #16A34A; }
.kpi-yellow .kpi-accent-bar { background: #CA8A04; }
.kpi-orange .kpi-accent-bar { background: #EA580C; }
.kpi-red    .kpi-accent-bar { background: #DC2626; }
.kpi-blue   .kpi-accent-bar { background: #2563EB; }

/* ── Health Alert Cards ── */
.health-alert {
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    margin: 0.75rem 0;
    border-left: 4px solid;
}
.health-alert-title {
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.health-alert-body {
    font-size: 0.8rem;
    line-height: 1.65;
    opacity: 0.9;
}
.health-alert-body li {
    margin-bottom: 3px;
    list-style: none;
    padding-left: 0;
}
.health-alert-body li::before {
    content: '→ ';
    opacity: 0.6;
}
.alert-green  { background: var(--status-green-bg);  border-color: var(--status-green);  color: #14532D; }
.alert-yellow { background: var(--status-yellow-bg); border-color: var(--status-yellow); color: #713F12; }
.alert-orange { background: var(--status-orange-bg); border-color: var(--status-orange); color: #7C2D12; }
.alert-red    { background: var(--status-red-bg);    border-color: var(--status-red);    color: #7F1D1D; }

/* ── Input group cards ── */
.input-group {
    background: rgba(148,163,184,0.04);
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 10px;
    padding: 1rem 1.1rem 0.5rem;
    margin-bottom: 0.75rem;
}
.input-group-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #2563EB;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Derived metric pill ── */
.derived-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(37,99,235,0.08);
    border: 1px solid rgba(37,99,235,0.15);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #2563EB;
    margin-top: 0.4rem;
}

/* ── Progress bar ── */
.biq-progress-wrap {
    margin: 0.75rem 0;
}
.biq-progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #94A3B8;
    margin-bottom: 5px;
    font-weight: 500;
}
.biq-progress-track {
    height: 8px;
    border-radius: 6px;
    background: rgba(148,163,184,0.15);
    overflow: hidden;
    position: relative;
}
.biq-progress-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}
.biq-eol-marker {
    position: absolute;
    left: 80%;
    top: -4px;
    bottom: -4px;
    width: 2px;
    background: #EF4444;
    border-radius: 2px;
}
.biq-eol-label {
    font-size: 0.65rem;
    color: #EF4444;
    text-align: right;
    margin-top: 3px;
    font-weight: 500;
}

/* ── Info callout ── */
.biq-info {
    background: rgba(37,99,235,0.06);
    border: 1px solid rgba(37,99,235,0.15);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.78rem;
    color: #3B82F6;
    margin-bottom: 1rem;
    display: flex;
    gap: 8px;
    align-items: flex-start;
    line-height: 1.5;
}

/* ── Breakdown table ── */
.breakdown-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
    margin-top: 0.5rem;
}
.breakdown-table th {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94A3B8;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid rgba(148,163,184,0.15);
    text-align: left;
}
.breakdown-table td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid rgba(148,163,184,0.07);
    color: inherit;
    vertical-align: top;
}
.breakdown-table tr:last-child td { border-bottom: none; }
.factor-tag {
    display: inline-block;
    background: rgba(37,99,235,0.08);
    border: 1px solid rgba(37,99,235,0.15);
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #2563EB;
}

/* ── Fleet upload zone ── */
.upload-zone {
    border: 2px dashed rgba(37,99,235,0.25);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    background: rgba(37,99,235,0.03);
    margin-bottom: 1rem;
    transition: border-color 0.2s ease;
}
.upload-zone:hover { border-color: rgba(37,99,235,0.5); }
.upload-zone-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.upload-zone-title { font-size: 0.9rem; font-weight: 600; color: #1E40AF; margin-bottom: 0.25rem; }
.upload-zone-sub { font-size: 0.75rem; color: #94A3B8; }

/* ── India mode ── */
.india-header {
    background: linear-gradient(135deg, #FF6B35 0%, #F7931E 50%, #FF6B35 100%);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.india-header::before {
    content: '🇮🇳';
    position: absolute;
    right: 1.5rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 3rem;
    opacity: 0.3;
}
.india-header h3 {
    font-size: 1.1rem;
    font-weight: 700;
    color: white;
    margin: 0 0 0.2rem;
}
.india-header p { font-size: 0.78rem; color: rgba(255,255,255,0.8); margin: 0; }

/* ── Recommendation item ── */
.rec-item {
    display: flex;
    gap: 0.75rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid rgba(148,163,184,0.1);
}
.rec-item:last-child { border-bottom: none; }
.rec-icon {
    width: 32px; height: 32px;
    border-radius: 8px;
    background: rgba(37,99,235,0.08);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.rec-title { font-size: 0.8rem; font-weight: 600; margin-bottom: 2px; }
.rec-body  { font-size: 0.75rem; color: #94A3B8; line-height: 1.5; }

/* ── Buttons ── */
.stButton > button {
    background: #2563EB !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    padding: 0.6rem 1.5rem !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important;
}
.stButton > button:hover {
    background: #1D4ED8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Slider label fix ── */
.stSlider label p, .stSelectbox label p {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
}

/* ── Divider ── */
hr { border-color: rgba(148,163,184,0.12) !important; margin: 1rem 0 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
    background: rgba(148,163,184,0.05) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(148,163,184,0.12) !important;
}
.streamlit-expanderContent {
    border: 1px solid rgba(148,163,184,0.1) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 0.75rem 1rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load Artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
    with open(os.path.join(base, 'soh_model.pkl'), 'rb') as f:
        soh_model = pickle.load(f)
    with open(os.path.join(base, 'rul_model.pkl'), 'rb') as f:
        rul_model = pickle.load(f)
    with open(os.path.join(base, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(base, 'feature_cols.json'), 'r') as f:
        feature_cols = json.load(f)
    return soh_model, rul_model, scaler, feature_cols

soh_model, rul_model, scaler, feature_cols = load_artifacts()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">⚡</div>
        <div>
            <div class="sidebar-logo-text">BatteryIQ</div>
            <div class="sidebar-logo-sub">SoH Intelligence Platform</div>
        </div>
    </div>

    <div class="sidebar-section">
        <div class="sidebar-section-label">Model Performance</div>
        <div class="sidebar-row">
            <span class="sidebar-row-label">SoH Model R²</span>
            <span class="sidebar-row-value">0.9421</span>
        </div>
        <div class="sidebar-row">
            <span class="sidebar-row-label">RUL Model R²</span>
            <span class="sidebar-row-value">0.8594</span>
        </div>
        <div class="sidebar-row">
            <span class="sidebar-row-label">Training cycles</span>
            <span class="sidebar-row-value">504</span>
        </div>
        <div class="sidebar-row">
            <span class="sidebar-row-label">Test battery</span>
            <span class="sidebar-row-value">B0018</span>
        </div>
        <div class="sidebar-row">
            <span class="sidebar-row-label">SoH MAE</span>
            <span class="sidebar-row-value">1.69%</span>
        </div>
        <div class="sidebar-row">
            <span class="sidebar-row-label">RUL MAE</span>
            <span class="sidebar-row-value">6.19 cycles</span>
        </div>
    </div>

    <hr class="sidebar-divider">

    <div class="sidebar-section">
        <div class="sidebar-section-label">Industry Targets</div>
        <div class="sidebar-target-item">Replus Engitech</div>
        <div class="sidebar-target-item">Exide Industries</div>
        <div class="sidebar-target-item">Amara Raja Group</div>
        <div class="sidebar-target-item">JSW Neo Energy</div>
        <div class="sidebar-target-item">Tata Power / Agratas</div>
    </div>

    <hr class="sidebar-divider">

    <div class="sidebar-dataset">
        <div class="sidebar-dataset-title">Dataset</div>
        <div class="sidebar-dataset-text">
            NASA PCoE Li-ion Battery Dataset<br>
            Batteries: B0005, B0006, B0007, B0018<br>
            636 discharge cycles · 25 features
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="biq-header">
    <div class="biq-header-inner">
        <div class="biq-eyebrow">Enterprise Battery Analytics</div>
        <h1>Battery State-of-Health Intelligence</h1>
        <p>Predict degradation · Estimate remaining useful life · Recommend maintenance actions for Indian operating conditions</p>
        <div class="biq-badge-row">
            <span class="biq-badge">⚡ NASA PCoE Dataset</span>
            <span class="biq-badge">🤖 Random Forest + XGBoost</span>
            <span class="biq-badge">🔍 SHAP Explainability</span>
            <span class="biq-badge">🇮🇳 India Climate Mode</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "⚡  Single Battery Analysis",
    "🗂  Fleet Batch Scoring",
    "🌡  India Climate Mode"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Battery Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Default insight state (before prediction) ──────────────────────────
    if 'soh_result' not in st.session_state:
        st.session_state.soh_result    = None
        st.session_state.rul_result    = None
        st.session_state.ir_result     = None
        st.session_state.energy_result = None
        st.session_state.shap_vals     = None
        st.session_state.df_scaled     = None

    # ── INSIGHT PANEL — always at top ─────────────────────────────────────
    if st.session_state.soh_result is not None:
        soh  = st.session_state.soh_result
        rul  = st.session_state.rul_result
        ir   = st.session_state.ir_result
        enrg = st.session_state.energy_result

        soh_cls = ('kpi-green'  if soh >= 90 else
                   'kpi-yellow' if soh >= 80 else
                   'kpi-orange' if soh >= 70 else 'kpi-red')
        rul_cls = ('kpi-green'  if rul >= 50 else
                   'kpi-yellow' if rul >= 20 else
                   'kpi-orange' if rul >= 5  else 'kpi-red')

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card {soh_cls}">
                <div class="kpi-accent-bar"></div>
                <div class="kpi-label">State of Health</div>
                <div class="kpi-value">{soh:.1f}%</div>
                <div class="kpi-sub">Industry EOL at 80%</div>
            </div>
            <div class="kpi-card {rul_cls}">
                <div class="kpi-accent-bar"></div>
                <div class="kpi-label">Remaining Useful Life</div>
                <div class="kpi-value">{rul}</div>
                <div class="kpi-sub">Discharge cycles left</div>
            </div>
            <div class="kpi-card kpi-blue">
                <div class="kpi-accent-bar"></div>
                <div class="kpi-label">Internal Resistance</div>
                <div class="kpi-value">{ir:.3f}</div>
                <div class="kpi-sub">Ohms — lower is healthier</div>
            </div>
            <div class="kpi-card kpi-blue">
                <div class="kpi-accent-bar"></div>
                <div class="kpi-label">Energy Delivered</div>
                <div class="kpi-value">{enrg:.2f}</div>
                <div class="kpi-sub">Wh this cycle</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # SoH progress bar
        fill_color = ('#16A34A' if soh >= 90 else
                      '#CA8A04' if soh >= 80 else
                      '#EA580C' if soh >= 70 else '#DC2626')
        st.markdown(f"""
        <div class="biq-progress-wrap">
            <div class="biq-progress-label">
                <span>Current SoH</span>
                <span>{soh:.1f}%</span>
            </div>
            <div class="biq-progress-track">
                <div class="biq-progress-fill" style="width:{soh:.1f}%;background:{fill_color};"></div>
                <div class="biq-eol-marker" title="80% EOL threshold"></div>
            </div>
            <div class="biq-eol-label">EOL threshold at 80%</div>
        </div>
        """, unsafe_allow_html=True)

        # Health alert
        if soh >= 90:
            st.markdown("""
            <div class="health-alert alert-green">
                <div class="health-alert-title">✓ &nbsp;HEALTHY — No Action Required</div>
                <div class="health-alert-body"><ul>
                    <li>Battery operating within optimal parameters</li>
                    <li>Continue normal operation and standard charging protocol</li>
                    <li>Schedule next routine inspection at standard maintenance interval</li>
                </ul></div>
            </div>""", unsafe_allow_html=True)
        elif soh >= 80:
            st.markdown("""
            <div class="health-alert alert-yellow">
                <div class="health-alert-title">⚠ &nbsp;MONITOR — Approaching End-of-Life</div>
                <div class="health-alert-body"><ul>
                    <li>Battery nearing the 80% SoH industry EOL threshold</li>
                    <li>Schedule detailed inspection within the next 30 days</li>
                    <li>Notify procurement team to begin replacement planning</li>
                    <li>Consider reducing peak load to slow degradation</li>
                </ul></div>
            </div>""", unsafe_allow_html=True)
        elif soh >= 70:
            st.markdown("""
            <div class="health-alert alert-orange">
                <div class="health-alert-title">⚡ &nbsp;ACTION REQUIRED — Below Optimal Health</div>
                <div class="health-alert-body"><ul>
                    <li>Battery has crossed the 80% EOL threshold — immediate inspection required</li>
                    <li>Plan replacement within the next 2 weeks</li>
                    <li>Avoid deployment in high-load or safety-critical applications</li>
                    <li>Increase monitoring frequency; log all anomalous cycles</li>
                </ul></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="health-alert alert-red">
                <div class="health-alert-title">✕ &nbsp;CRITICAL — Immediate Replacement Required</div>
                <div class="health-alert-body"><ul>
                    <li>Battery at critical degradation — risk of sudden failure</li>
                    <li>Remove from service immediately before next operation</li>
                    <li>Do not deploy in any application</li>
                    <li>Conduct root cause analysis and log failure event for quality records</li>
                </ul></div>
            </div>""", unsafe_allow_html=True)

        # SHAP chart
        st.markdown('<div class="biq-section-label">🔍 &nbsp;Degradation Driver Analysis</div>',
                    unsafe_allow_html=True)
        sv   = st.session_state.shap_vals
        dsc  = st.session_state.df_scaled
        top_idx   = np.argsort(np.abs(sv[0]))[-8:][::-1]
        top_feats = [feature_cols[i] for i in top_idx]
        top_shap  = sv[0][top_idx]

        feat_labels = {
            'avg_voltage'              : 'Avg Voltage',
            'min_voltage'              : 'Min Voltage',
            'max_voltage'              : 'Max Voltage',
            'voltage_drop'             : 'Voltage Drop',
            'avg_current'              : 'Avg Current',
            'avg_temp'                 : 'Avg Temperature',
            'max_temp'                 : 'Max Temperature',
            'temp_rise'                : 'Temp Rise',
            'discharge_time'           : 'Discharge Time',
            'rolling_avg_capacity'     : 'Rolling Avg Capacity',
            'rolling_std_capacity'     : 'Rolling Std Capacity',
            'rolling_avg_voltage'      : 'Rolling Avg Voltage',
            'rolling_avg_temp'         : 'Rolling Avg Temp',
            'rolling_discharge_time'   : 'Rolling Discharge Time',
            'capacity_fade_rate'       : 'Capacity Fade Rate',
            'voltage_fade_rate'        : 'Voltage Fade Rate',
            'temp_increase_rate'       : 'Temp Increase Rate',
            'internal_resistance'      : 'Internal Resistance',
            'energy_delivered'         : 'Energy Delivered',
            'cumulative_energy'        : 'Cumulative Energy',
            'discharge_efficiency'     : 'Discharge Efficiency',
            'thermal_stress'           : 'Thermal Stress',
            'life_percentage'          : 'Life Percentage',
            'india_thermal_factor'     : 'India Thermal Factor',
            'india_adjusted_resistance': 'India Adj. Resistance',
        }
        display_labels = [feat_labels.get(f, f) for f in top_feats]

        fig, ax = plt.subplots(figsize=(9, 3.8))
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        bar_colors = ['#DC2626' if v > 0 else '#16A34A' for v in top_shap]
        bars = ax.barh(range(len(display_labels)), top_shap[::-1],
                       color=bar_colors[::-1], height=0.52,
                       edgecolor='none')
        ax.set_yticks(range(len(display_labels)))
        ax.set_yticklabels(display_labels[::-1], fontsize=8.5,
                           fontfamily='sans-serif', color='#94A3B8')
        ax.axvline(x=0, color='#475569', linewidth=0.7, linestyle='--')
        ax.set_xlabel('SHAP value (impact on SoH)', fontsize=8, color='#64748B')
        ax.tick_params(axis='x', colors='#64748B', labelsize=8)
        ax.tick_params(axis='y', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title('Red = accelerates degradation   ·   Green = preserves health',
                     fontsize=8, color='#64748B', pad=8, loc='left')
        plt.tight_layout(pad=0.8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.divider()

    else:
        # Empty state — before prediction
        st.markdown("""
        <div class="biq-info">
            ℹ &nbsp;Configure battery parameters below and click <strong>Run Analysis</strong> to see health predictions, risk alerts, and degradation driver explanations.
        </div>
        """, unsafe_allow_html=True)

    # ── INPUT SECTION ─────────────────────────────────────────────────────
    st.markdown('<div class="biq-section-label">⚙ &nbsp;Battery Cycle Parameters</div>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        with st.expander("Electrical Signals — Voltage & Current", expanded=True):
            avg_voltage  = st.slider('Average Voltage (V)',  2.5, 4.5, 3.5, 0.01)
            min_voltage  = st.slider('Minimum Voltage (V)',  2.0, 3.5, 2.7, 0.01)
            max_voltage  = st.slider('Maximum Voltage (V)',  3.5, 4.5, 4.2, 0.01)
            avg_current  = st.slider('Average Current (A)',  0.5, 3.0, 1.8, 0.1)
            voltage_drop = max_voltage - min_voltage
            st.markdown(f'<div class="derived-pill">⚡ Voltage drop: <strong>{voltage_drop:.3f} V</strong></div>',
                        unsafe_allow_html=True)

        with st.expander("Thermal Parameters — Temperature"):
            avg_temp  = st.slider('Average Temperature (°C)', 20.0, 50.0, 32.0, 0.5)
            max_temp  = st.slider('Maximum Temperature (°C)', 25.0, 60.0, 39.0, 0.5)
            temp_rise = max_temp - avg_temp
            st.markdown(f'<div class="derived-pill">🌡 Temp rise: <strong>{temp_rise:.1f} °C</strong></div>',
                        unsafe_allow_html=True)

    with col_r:
        with st.expander("Discharge Parameters", expanded=True):
            discharge_time = st.slider('Discharge Time (seconds)', 1000, 5000, 3500, 10)
            cycle_index    = st.slider('Current Cycle Number', 0, 200, 50)

        with st.expander("Rolling History — Last 5 Cycles"):
            rolling_avg_cap   = st.slider('Rolling Avg Capacity (Ah)',       1.0, 2.5, 1.8, 0.01)
            rolling_std_cap   = st.slider('Rolling Std Capacity',            0.0, 0.1, 0.01, 0.001)
            rolling_avg_volt  = st.slider('Rolling Avg Voltage (V)',         3.0, 4.0, 3.5, 0.01)
            rolling_avg_temp  = st.slider('Rolling Avg Temperature (°C)',    20.0, 50.0, 32.0, 0.5)
            rolling_disc_time = st.slider('Rolling Avg Discharge Time (s)',  1000, 5000, 3400, 10)

    st.markdown('<br>', unsafe_allow_html=True)
    run = st.button('Run Battery Analysis', use_container_width=True)

    if run:
        capacity_fade_rate   = -0.003 * cycle_index
        voltage_fade_rate    = -0.0005 * cycle_index
        temp_increase_rate   = 0.005 * cycle_index
        internal_resistance  = voltage_drop / max(avg_current, 0.01)
        energy_delivered     = avg_voltage * avg_current * discharge_time / 3600
        cumulative_energy    = energy_delivered * cycle_index
        discharge_efficiency = discharge_time / 4000
        thermal_stress       = max_temp * temp_rise
        life_percentage      = cycle_index / 168 * 100
        india_thermal_factor = thermal_stress * 1.4
        india_adj_resistance = internal_resistance * 1.15

        row = {
            'avg_voltage'              : avg_voltage,
            'min_voltage'              : min_voltage,
            'max_voltage'              : max_voltage,
            'voltage_drop'             : voltage_drop,
            'avg_current'              : avg_current,
            'avg_temp'                 : avg_temp,
            'max_temp'                 : max_temp,
            'temp_rise'                : temp_rise,
            'discharge_time'           : discharge_time,
            'rolling_avg_capacity'     : rolling_avg_cap,
            'rolling_std_capacity'     : rolling_std_cap,
            'rolling_avg_voltage'      : rolling_avg_volt,
            'rolling_avg_temp'         : rolling_avg_temp,
            'rolling_discharge_time'   : rolling_disc_time,
            'capacity_fade_rate'       : capacity_fade_rate,
            'voltage_fade_rate'        : voltage_fade_rate,
            'temp_increase_rate'       : temp_increase_rate,
            'internal_resistance'      : internal_resistance,
            'energy_delivered'         : energy_delivered,
            'cumulative_energy'        : cumulative_energy,
            'discharge_efficiency'     : discharge_efficiency,
            'thermal_stress'           : thermal_stress,
            'life_percentage'          : life_percentage,
            'india_thermal_factor'     : india_thermal_factor,
            'india_adjusted_resistance': india_adj_resistance,
        }

        df_input  = pd.DataFrame([row])[feature_cols]
        df_scaled = scaler.transform(df_input)

        soh_pred = float(np.clip(soh_model.predict(df_scaled)[0], 0, 100))
        rul_pred = max(0, int(rul_model.predict(df_scaled)[0]))

        explainer = shap.TreeExplainer(soh_model)
        shap_vals = explainer.shap_values(df_scaled)

        st.session_state.soh_result    = soh_pred
        st.session_state.rul_result    = rul_pred
        st.session_state.ir_result     = internal_resistance
        st.session_state.energy_result = energy_delivered
        st.session_state.shap_vals     = shap_vals
        st.session_state.df_scaled     = df_scaled

        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Fleet Batch Scoring
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown('<div class="biq-section-label">🗂 &nbsp;Fleet Battery Analysis</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="biq-info">
        ℹ &nbsp;Upload a CSV file containing battery cycle data. Each row represents one discharge cycle.
        The platform will score each cycle for SoH and RUL, then generate a prioritised maintenance report
        with fleet-level health summary.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-zone">
        <div class="upload-zone-icon">📂</div>
        <div class="upload-zone-title">Drop your battery fleet CSV here</div>
        <div class="upload-zone-sub">Supported format: CSV · Each row = one discharge cycle · Max 200 MB</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader('Select file', type='csv',
                                label_visibility='collapsed')

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.markdown(f'**{len(batch_df):,} cycles loaded** — Preview (first 5 rows):')
        st.dataframe(batch_df.head(), use_container_width=True)
        st.markdown('<br>', unsafe_allow_html=True)

        if st.button('Run Fleet Analysis', use_container_width=True):
            try:
                for col in feature_cols:
                    if col not in batch_df.columns:
                        batch_df[col] = 0

                X_batch   = scaler.transform(batch_df[feature_cols])
                soh_preds = np.clip(soh_model.predict(X_batch), 0, 100)
                rul_preds = np.maximum(rul_model.predict(X_batch), 0).astype(int)

                batch_df['Predicted_SoH_%'] = soh_preds.round(1)
                batch_df['Predicted_RUL']   = rul_preds
                batch_df['Health_Status']   = pd.cut(
                    batch_df['Predicted_SoH_%'],
                    bins=[0, 70, 80, 90, 100],
                    labels=['CRITICAL', 'ACTION REQUIRED', 'MONITOR', 'HEALTHY']
                )
                batch_df['Maintenance_Priority'] = batch_df['Predicted_SoH_%'].rank(
                    ascending=True).astype(int)

                healthy  = int((batch_df['Predicted_SoH_%'] >= 90).sum())
                monitor  = int(((batch_df['Predicted_SoH_%'] >= 80) & (batch_df['Predicted_SoH_%'] < 90)).sum())
                action   = int(((batch_df['Predicted_SoH_%'] >= 70) & (batch_df['Predicted_SoH_%'] < 80)).sum())
                critical = int((batch_df['Predicted_SoH_%'] < 70).sum())
                avg_soh  = batch_df['Predicted_SoH_%'].mean()

                st.markdown('---')
                st.markdown('<div class="biq-section-label">Fleet Health Summary</div>',
                            unsafe_allow_html=True)

                st.markdown(f"""
                <div class="kpi-grid">
                    <div class="kpi-card kpi-green">
                        <div class="kpi-accent-bar"></div>
                        <div class="kpi-label">Healthy</div>
                        <div class="kpi-value">{healthy}</div>
                        <div class="kpi-sub">SoH above 90%</div>
                    </div>
                    <div class="kpi-card kpi-yellow">
                        <div class="kpi-accent-bar"></div>
                        <div class="kpi-label">Monitor</div>
                        <div class="kpi-value">{monitor}</div>
                        <div class="kpi-sub">SoH 80 – 90%</div>
                    </div>
                    <div class="kpi-card kpi-orange">
                        <div class="kpi-accent-bar"></div>
                        <div class="kpi-label">Action Required</div>
                        <div class="kpi-value">{action}</div>
                        <div class="kpi-sub">SoH 70 – 80%</div>
                    </div>
                    <div class="kpi-card kpi-red">
                        <div class="kpi-accent-bar"></div>
                        <div class="kpi-label">Critical</div>
                        <div class="kpi-value">{critical}</div>
                        <div class="kpi-sub">SoH below 70%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="biq-section-label" style="margin-top:1rem;">Detailed Results — Sorted by Maintenance Priority</div>',
                            unsafe_allow_html=True)
                st.dataframe(
                    batch_df[['Predicted_SoH_%', 'Predicted_RUL',
                              'Health_Status', 'Maintenance_Priority']]
                    .sort_values('Maintenance_Priority')
                    .head(30),
                    use_container_width=True
                )

                csv = batch_df.to_csv(index=False)
                st.download_button(
                    'Download Full Fleet Report (CSV)',
                    csv, 'battery_fleet_report.csv', 'text/csv',
                    use_container_width=True
                )

            except Exception as e:
                st.markdown(f"""
                <div class="health-alert alert-red">
                    <div class="health-alert-title">Analysis Failed</div>
                    <div class="health-alert-body">{str(e)}<br>
                    Ensure your CSV contains the required feature columns or compatible battery cycle data.</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — India Climate Mode
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.markdown("""
    <div class="india-header">
        <h3>India Climate Adjustment Engine</h3>
        <p>NASA battery data was collected at 24°C in controlled lab conditions. Indian batteries operate at
        35–45°C ambient temperature, reducing effective RUL by 15–30%. This module applies state-wise,
        seasonal, and application-specific correction factors for India-realistic lifetime estimates.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="biq-section-label">📍 &nbsp;Location & Season</div>',
                    unsafe_allow_html=True)
        state = st.selectbox('State / Region', [
            'Rajasthan', 'Gujarat', 'Maharashtra', 'Tamil Nadu',
            'Delhi', 'Karnataka', 'West Bengal', 'Bihar', 'Uttar Pradesh'])
        season = st.selectbox('Operating Season', [
            'Summer (April – June)',
            'Monsoon (July – September)',
            'Winter (October – March)'])
        use_case = st.selectbox('Battery Application', [
            'EV 2-Wheeler', 'EV 3-Wheeler / Auto',
            'Solar Storage', 'Telecom Tower', 'Industrial UPS'])

    with col_b:
        st.markdown('<div class="biq-section-label">🔋 &nbsp;Battery Parameters</div>',
                    unsafe_allow_html=True)
        base_soh     = st.slider('Current State of Health (%)',    50.0, 100.0, 85.0, 0.5)
        base_rul     = st.slider('Lab-Condition RUL (cycles)',      0, 150, 50)
        ambient_temp = st.slider('Ambient Temperature (°C)',       20.0, 48.0, 38.0, 0.5)

    st.markdown('<br>', unsafe_allow_html=True)

    if st.button('Calculate India-Adjusted RUL', use_container_width=True):

        temp_factor = 1 - ((ambient_temp - 24) * 0.008)
        season_map = {
            'Summer (April – June)'      : 0.82,
            'Monsoon (July – September)' : 0.91,
            'Winter (October – March)'   : 0.97
        }
        state_map = {
            'Rajasthan': 0.78, 'Gujarat': 0.82, 'Maharashtra': 0.88,
            'Tamil Nadu': 0.85, 'Delhi': 0.84, 'Karnataka': 0.90,
            'West Bengal': 0.89, 'Bihar': 0.86, 'Uttar Pradesh': 0.87
        }
        use_map = {
            'EV 2-Wheeler': 0.95, 'EV 3-Wheeler / Auto': 0.92,
            'Solar Storage': 0.88, 'Telecom Tower': 0.85,
            'Industrial UPS': 0.90
        }

        combined  = temp_factor * season_map[season] * state_map[state] * use_map[use_case]
        india_rul = max(0, int(base_rul * combined))
        reduction = base_rul - india_rul
        pct_loss  = (1 - combined) * 100

        st.markdown('---')
        st.markdown('<div class="biq-section-label">Adjusted Prediction Results</div>',
                    unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card kpi-blue">
                <div class="kpi-accent-bar"></div>
                <div class="kpi-label">Lab RUL (24°C)</div>
                <div class="kpi-value">{base_rul}</div>
                <div class="kpi-sub">NASA controlled conditions</div>
            </div>
            <div class="kpi-card kpi-orange">
                <div class="kpi-accent-bar"></div>
                <div class="kpi-label">India-Adjusted RUL</div>
                <div class="kpi-value">{india_rul}</div>
                <div class="kpi-sub">Field conditions estimate</div>
            </div>
            <div class="kpi-card kpi-red">
                <div class="kpi-accent-bar"></div>
                <div class="kpi-label">Cycles Lost</div>
                <div class="kpi-value">{reduction}</div>
                <div class="kpi-sub">Due to India climate</div>
            </div>
            <div class="kpi-card kpi-yellow">
                <div class="kpi-accent-bar"></div>
                <div class="kpi-label">Climate Factor</div>
                <div class="kpi-value">{combined:.2f}x</div>
                <div class="kpi-sub">RUL reduced by {pct_loss:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Breakdown table
        st.markdown('<div class="biq-section-label" style="margin-top:0.75rem;">Adjustment Factor Breakdown</div>',
                    unsafe_allow_html=True)

        breakdown_rows = [
            ('Temperature',      f'{temp_factor:.3f}x', f'{ambient_temp}°C ambient vs 24°C lab  (+{ambient_temp-24:.0f}°C)'),
            ('Season',           f'{season_map[season]:.3f}x', season),
            ('State / Region',   f'{state_map[state]:.3f}x', state),
            ('Application',      f'{use_map[use_case]:.3f}x', use_case),
            ('Combined',         f'{combined:.3f}x', f'RUL reduced by {pct_loss:.1f}% vs NASA lab'),
        ]

        rows_html = ''.join([
            f'<tr><td>{r[0]}</td><td><span class="factor-tag">{r[1]}</span></td><td>{r[2]}</td></tr>'
            for r in breakdown_rows
        ])

        st.markdown(f"""
        <table class="breakdown-table">
            <thead><tr><th>Factor</th><th>Multiplier</th><th>Description</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # Recommendations
        st.markdown('<div class="biq-section-label" style="margin-top:1.25rem;">Maintenance Recommendations</div>',
                    unsafe_allow_html=True)

        recs = []
        if ambient_temp > 35:
            recs.append(('🌡', 'Temperature Management',
                         f'Ambient {ambient_temp}°C exceeds the 35°C threshold. Install active cooling or ventilation. '
                         f'Each 10°C rise above 25°C approximately halves battery cycle life.'))
        if 'Summer' in season:
            recs.append(('☀️', 'Summer Charging Protocol',
                         'Avoid charging between 12:00 PM and 4:00 PM during peak heat. '
                         'Schedule charging for night or early morning to reduce thermal stress.'))
        if 'Monsoon' in season:
            recs.append(('🌧', 'Monsoon Sealing Inspection',
                         'Inspect IP rating, gaskets, and connectors monthly during July–September. '
                         'Humidity above 80% degrades electrolyte and increases corrosion risk.'))
        if state in ['Rajasthan', 'Gujarat']:
            recs.append(('🏜', 'High-Heat Region Protocol',
                         'Reduce maximum charge rate by 20% during peak summer months. '
                         'Consider LFP chemistry over NMC for superior thermal stability in this region.'))
        if use_case == 'Solar Storage':
            recs.append(('⚡', 'Irregular Charging Pattern',
                         'Solar charging creates variable profiles that accelerate degradation. '
                         'Deploy a BMS with adaptive charging algorithm and charge-current limiting.'))
        if use_case == 'Telecom Tower':
            recs.append(('📡', 'Telecom Replacement Scheduling',
                         'Plan replacement before monsoon season to avoid outage during network-critical periods. '
                         'Maintain 20% emergency buffer stock at depot.'))

        recs.append(('📅', 'Next Inspection Date',
                     f'Schedule detailed inspection at cycle {max(0, india_rul - 10)}, '
                     f'approximately {max(1, india_rul // 30)} month(s) from now.'))
        recs.append(('🔄', 'Procurement Lead Time',
                     f'Initiate replacement procurement now. '
                     f'Target delivery before cycle {india_rul}. Factor 2–4 weeks sourcing lead time.'))

        recs_html = ''.join([
            f'<div class="rec-item"><div class="rec-icon">{icon}</div>'
            f'<div><div class="rec-title">{title}</div>'
            f'<div class="rec-body">{body}</div></div></div>'
            for icon, title, body in recs
        ])
        st.markdown(recs_html, unsafe_allow_html=True)
