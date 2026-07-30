import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import shap
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Battery SoH Intelligence Platform',
    page_icon='🔋',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Professional CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main background */
    .main { background-color: #F8F9FB; }
    .block-container { padding: 2rem 2.5rem; }

    /* Top header bar */
    .app-header {
        background: linear-gradient(135deg, #1B2A47 0%, #243B6E 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.3px;
        color: white;
    }
    .app-header p {
        font-size: 0.85rem;
        color: #A8B8D8;
        margin: 0.3rem 0 0;
    }
    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        color: #A8B8D8;
        margin-top: 0.5rem;
        margin-right: 6px;
    }

    /* KPI Cards */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #E8ECF0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #6B7A8D;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1B2A47;
        line-height: 1;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #6B7A8D;
        margin-top: 0.3rem;
    }
    .kpi-green  { border-left: 3px solid #22C55E; }
    .kpi-yellow { border-left: 3px solid #F59E0B; }
    .kpi-orange { border-left: 3px solid #F97316; }
    .kpi-red    { border-left: 3px solid #EF4444; }
    .kpi-blue   { border-left: 3px solid #3B82F6; }

    /* Section headers */
    .section-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1B2A47;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E8ECF0;
    }

    /* Alert boxes */
    .alert-green {
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-left: 4px solid #22C55E;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    .alert-yellow {
        background: #FFFBEB;
        border: 1px solid #FCD34D;
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    .alert-orange {
        background: #FFF7ED;
        border: 1px solid #FDBA74;
        border-left: 4px solid #F97316;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    .alert-red {
        background: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-left: 4px solid #EF4444;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    .alert-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1B2A47;
        margin-bottom: 0.4rem;
    }
    .alert-body {
        font-size: 0.82rem;
        color: #374151;
        line-height: 1.6;
    }

    /* Info box */
    .info-box {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: #1D4ED8;
        margin-bottom: 1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #1B2A47;
    }
    [data-testid="stSidebar"] * {
        color: #A8B8D8 !important;
    }
    [data-testid="stSidebar"] .sidebar-title {
        color: white !important;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border-radius: 8px;
        padding: 4px;
        border: 1px solid #E8ECF0;
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        font-size: 0.83rem;
        font-weight: 500;
        padding: 0.5rem 1.25rem;
        color: #6B7A8D;
    }
    .stTabs [aria-selected="true"] {
        background: #1B2A47 !important;
        color: white !important;
    }

    /* Sliders */
    .stSlider label { font-size: 0.82rem; color: #374151; font-weight: 500; }

    /* Select boxes */
    .stSelectbox label { font-size: 0.82rem; color: #374151; font-weight: 500; }

    /* Button */
    .stButton > button {
        background: #1B2A47;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.65rem 1.5rem;
        letter-spacing: 0.02em;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #243B6E; }

    /* Divider */
    hr { border-color: #E8ECF0; margin: 1.25rem 0; }

    /* Dataframe */
    .dataframe { font-size: 0.82rem; }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #22C55E, #3B82F6);
        border-radius: 4px;
    }

    /* Metric override */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #E8ECF0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ────────────────────────────────────────────────────────────
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Battery SoH Platform</div>',
                unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('**About**')
    st.markdown('''
    This platform predicts battery State of Health (SoH)
    and Remaining Useful Life (RUL) using machine learning
    trained on NASA PCoE battery degradation data.
    ''')
    st.markdown('---')
    st.markdown('**Model Performance**')
    st.markdown('SoH Model R² : **0.9421**')
    st.markdown('RUL Model R² : **0.8594**')
    st.markdown('Training data : **504 cycles**')
    st.markdown('Test battery  : **B0018 (unseen)**')
    st.markdown('---')
    st.markdown('**Built for**')
    st.markdown('Replus Engitech · Exide · Amara Raja · JSW Neo Energy')
    st.markdown('---')
    st.markdown('**Dataset**')
    st.markdown('NASA PCoE Li-ion Battery Dataset')
    st.markdown('Batteries: B0005, B0006, B0007, B0018')

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>Battery State-of-Health Intelligence Platform</h1>
    <p>Predict degradation · Estimate remaining useful life · Recommend maintenance actions</p>
    <span class="header-badge">NASA PCoE Dataset</span>
    <span class="header-badge">XGBoost + Random Forest</span>
    <span class="header-badge">SHAP Explainability</span>
    <span class="header-badge">India Climate Mode</span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    'Single Battery Analysis',
    'Batch Fleet Scoring',
    'India Climate Mode'
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Battery Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Battery Cycle Parameters</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">Voltage Signals</div>',
                    unsafe_allow_html=True)
        avg_voltage  = st.slider('Average Voltage (V)', 2.5, 4.5, 3.5, 0.01)
        min_voltage  = st.slider('Minimum Voltage (V)', 2.0, 3.5, 2.7, 0.01)
        max_voltage  = st.slider('Maximum Voltage (V)', 3.5, 4.5, 4.2, 0.01)
        voltage_drop = max_voltage - min_voltage
        st.markdown(f'''
        <div class="kpi-card kpi-blue" style="margin-top:0.5rem;">
            <div class="kpi-label">Voltage Drop</div>
            <div class="kpi-value" style="font-size:1.3rem;">{voltage_drop:.3f} V</div>
        </div>''', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Thermal & Current</div>',
                    unsafe_allow_html=True)
        avg_current    = st.slider('Average Current (A)', 0.5, 3.0, 1.8, 0.1)
        avg_temp       = st.slider('Average Temperature (°C)', 20.0, 50.0, 32.0, 0.5)
        max_temp       = st.slider('Maximum Temperature (°C)', 25.0, 60.0, 39.0, 0.5)
        discharge_time = st.slider('Discharge Time (seconds)', 1000, 5000, 3500, 10)
        temp_rise      = max_temp - avg_temp
        st.markdown(f'''
        <div class="kpi-card kpi-orange" style="margin-top:0.5rem;">
            <div class="kpi-label">Temperature Rise</div>
            <div class="kpi-value" style="font-size:1.3rem;">{temp_rise:.1f} °C</div>
        </div>''', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-header">Cycle History</div>',
                    unsafe_allow_html=True)
        cycle_index       = st.slider('Current Cycle Number', 0, 200, 50)
        rolling_avg_cap   = st.slider('Rolling Avg Capacity (Ah)', 1.0, 2.5, 1.8, 0.01)
        rolling_std_cap   = st.slider('Rolling Std Capacity', 0.0, 0.1, 0.01, 0.001)
        rolling_avg_volt  = st.slider('Rolling Avg Voltage (V)', 3.0, 4.0, 3.5, 0.01)
        rolling_avg_temp  = st.slider('Rolling Avg Temperature (°C)', 20.0, 50.0, 32.0, 0.5)
        rolling_disc_time = st.slider('Rolling Avg Discharge Time (s)', 1000, 5000, 3400, 10)

    st.markdown('<br>', unsafe_allow_html=True)

    if st.button('Run Battery Analysis', use_container_width=True):

        # Feature engineering
        capacity_fade_rate   = -0.003 * cycle_index
        voltage_fade_rate    = -0.0005 * cycle_index
        temp_increase_rate   = 0.005 * cycle_index
        internal_resistance  = voltage_drop / avg_current
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

        df_input = pd.DataFrame([row])[feature_cols]
        df_scaled = scaler.transform(df_input)

        soh_pred = float(np.clip(soh_model.predict(df_scaled)[0], 0, 100))
        rul_pred = max(0, int(rul_model.predict(df_scaled)[0]))
        internal_res = voltage_drop / avg_current

        # ── KPI Row ──────────────────────────────────────────────────────────
        st.markdown('---')
        st.markdown('<div class="section-header">Analysis Results</div>',
                    unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)

        soh_color = ('kpi-green' if soh_pred >= 90 else
                     'kpi-yellow' if soh_pred >= 80 else
                     'kpi-orange' if soh_pred >= 70 else 'kpi-red')

        with k1:
            st.markdown(f'''
            <div class="kpi-card {soh_color}">
                <div class="kpi-label">State of Health</div>
                <div class="kpi-value">{soh_pred:.1f}%</div>
                <div class="kpi-sub">Current battery health</div>
            </div>''', unsafe_allow_html=True)
        with k2:
            st.markdown(f'''
            <div class="kpi-card kpi-blue">
                <div class="kpi-label">Remaining Useful Life</div>
                <div class="kpi-value">{rul_pred}</div>
                <div class="kpi-sub">Cycles remaining</div>
            </div>''', unsafe_allow_html=True)
        with k3:
            st.markdown(f'''
            <div class="kpi-card kpi-blue">
                <div class="kpi-label">Internal Resistance</div>
                <div class="kpi-value">{internal_res:.3f}</div>
                <div class="kpi-sub">Ohms (lower is better)</div>
            </div>''', unsafe_allow_html=True)
        with k4:
            st.markdown(f'''
            <div class="kpi-card kpi-blue">
                <div class="kpi-label">Energy Delivered</div>
                <div class="kpi-value">{energy_delivered:.2f}</div>
                <div class="kpi-sub">Wh this cycle</div>
            </div>''', unsafe_allow_html=True)

        # SoH progress bar
        st.markdown('<br>', unsafe_allow_html=True)
        st.progress(int(soh_pred))
        st.caption(f'State of Health: {soh_pred:.1f}% — Industry EOL threshold: 80%')

        # ── Alert ─────────────────────────────────────────────────────────────
        st.markdown('---')
        st.markdown('<div class="section-header">Health Status & Recommended Action</div>',
                    unsafe_allow_html=True)

        if soh_pred >= 90:
            st.markdown('''
            <div class="alert-green">
                <div class="alert-title">HEALTHY — No Action Required</div>
                <div class="alert-body">
                Battery is operating within optimal parameters. Continue normal operation.<br>
                Schedule next routine inspection at the standard maintenance interval.<br>
                Monitor rolling capacity trend at next 5 cycles.
                </div>
            </div>''', unsafe_allow_html=True)
        elif soh_pred >= 80:
            st.markdown('''
            <div class="alert-yellow">
                <div class="alert-title">MONITOR — Approaching End-of-Life Threshold</div>
                <div class="alert-body">
                Battery is nearing the 80% SoH industry EOL threshold.<br>
                Schedule a detailed inspection within the next 30 days.<br>
                Review charging patterns and consider reducing peak load.<br>
                Procurement team should be notified for replacement planning.
                </div>
            </div>''', unsafe_allow_html=True)
        elif soh_pred >= 70:
            st.markdown('''
            <div class="alert-orange">
                <div class="alert-title">ACTION REQUIRED — Below Optimal Health</div>
                <div class="alert-body">
                Battery has crossed the 80% EOL threshold — immediate inspection required.<br>
                Plan replacement within the next 2 weeks.<br>
                Avoid deployment in high-load or safety-critical applications.<br>
                Increase monitoring frequency to every cycle.
                </div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="alert-red">
                <div class="alert-title">CRITICAL — Immediate Replacement Required</div>
                <div class="alert-body">
                Battery is at critical degradation level — risk of sudden failure.<br>
                Remove from service immediately and replace before next operation.<br>
                Do not use in any application. Log failure event for quality records.<br>
                Conduct root cause analysis to prevent recurrence.
                </div>
            </div>''', unsafe_allow_html=True)

        # ── SHAP Chart ────────────────────────────────────────────────────────
        st.markdown('---')
        st.markdown('<div class="section-header">Prediction Explanation — Key Degradation Drivers</div>',
                    unsafe_allow_html=True)

        explainer  = shap.TreeExplainer(soh_model)
        shap_vals  = explainer.shap_values(df_scaled)
        top_idx    = np.argsort(np.abs(shap_vals[0]))[-8:][::-1]
        top_feats  = [feature_cols[i] for i in top_idx]
        top_shap   = shap_vals[0][top_idx]

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        colors = ['#EF4444' if v > 0 else '#22C55E' for v in top_shap]
        bars = ax.barh(range(len(top_feats)), top_shap[::-1],
                       color=colors[::-1], height=0.55, edgecolor='none')
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels(top_feats[::-1], fontsize=9,
                           fontfamily='DejaVu Sans', color='#374151')
        ax.axvline(x=0, color='#9CA3AF', linewidth=0.8)
        ax.set_xlabel('SHAP Value (impact on SoH prediction)',
                      fontsize=9, color='#6B7A8D')
        ax.set_title('Feature Impact on State of Health Prediction\n'
                     'Red = increases degradation risk  |  Green = improves health',
                     fontsize=10, color='#1B2A47', pad=12)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#E8ECF0')
        ax.tick_params(colors='#6B7A8D')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Fleet Scoring
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Battery Fleet Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('''
    <div class="info-box">
    Upload a CSV file containing battery cycle data. Each row represents one discharge cycle.
    The platform will predict SoH and RUL for each cycle and generate a prioritised maintenance report.
    </div>''', unsafe_allow_html=True)

    uploaded = st.file_uploader('Select CSV File', type='csv',
                                label_visibility='collapsed')

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.markdown(f'**{len(batch_df)} cycles loaded** — Preview:')
        st.dataframe(batch_df.head(), use_container_width=True)

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
                batch_df['Priority_Rank'] = batch_df['Predicted_SoH_%'].rank(
                    ascending=True).astype(int)

                # Summary KPIs
                st.markdown('---')
                st.markdown('<div class="section-header">Fleet Health Summary</div>',
                            unsafe_allow_html=True)

                s1, s2, s3, s4 = st.columns(4)
                healthy  = (batch_df['Predicted_SoH_%'] >= 90).sum()
                monitor  = ((batch_df['Predicted_SoH_%'] >= 80) &
                            (batch_df['Predicted_SoH_%'] < 90)).sum()
                action   = ((batch_df['Predicted_SoH_%'] >= 70) &
                            (batch_df['Predicted_SoH_%'] < 80)).sum()
                critical = (batch_df['Predicted_SoH_%'] < 70).sum()

                with s1:
                    st.markdown(f'''
                    <div class="kpi-card kpi-green">
                        <div class="kpi-label">Healthy</div>
                        <div class="kpi-value">{healthy}</div>
                        <div class="kpi-sub">SoH above 90%</div>
                    </div>''', unsafe_allow_html=True)
                with s2:
                    st.markdown(f'''
                    <div class="kpi-card kpi-yellow">
                        <div class="kpi-label">Monitor</div>
                        <div class="kpi-value">{monitor}</div>
                        <div class="kpi-sub">SoH 80–90%</div>
                    </div>''', unsafe_allow_html=True)
                with s3:
                    st.markdown(f'''
                    <div class="kpi-card kpi-orange">
                        <div class="kpi-label">Action Required</div>
                        <div class="kpi-value">{action}</div>
                        <div class="kpi-sub">SoH 70–80%</div>
                    </div>''', unsafe_allow_html=True)
                with s4:
                    st.markdown(f'''
                    <div class="kpi-card kpi-red">
                        <div class="kpi-label">Critical</div>
                        <div class="kpi-value">{critical}</div>
                        <div class="kpi-sub">SoH below 70%</div>
                    </div>''', unsafe_allow_html=True)

                st.markdown('<br>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">Detailed Results</div>',
                            unsafe_allow_html=True)
                st.dataframe(
                    batch_df[['Predicted_SoH_%', 'Predicted_RUL',
                              'Health_Status', 'Priority_Rank']].head(20),
                    use_container_width=True
                )

                csv = batch_df.to_csv(index=False)
                st.download_button(
                    'Download Fleet Report (CSV)', csv,
                    'battery_fleet_report.csv', 'text/csv',
                    use_container_width=True
                )

            except Exception as e:
                st.error(f'Analysis failed: {e}')

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — India Climate Mode
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">India Climate Adjustment</div>',
                unsafe_allow_html=True)
    st.markdown('''
    <div class="info-box">
    NASA battery data was collected under controlled lab conditions at 24°C.
    Indian batteries operate at 35–45°C ambient temperature, reducing effective RUL
    by 15–30% depending on state, season, and use case. This module applies
    climate-specific correction factors to provide India-realistic predictions.
    </div>''', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Location & Operating Conditions</div>',
                    unsafe_allow_html=True)
        state    = st.selectbox('State / Region', [
            'Rajasthan', 'Gujarat', 'Maharashtra', 'Tamil Nadu',
            'Delhi', 'Karnataka', 'West Bengal', 'Bihar', 'Uttar Pradesh'])
        season   = st.selectbox('Season', [
            'Summer (April — June)',
            'Monsoon (July — September)',
            'Winter (October — March)'])
        use_case = st.selectbox('Battery Application', [
            'EV 2-Wheeler', 'EV 3-Wheeler / Auto',
            'Solar Storage', 'Telecom Tower', 'Industrial UPS'])

    with c2:
        st.markdown('<div class="section-header">Battery Parameters</div>',
                    unsafe_allow_html=True)
        base_soh     = st.slider('Current State of Health (%)', 50.0, 100.0, 85.0, 0.5)
        base_rul     = st.slider('Lab-Condition RUL (cycles)', 0, 150, 50)
        ambient_temp = st.slider('Ambient Temperature (°C)', 20.0, 48.0, 38.0, 0.5)

    st.markdown('<br>', unsafe_allow_html=True)

    if st.button('Calculate India-Adjusted RUL', use_container_width=True):

        temp_factor = 1 - ((ambient_temp - 24) * 0.008)

        season_map = {
            'Summer (April — June)'      : 0.82,
            'Monsoon (July — September)' : 0.91,
            'Winter (October — March)'   : 0.97
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

        combined = temp_factor * season_map[season] * state_map[state] * use_map[use_case]
        india_rul = max(0, int(base_rul * combined))
        reduction = base_rul - india_rul

        # Results
        st.markdown('---')
        st.markdown('<div class="section-header">Adjusted Prediction Results</div>',
                    unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'''
            <div class="kpi-card kpi-blue">
                <div class="kpi-label">Lab-Condition RUL</div>
                <div class="kpi-value">{base_rul}</div>
                <div class="kpi-sub">Cycles (NASA 24°C)</div>
            </div>''', unsafe_allow_html=True)
        with m2:
            st.markdown(f'''
            <div class="kpi-card kpi-orange">
                <div class="kpi-label">India-Adjusted RUL</div>
                <div class="kpi-value">{india_rul}</div>
                <div class="kpi-sub">Cycles (field conditions)</div>
            </div>''', unsafe_allow_html=True)
        with m3:
            st.markdown(f'''
            <div class="kpi-card kpi-red">
                <div class="kpi-label">RUL Reduction</div>
                <div class="kpi-value">{reduction}</div>
                <div class="kpi-sub">Cycles lost to climate</div>
            </div>''', unsafe_allow_html=True)
        with m4:
            st.markdown(f'''
            <div class="kpi-card kpi-blue">
                <div class="kpi-label">Climate Factor</div>
                <div class="kpi-value">{combined:.2f}x</div>
                <div class="kpi-sub">Combined adjustment</div>
            </div>''', unsafe_allow_html=True)

        # Factor breakdown table
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Adjustment Factor Breakdown</div>',
                    unsafe_allow_html=True)

        breakdown = pd.DataFrame({
            'Factor'     : ['Temperature', 'Season', 'State / Region',
                            'Battery Application', 'Combined'],
            'Multiplier' : [f'{temp_factor:.3f}x', f'{season_map[season]:.3f}x',
                            f'{state_map[state]:.3f}x', f'{use_map[use_case]:.3f}x',
                            f'{combined:.3f}x'],
            'Description': [
                f'{ambient_temp}°C ambient vs 24°C lab (+{ambient_temp-24:.0f}°C)',
                season,
                state,
                use_case,
                f'RUL reduced by {100*(1-combined):.1f}% vs lab conditions'
            ]
        })
        st.dataframe(breakdown, hide_index=True, use_container_width=True)

        # Recommendations
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Maintenance Recommendations</div>',
                    unsafe_allow_html=True)

        recs = []
        if ambient_temp > 35:
            recs.append(('Temperature Management',
                         f'Ambient temperature of {ambient_temp}°C exceeds recommended threshold. '
                         f'Install active cooling or ventilation. Each 10°C above 25°C roughly '
                         f'halves battery lifetime.'))
        if 'Summer' in season:
            recs.append(('Summer Charging Protocol',
                         'Avoid charging between 12:00 PM and 4:00 PM during peak heat. '
                         'Prefer night or early morning charging to reduce thermal stress.'))
        if 'Monsoon' in season:
            recs.append(('Monsoon Sealing Check',
                         'Inspect IP rating and seals monthly during July–September. '
                         'Humidity above 80% degrades electrolyte and connector integrity.'))
        if state in ['Rajasthan', 'Gujarat']:
            recs.append(('High Heat Region Protocol',
                         'Reduce maximum charge rate by 20% during peak summer. '
                         'Consider LFP chemistry over NMC for better thermal stability.'))
        if use_case == 'Solar Storage':
            recs.append(('Irregular Charging Pattern',
                         'Solar charging creates variable charge profiles that accelerate '
                         'degradation. Use a battery management system with adaptive charging.'))
        if use_case == 'Telecom Tower':
            recs.append(('Telecom Replacement Schedule',
                         'Plan battery replacement before monsoon season to avoid failure '
                         'during network-critical periods. Maintain 20% buffer stock.'))

        recs.append(('Next Inspection',
                     f'Schedule detailed inspection at cycle {max(0, india_rul - 10)} '
                     f'— approximately {max(1, india_rul // 30)} months from now.'))
        recs.append(('Replacement Planning',
                     f'Initiate procurement process now for replacement at cycle {india_rul}. '
                     f'Factor 2–4 weeks lead time for battery sourcing.'))

        for title, body in recs:
            st.markdown(f'''
            <div class="alert-yellow">
                <div class="alert-title">{title}</div>
                <div class="alert-body">{body}</div>
            </div>''', unsafe_allow_html=True)