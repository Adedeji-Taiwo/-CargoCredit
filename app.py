"""
app.py
CargoCredit – XchangeBox Perishable Trade Finance Risk Engine
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CargoCredit · XchangeBox",
    page_icon="📦",
    layout="wide",
)

# ── Design tokens ─────────────────────────────────────────────────────────
# Page bg:    #f0f4f8  (light blue-grey)
# Card bg:    #ffffff  (white)
# Navy:       #0a2342  (XchangeBox dark)
# Green:      #15803d  (approve / low risk)
# Amber:      #b45309  (review / medium risk)
# Red:        #b91c1c  (decline / high risk)
# Muted text: #4b5563  (WCAG AA on white)
# Chart bg:   #f8fafc  (off-white so axes are visible)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#6b7280; }
[data-testid="stHeader"]           { background:transparent; }
section[data-testid="stSidebar"]   { display:none; }

/* ── Cursor pointer on ALL interactive elements ── */
input, select, textarea,
button, [role="button"],
[data-testid="stSelectbox"],
[data-testid="stMultiSelect"],
[data-testid="stSlider"],
[data-testid="stNumberInput"],
[data-testid="stToggle"],
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label,
.stSelectbox > div,
.stMultiSelect > div,
div[data-baseweb="select"],
div[data-baseweb="select"] *,
div[data-baseweb="input"],
div[data-baseweb="slider"] *,
div[data-baseweb="checkbox"] *,
div[data-baseweb="radio"] *,
div[data-baseweb="toggle"] *,
[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button {
    cursor: pointer !important;
}
input[type="number"] { cursor: text !important; }

/* ── Top bar ── */
.topbar {
    background:#0a2342;
    padding:14px 28px;
    border-radius:12px;
    margin-bottom:1.5rem;
    display:flex;
    align-items:center;
    justify-content:space-between;
}
.topbar-logo  { color:#ffffff; font-size:1.25rem; font-weight:700; }
.topbar-sub   { color:#93c5fd; font-size:0.8rem;  margin-top:3px; }
.topbar-badge {
    background:#166534; color:#dcfce7;
    font-size:0.75rem; font-weight:600;
    padding:5px 14px; border-radius:20px;
    border:1px solid #22c55e;
    margin-left:8px;
}
.topbar-time {
    background:#1e3a5f; color:#93c5fd;
    font-size:0.75rem; font-weight:500;
    padding:5px 14px; border-radius:20px;
    border:1px solid #2d5a8e;
    margin-left:8px;
}

/* ── KPI cards ── */
.kpi-card {
    background:#ffffff;
    border:1px solid #d1dae8;
    border-radius:12px;
    padding:1.1rem 1.3rem;
    text-align:center;
    box-shadow:0 1px 3px rgba(0,0,0,0.06);
}
.kpi-val { font-size:1.8rem; font-weight:700; color:#0a2342; line-height:1.1; }
.kpi-lbl { font-size:0.75rem; color:#4b5563; margin-top:4px; font-weight:500; }

/* ── Section headings ── */
.section-hd {
    font-size:0.95rem; font-weight:700; color:#0a2342;
    margin:1.6rem 0 0.7rem;
    border-left:4px solid #15803d;
    padding-left:10px;
}

/* ── Decision cards ── */
.decision-card { border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1rem; }
.dec-approve { background:#f0fdf4; border:2px solid #16a34a; }
.dec-review  { background:#fffbeb; border:2px solid #d97706; }
.dec-decline { background:#fff1f2; border:2px solid #dc2626; }

.dec-title { font-size:1.15rem; font-weight:800; margin-bottom:5px; }
.dec-approve .dec-title { color:#15803d; }
.dec-review  .dec-title { color:#b45309; }
.dec-decline .dec-title { color:#b91c1c; }
.dec-body { font-size:0.85rem; color:#1f2937; line-height:1.65; }

/* ── Alert rows ── */
.alert-row {
    background:#fff1f2;
    border-left:4px solid #dc2626;
    border-radius:8px;
    padding:10px 14px;
    margin-bottom:8px;
    font-size:0.83rem;
    color:#1f2937;
}
.alert-row strong { color:#b91c1c; }

/* ── Footer ── */
.footer { color:#4b5563; font-size:0.75rem; text-align:center; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

# ── Chart colour palette ──────────────────────────────────────────────────
C_NAVY      = '#0a2342'
C_GREEN     = '#15803d'
C_AMBER     = '#d97706'
C_RED       = '#dc2626'
C_CHART_BG  = '#f8fafc'
C_GRID      = '#e2e8f0'
C_AXIS_TEXT = '#374151'
C_TEXT_DARK = '#1f2937'

# ── Load artefacts ────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    spoil = joblib.load('spoilage_model.pkl')
    deflt = joblib.load('default_model.pkl')
    prep  = joblib.load('preprocessor.pkl')
    feats = joblib.load('feature_names.pkl')
    with open('model_meta.json') as f:
        meta = json.load(f)
    return spoil, deflt, prep, feats, meta

spoil_model, def_model, preprocessor, feature_names, meta = load_models()

@st.cache_data
def load_portfolio():
    df = pd.read_csv('cargo_data.csv')
    df['risk_tier'] = pd.cut(
        df['default_prob'],
        bins=[0, 0.30, 0.60, 1.0],
        labels=['Low', 'Medium', 'High'],
    )
    return df

df_port = load_portfolio()

# ── Static data ───────────────────────────────────────────────────────────
CITIES = {
    'Kano':          (12.00,  8.52),
    'Lagos':         ( 6.52,  3.38),
    'Ibadan':        ( 7.38,  3.93),
    'Abuja':         ( 9.07,  7.40),
    'Port Harcourt': ( 4.82,  7.04),
    'Kaduna':        (10.52,  7.44),
    'Onitsha':       ( 6.14,  6.79),
    'Zaria':         (11.08,  7.71),
    'Jos':           ( 9.92,  8.89),
    'Enugu':         ( 6.44,  7.50),
}

PRODUCTS = {
    'Tomatoes':     {'max_temp': 25, 'shelf_hrs': 72},
    'Leafy Greens': {'max_temp': 10, 'shelf_hrs': 48},
    'Strawberries': {'max_temp':  8, 'shelf_hrs': 36},
    'Fresh Fish':   {'max_temp':  4, 'shelf_hrs': 24},
    'Dairy':        {'max_temp':  6, 'shelf_hrs': 48},
    'Mangoes':      {'max_temp': 20, 'shelf_hrs': 96},
    'Peppers':      {'max_temp': 22, 'shelf_hrs': 96},
}

# ── Chart layout helper ───────────────────────────────────────────────────
def base_layout(height=300, **kwargs):
    layout = dict(
        height=height,
        paper_bgcolor='#ffffff',
        plot_bgcolor=C_CHART_BG,
        font=dict(color=C_TEXT_DARK, size=12),
        margin=dict(t=30, b=40, l=10, r=20),
        xaxis=dict(
            gridcolor=C_GRID, linecolor=C_GRID,
            tickcolor=C_AXIS_TEXT,
            tickfont=dict(color=C_AXIS_TEXT, size=11),
            title_font=dict(color=C_AXIS_TEXT),
        ),
        yaxis=dict(
            gridcolor=C_GRID, linecolor=C_GRID,
            tickcolor=C_AXIS_TEXT,
            tickfont=dict(color=C_AXIS_TEXT, size=11),
            title_font=dict(color=C_AXIS_TEXT),
        ),
    )
    layout.update(kwargs)
    return layout

# ── Prediction helper ─────────────────────────────────────────────────────
def predict_risk(row_dict: dict) -> dict:
    df_in      = pd.DataFrame([row_dict])
    p          = preprocessor.transform(df_in)
    spoil_prob = float(spoil_model.predict_proba(p)[0][1])
    def_prob   = float(def_model.predict_proba(p)[0][1])
    rate       = round(2.0 + def_prob * 8.0, 2)

    if def_prob < 0.30:
        decision, dec_class = 'APPROVE', 'dec-approve'
        advice = (f"Risk profile is acceptable. Offer financing at "
                  f"<strong>{rate:.1f}%</strong> invoice discount. "
                  f"Standard monitoring applies.")
    elif def_prob < 0.60:
        decision, dec_class = 'REVIEW', 'dec-review'
        advice = (f"Elevated risk detected. Consider short-tenor financing "
                  f"only at <strong>{rate:.1f}%</strong>. Request additional "
                  f"collateral or guarantor before proceeding.")
    else:
        decision, dec_class = 'DECLINE', 'dec-decline'
        advice = (f"Risk exceeds lending threshold. Decline or escalate to "
                  f"senior credit officer. Spoilage probability: "
                  f"<strong>{spoil_prob:.0%}</strong>.")

    tier = ('Low' if def_prob < 0.30 else
            'Medium' if def_prob < 0.60 else 'High')

    return dict(spoil_prob=round(spoil_prob, 4),
                def_prob=round(def_prob, 4),
                rate=rate, decision=decision,
                dec_class=dec_class, advice=advice, tier=tier)


# ─────────────────────────────────────────────────────────────────────────
# TOP BAR  — f-string so datetime renders correctly
# ─────────────────────────────────────────────────────────────────────────
now_str = datetime.now().strftime('%d %b %Y  %H:%M')

st.markdown(f"""
<div class="topbar">
  <div>
    <div class="topbar-logo">📦 CargoCredit &nbsp;·&nbsp; XchangeBox</div>
    <div class="topbar-sub">Perishable Trade Finance Risk Engine · Nigeria</div>
  </div>
  <div style="display:flex;align-items:center">
    <span class="topbar-badge">🟢 LIVE</span>
    <span class="topbar-time">🕐 {now_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

page = st.radio(
    '', ['Credit Assessor', 'Portfolio Monitor', 'Model Intelligence'],
    horizontal=True, label_visibility='collapsed',
)
st.markdown('---')


# ═════════════════════════════════════════════════════════════════════════
# PAGE 1 — CREDIT ASSESSOR
# ═════════════════════════════════════════════════════════════════════════
if page == 'Credit Assessor':

    st.markdown('<p class="section-hd">New Financing Application</p>',
                unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap='large')

    with left:
        st.markdown("**Route & Cargo**")
        c1, c2 = st.columns(2)
        with c1:
            origin = st.selectbox('Origin', list(CITIES.keys()), index=0)
        with c2:
            dest_opts   = [c for c in CITIES if c != origin]
            destination = st.selectbox('Destination', dest_opts, index=3)

        product   = st.selectbox('Product', list(PRODUCTS.keys()))
        prod_info = PRODUCTS[product]

        c3, c4 = st.columns(2)
        with c3:
            cargo_kg    = st.number_input('Cargo weight (kg)',   200, 5000, 1000, 100)
            invoice_ngn = st.number_input('Invoice value (₦)', 50_000, 5_000_000,
                                          500_000, 50_000)
        with c4:
            distance_km = st.number_input('Distance (km)', 50, 800, 300, 50)
            delay_hrs   = st.slider('Expected delay (hrs)', 0.0, 12.0, 1.5, 0.5)

        st.markdown("**Transit Conditions**")
        c5, c6 = st.columns(2)
        with c5:
            has_cold_chain = st.toggle('Cold chain available', value=True)
            default_temp   = float(prod_info['max_temp'] + (0 if has_cold_chain else 12))
            avg_temp       = st.slider('Avg temperature (°C)', 2.0, 40.0,
                                       default_temp, 0.5)
        with c6:
            road_quality = st.selectbox('Road quality',
                                        ['good', 'fair', 'poor'], index=1)
            avg_humidity = st.slider('Avg humidity (%)', 40, 95, 72, 1)

        st.markdown("**Borrower Profile**")
        c7, c8 = st.columns(2)
        with c7:
            borrower_history = st.selectbox(
                'Payment history',
                ['excellent', 'good', 'fair', 'poor'], index=1,
            )
        with c8:
            prior_defaults = st.number_input('Prior defaults', 0, 10, 0, 1)

        assess = st.button('▶  Assess Credit Risk',
                           use_container_width=True, type='primary')

    with right:
        if assess:
            transit_hrs = distance_km / 50.0
            total_hrs   = transit_hrs + delay_hrs
            max_temp    = avg_temp + np.random.uniform(2, 6)
            temp_exc    = max(0.0, max_temp - prod_info['max_temp'])
            time_ratio  = total_hrs / prod_info['shelf_hrs']
            road_num    = {'good': 0, 'fair': 1, 'poor': 2}[road_quality]
            vib_score   = float(road_num)

            row = dict(
                distance_km=distance_km,  transit_hrs=transit_hrs,
                delay_hrs=delay_hrs,       time_ratio=time_ratio,
                has_cold_chain=int(has_cold_chain),
                avg_temp=avg_temp,         max_temp=max_temp,
                temp_exceedance=temp_exc,  avg_humidity=avg_humidity,
                vibration_score=vib_score, cargo_kg=cargo_kg,
                prior_defaults=prior_defaults,
                product=product, road_quality=road_quality,
                borrower_history=borrower_history,
            )
            res = predict_risk(row)

            # Decision card
            st.markdown(
                f'<div class="decision-card {res["dec_class"]}">'
                f'<div class="dec-title">{res["decision"]}</div>'
                f'<div class="dec-body">{res["advice"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Risk metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-val">{res['spoil_prob']:.0%}</div>
                    <div class="kpi-lbl">Spoilage probability</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-val">{res['def_prob']:.0%}</div>
                    <div class="kpi-lbl">Default probability</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-val">{res['rate']:.1f}%</div>
                    <div class="kpi-lbl">Suggested financing rate</div>
                </div>""", unsafe_allow_html=True)

            # Financial summary
            st.markdown('<p class="section-hd">Financial Summary</p>',
                        unsafe_allow_html=True)
            fee           = invoice_ngn * res['rate'] / 100
            expected_loss = invoice_ngn * res['def_prob']

            f1, f2, f3 = st.columns(3)
            f1.metric('Invoice value',  f'₦{invoice_ngn:,.0f}')
            f2.metric('Financing fee',  f'₦{fee:,.0f}',
                      delta=f"{res['rate']:.1f}%", delta_color='off')
            f3.metric('Expected loss',  f'₦{expected_loss:,.0f}',
                      delta=f"{res['def_prob']:.1%} risk",
                      delta_color='inverse')

            # Risk gauge
            bar_color = (C_GREEN if res['def_prob'] < 0.30 else
                         C_AMBER if res['def_prob'] < 0.60 else C_RED)
            fig_g = go.Figure(go.Indicator(
                mode='gauge+number',
                value=res['def_prob'] * 100,
                number={'suffix': '%',
                        'font': {'size': 32, 'color': C_TEXT_DARK}},
                gauge={
                    'axis': {
                        'range': [0, 100],
                        'tickcolor': C_AXIS_TEXT,
                        'tickfont':  {'color': C_AXIS_TEXT, 'size': 11},
                    },
                    'bar':  {'color': bar_color, 'thickness': 0.25},
                    'bgcolor': C_CHART_BG,
                    'bordercolor': C_GRID,
                    'steps': [
                        {'range': [0,  30], 'color': '#dcfce7'},
                        {'range': [30, 60], 'color': '#fef9c3'},
                        {'range': [60,100], 'color': '#fee2e2'},
                    ],
                    'threshold': {
                        'line':      {'color': C_TEXT_DARK, 'width': 2},
                        'thickness': 0.75,
                        'value':     res['def_prob'] * 100,
                    },
                },
                title={'text': 'Default Risk Score',
                       'font': {'color': C_TEXT_DARK, 'size': 13}},
            ))
            fig_g.update_layout(
                height=250,
                paper_bgcolor='#ffffff',
                margin=dict(t=40, b=10, l=30, r=30),
            )
            st.plotly_chart(fig_g, use_container_width=True)

        else:
            st.info('Fill in the application on the left and click '
                    '**▶ Assess Credit Risk**.')


# ═════════════════════════════════════════════════════════════════════════
# PAGE 2 — PORTFOLIO MONITOR
# ═════════════════════════════════════════════════════════════════════════
elif page == 'Portfolio Monitor':

    stats = meta['data_stats']
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (f"{stats['n_shipments']:,}",         "Total shipments"),
        (f"₦{stats['avg_invoice']/1e6:.1f}M", "Avg invoice value"),
        (f"{stats['spoilage_rate']:.1%}",      "Historical spoilage rate"),
        (f"{stats['default_rate']:.1%}",       "Historical default rate"),
        (f"{stats['avg_rate']:.1f}%",          "Avg financing rate"),
    ]
    for col, (val, lbl) in zip([k1, k2, k3, k4, k5], kpis):
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-val">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-hd">Active Shipment Map — Nigeria</p>',
                unsafe_allow_html=True)

    df_map = df_port.sample(300, random_state=1).copy()
    df_map['lat'] = (df_map['origin'].map(lambda c: CITIES.get(c, (9, 8))[0])
                     + np.random.uniform(-0.25, 0.25, len(df_map)))
    df_map['lon'] = (df_map['origin'].map(lambda c: CITIES.get(c, (9, 8))[1])
                     + np.random.uniform(-0.25, 0.25, len(df_map)))

    tier_colors = {'Low': C_GREEN, 'Medium': C_AMBER, 'High': C_RED}
    fig_map = go.Figure()
    for tier, color in tier_colors.items():
        sub = df_map[df_map['risk_tier'] == tier]
        fig_map.add_trace(go.Scattermapbox(
            lat=sub['lat'], lon=sub['lon'],
            mode='markers',
            marker=dict(size=10, color=color, opacity=0.85),
            name=f'{tier} risk ({len(sub)})',
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Product: %{customdata[1]}<br>'
                'Route: %{customdata[2]} → %{customdata[3]}<br>'
                'Default prob: %{customdata[4]:.1%}<br>'
                'Rate: %{customdata[5]:.1f}%<extra></extra>'
            ),
            customdata=sub[['shipment_id', 'product', 'origin',
                             'destination', 'default_prob',
                             'financing_rate']].values,
        ))
    fig_map.update_layout(
        mapbox=dict(style='carto-positron',
                    center=dict(lat=9.0, lon=7.5), zoom=5),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    font=dict(color=C_TEXT_DARK, size=11),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor=C_GRID, borderwidth=1),
        margin=dict(t=10, b=0, l=0, r=0),
        height=440,
        paper_bgcolor='#ffffff',
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Three charts ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<p class="section-hd">Risk distribution</p>',
                    unsafe_allow_html=True)
        tc = df_port['risk_tier'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=tc.index.tolist(),
            values=tc.values.tolist(),
            marker_colors=[C_GREEN, C_AMBER, C_RED],
            hole=0.55,
            textinfo='label+percent',
            textfont=dict(size=12, color=C_TEXT_DARK),
            outsidetextfont=dict(color=C_TEXT_DARK),
        ))
        fig_pie.update_layout(
            showlegend=False, height=260,
            paper_bgcolor='#ffffff',
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown('<p class="section-hd">Spoilage rate by product</p>',
                    unsafe_allow_html=True)
        sp = (df_port.groupby('product')['spoiled']
              .mean().sort_values().reset_index())
        bar_colors = [C_RED if v > 0.5 else
                      C_AMBER if v > 0.3 else C_GREEN
                      for v in sp['spoiled']]
        fig_sp = go.Figure(go.Bar(
            x=sp['spoiled'], y=sp['product'],
            orientation='h',
            marker_color=bar_colors,
            text=[f'{v:.0%}' for v in sp['spoiled']],
            textposition='outside',
            textfont=dict(color=C_TEXT_DARK, size=11),
        ))
        fig_sp.update_layout(**base_layout(
            height=280,
            xaxis=dict(tickformat='.0%', range=[0, 0.85],
                       gridcolor=C_GRID, tickfont=dict(color=C_AXIS_TEXT),
                       title_font=dict(color=C_AXIS_TEXT)),
            yaxis=dict(gridcolor=C_GRID, tickfont=dict(color=C_AXIS_TEXT)),
        ))
        st.plotly_chart(fig_sp, use_container_width=True)

    with c3:
        st.markdown('<p class="section-hd">Default rate by borrower history</p>',
                    unsafe_allow_html=True)
        hist_order = ['excellent', 'good', 'fair', 'poor']
        dh = (df_port.groupby('borrower_history')['defaulted']
              .mean().reindex(hist_order).reset_index())
        fig_dh = go.Figure(go.Bar(
            x=dh['borrower_history'],
            y=dh['defaulted'],
            marker_color=[C_GREEN, '#22c55e', C_AMBER, C_RED],
            text=[f'{v:.0%}' for v in dh['defaulted']],
            textposition='outside',
            textfont=dict(color=C_TEXT_DARK, size=11),
        ))
        fig_dh.update_layout(**base_layout(
            height=280,
            yaxis=dict(tickformat='.0%', range=[0, 0.55],
                       gridcolor=C_GRID, tickfont=dict(color=C_AXIS_TEXT)),
            xaxis=dict(gridcolor=C_GRID, tickfont=dict(color=C_AXIS_TEXT)),
        ))
        st.plotly_chart(fig_dh, use_container_width=True)

    # ── Alerts ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-hd">🚨 High-Risk Shipments Requiring Action</p>',
                unsafe_allow_html=True)
    high_risk = (df_port[df_port['risk_tier'] == 'High']
                 .sort_values('default_prob', ascending=False).head(6))
    for _, row in high_risk.iterrows():
        st.markdown(
            f'<div class="alert-row">'
            f'<strong>{row["shipment_id"]}</strong> &nbsp;·&nbsp; '
            f'{row["product"]} &nbsp;·&nbsp; '
            f'{row["origin"]} → {row["destination"]} &nbsp;·&nbsp; '
            f'Default risk: <strong>{row["default_prob"]:.0%}</strong> &nbsp;·&nbsp; '
            f'Spoilage: {row["spoilage_prob"]:.0%} &nbsp;·&nbsp; '
            f'Rate: {row["financing_rate"]:.1f}% &nbsp;·&nbsp; '
            f'Invoice: ₦{row["invoice_ngn"]:,.0f}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════════
elif page == 'Model Intelligence':

    st.markdown('<p class="section-hd">Model Performance</p>',
                unsafe_allow_html=True)
    tab1, tab2 = st.tabs(['Spoilage Model', 'Default Model'])

    for tab, model_key, label in [
        (tab1, 'spoilage', 'Spoilage'),
        (tab2, 'default',  'Default'),
    ]:
        with tab:
            mm = meta[model_key]
            comp_rows = [{
                'Model':    ('★ ' if n == mm['best'] else '  ') + n,
                'AUC':      s['AUC'],
                'Recall':   s['Recall'],
                'Accuracy': s['Accuracy'],
            } for n, s in mm['comparison'].items()]

            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown(f"**{label} model comparison**")
                st.dataframe(pd.DataFrame(comp_rows),
                             hide_index=True, use_container_width=True)
                st.caption(f"Best: **{mm['best']}** · selected by AUC")

            with col_b:
                imp = mm.get('importances', {})
                if imp:
                    imp_df = (pd.DataFrame(list(imp.items()),
                                           columns=['Feature', 'Importance'])
                              .sort_values('Importance', ascending=True))
                    fig_imp = go.Figure(go.Bar(
                        x=imp_df['Importance'],
                        y=imp_df['Feature'],
                        orientation='h',
                        marker_color=C_NAVY,
                        text=[f'{v:.3f}' for v in imp_df['Importance']],
                        textposition='outside',
                        textfont=dict(color=C_TEXT_DARK, size=11),
                    ))
                    fig_imp.update_layout(**base_layout(
                        height=340,
                        margin=dict(t=30, b=20, l=10, r=60),
                        xaxis=dict(
                            title='Importance',
                            range=[0, imp_df['Importance'].max() * 1.25],
                            gridcolor=C_GRID,
                            tickfont=dict(color=C_AXIS_TEXT),
                            title_font=dict(color=C_AXIS_TEXT),
                        ),
                        yaxis=dict(
                            gridcolor=C_GRID,
                            tickfont=dict(color=C_AXIS_TEXT, size=11),
                        ),
                        title=dict(
                            text=f'Top features — {label} model',
                            font=dict(size=12, color=C_TEXT_DARK),
                        ),
                    ))
                    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('<p class="section-hd">Financing Rate Formula</p>',
                unsafe_allow_html=True)
    st.markdown("""
```
financing_rate (%) = 2.0 + default_probability × 8.0
```

| Default probability | Rate | Decision |
|---|---|---|
| 0 – 30% | 2.0 – 4.4% | ✅ Approve |
| 30 – 60% | 4.4 – 6.8% | ⚠ Review |
| 60 – 100% | 6.8 – 10.0% | ❌ Decline |

Base rate of **2%** reflects the Nigerian interbank lending floor.
Risk premium up to **8%** reflects perishability and borrower credit history.
    """)

    st.markdown('<p class="section-hd">Methodology</p>',
                unsafe_allow_html=True)
    st.markdown("""
**Two-model architecture**

CargoCredit trains two independent classifiers on the same feature set:

1. **Spoilage model** — predicts whether cargo will spoil in transit based on cold chain
   availability, temperature exceedance, time-to-shelf ratio, road quality, and humidity.
   Captures the *physical* risk of the asset.

2. **Default model** — predicts whether the borrower will default on the invoice.
   Incorporates spoilage risk plus borrower payment history and prior defaults.
   Captures the *financial* risk of the loan.

**Why separate models?** A borrower can be creditworthy but ship cargo that spoils
(operational risk). Cargo can arrive intact but the borrower still defaults (credit risk).
Separating them makes the system more transparent and gives loan officers two distinct
levers to act on.
    """)

    st.markdown(
        '<p class="footer">CargoCredit · XchangeBox Technologies · </p>',
        unsafe_allow_html=True,
    )