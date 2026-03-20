# frontend/ui_components.py
import streamlit as st
import plotly.graph_objects as go


def inject_styles():
    """Inject global CSS styles."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

        html, body, [class*="css"] {
            background-color: #0a0a0a;
            color: #e8e8e8;
            font-family: 'DM Sans', sans-serif;
        }
        .stApp { background: #0a0a0a; }
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding: 2rem 3rem; max-width: 1100px; }

        /* Hero */
        .hero {
            text-align: center;
            padding: 3rem 0 2rem;
            border-bottom: 1px solid #1f1f1f;
            margin-bottom: 2.5rem;
        }
        .hero-title {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 4.5rem;
            letter-spacing: 0.08em;
            color: #ffffff;
            line-height: 1;
            margin: 0;
        }
        .hero-accent { color: #00e676; }
        .hero-sub {
            font-size: 0.9rem;
            color: #555;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-top: 0.75rem;
            font-weight: 300;
        }
        .hero-badge {
            display: inline-block;
            background: #111;
            border: 1px solid #222;
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.75rem;
            color: #00e676;
            letter-spacing: 0.1em;
            margin-top: 1rem;
        }

        /* Buttons */
        .stButton > button {
            background: #00e676 !important;
            color: #000 !important;
            border: none !important;
            border-radius: 8px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            letter-spacing: 0.08em !important;
            padding: 0.65rem 2rem !important;
            text-transform: uppercase !important;
            transition: all 0.2s !important;
        }
        .stButton > button:hover {
            background: #00ff88 !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 20px rgba(0,230,118,0.3) !important;
        }

        /* Inputs */
        .stSelectbox > div > div {
            background: #111 !important;
            border: 1px solid #222 !important;
            border-radius: 8px !important;
            color: #e8e8e8 !important;
        }
        [data-testid="stFileUploader"] {
            background: #0f0f0f;
            border: 1px dashed #2a2a2a;
            border-radius: 12px;
            padding: 1.5rem;
        }

        /* Result card */
        .result-card {
            background: #0f0f0f;
            border: 1px solid #1a1a1a;
            border-radius: 16px;
            padding: 2rem;
            margin: 1.5rem 0;
            text-align: center;
        }
        .result-label {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 3.5rem;
            letter-spacing: 0.06em;
            line-height: 1;
            margin-bottom: 0.5rem;
        }
        .result-conf {
            font-size: 1rem;
            color: #555;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            font-weight: 300;
        }

        /* Section labels */
        .section-label {
            font-size: 0.7rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: #444;
            margin-bottom: 1rem;
            font-weight: 400;
        }

        /* Placeholder */
        .placeholder {
            border: 1px dashed #1f1f1f;
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            color: #2a2a2a;
        }

        /* Video */
        video { border-radius: 10px; border: 1px solid #1a1a1a; }
        hr { border-color: #1a1a1a !important; }
        .stSpinner > div { border-top-color: #00e676 !important; }
    </style>
    """, unsafe_allow_html=True)


def render_hero():
    """Render the top hero section."""
    st.markdown("""
    <div class="hero">
        <p class="hero-sub">AI · Computer Vision · Sports Analytics</p>
        <h1 class="hero-title">Football <span class="hero-accent">Event</span> Detection</h1>
        <div class="hero-badge">VideoMAE · 14 Classes · 77.6% Accuracy</div>
    </div>
    """, unsafe_allow_html=True)


def render_placeholder():
    """Render empty state on result panel."""
    st.markdown("""
    <div class="placeholder">
        <div style="font-size:2.5rem; margin-bottom:0.75rem">⚽</div>
        <div style="font-family:'Bebas Neue',sans-serif; font-size:1.4rem; letter-spacing:0.1em">
            Awaiting clip
        </div>
        <div style="font-size:0.75rem; margin-top:0.5rem; letter-spacing:0.1em; text-transform:uppercase">
            Upload or select a sample to begin
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_result_card(predictions: list[dict]):
    """Render top prediction card + bar chart."""
    top  = predictions[0]
    conf = top["confidence"]
    color = _conf_color(conf)

    st.markdown(f"""
    <div class="result-card">
        <div class="section-label">Detected Event</div>
        <div class="result-label" style="color:{color}">{top["class"]}</div>
        <div class="result-conf">
            Confidence &nbsp;·&nbsp;
            <span style="color:{color}; font-weight:500">{conf:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Top 3 predictions</div>', unsafe_allow_html=True)
    _render_bar_chart(predictions)


def render_footer():
    """Render bottom footer."""
    st.markdown("""
    <div style="text-align:center; color:#2a2a2a; font-size:0.7rem;
                letter-spacing:0.15em; text-transform:uppercase;
                border-top:1px solid #111; padding-top:1.5rem; margin-top:3rem">
        VideoMAE · Fine-tuned on SoccerNet · 26k clips · 14 event classes
    </div>
    """, unsafe_allow_html=True)


def render_error(message: str):
    """Render a styled error message."""
    st.markdown(f"""
    <div style="background:#1a0a0a; border:1px solid #ff525233;
                border-radius:8px; padding:1rem 1.25rem; color:#ff5252;
                font-size:0.85rem; margin-top:1rem">
        ⚠ &nbsp; {message}
    </div>
    """, unsafe_allow_html=True)


# ── Private helpers ───────────────────────────────────────────────────────────

def _conf_color(conf: float) -> str:
    if conf >= 70:   return "#00e676"
    elif conf >= 50: return "#ffb300"
    return "#ff5252"


def _render_bar_chart(predictions: list[dict]):
    classes = [p["class"] for p in predictions]
    confs   = [p["confidence"] for p in predictions]
    colors  = [_conf_color(c) for c in confs]

    fig = go.Figure(go.Bar(
        x=confs,
        y=classes,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{c:.1f}%" for c in confs],
        textposition="outside",
        textfont=dict(color="#888", size=12, family="DM Sans"),
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        margin=dict(l=10, r=60, t=10, b=10),
        height=160,
        xaxis=dict(
            range=[0, max(confs) * 1.25],
            showgrid=False, zeroline=False, showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#888", size=12, family="DM Sans"),
            autorange="reversed"
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})