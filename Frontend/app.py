# frontend/app.py
import streamlit as st
from ui import (
    inject_styles,
    render_hero,
    render_placeholder,
    render_result_card,
    render_footer,
    render_error,
)
from apiClient import (
    get_samples,
    get_sample_video,
    predict_upload,
    predict_sample,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Event Detection",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_styles()
render_hero()

# ── Mode selector ─────────────────────────────────────────────────────────────
_, col, _ = st.columns([1, 2, 1])
with col:
    mode = st.radio(
        "",
        ["📁  Upload clip", "🎬  Sample clips"],
        horizontal=True,
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

video_bytes     = None
sample_filename = None
ready           = False

with left:
    # ── Upload mode ───────────────────────────────────────────────────────────
    if "Upload" in mode:
        st.markdown('<div class="section-label">Upload your clip</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a football clip here",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed"
        )
        if uploaded:
            video_bytes = uploaded.getvalue()
            st.video(video_bytes)
            ready = True

    # ── Sample mode ───────────────────────────────────────────────────────────
    else:
        st.markdown('<div class="section-label">Select a sample</div>', unsafe_allow_html=True)
        try:
            clips = get_samples()
        except ConnectionError as e:
            render_error(str(e))
            clips = []

        if clips:
            names           = [c["name"] for c in clips]
            selected        = st.selectbox("", names, label_visibility="collapsed")
            sample_filename = next(c["filename"] for c in clips if c["name"] == selected)

            preview = get_sample_video(sample_filename)
            if preview:
                st.video(preview)
            else:
                st.caption(f"Selected: {selected}")

            ready = True

    st.markdown("<br>", unsafe_allow_html=True)
    analyse = st.button("⚡  ANALYSE CLIP", use_container_width=True, disabled=not ready)

# ── Results panel ─────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Analysis result</div>', unsafe_allow_html=True)

    if not analyse:
        render_placeholder()
    else:
        with st.spinner("Analysing..."):
            try:
                if "Upload" in mode:
                    predictions = predict_upload(video_bytes)
                else:
                    predictions = predict_sample(sample_filename)

                render_result_card(predictions)

            except ConnectionError as e:
                render_error(str(e))
            except TimeoutError as e:
                render_error(str(e))
            except RuntimeError as e:
                render_error(str(e))
            except Exception as e:
                render_error(f"Unexpected error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
render_footer()