from __future__ import annotations

import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from src.dashboard import render_dashboard
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.predict import load_label_names, load_shadowvision_model, predict_image
from src.utils import add_prediction_history, history_dataframe, load_css, metric_card, pil_from_upload, prediction_report, read_text
from src.webcam import ShadowVisionVideoProcessor


st.set_page_config(
    page_title="ShadowVision AI",
    page_icon=":camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()


@st.cache_resource(show_spinner="Loading ShadowVision model...")
def cached_model():
    return load_shadowvision_model()


@st.cache_data(show_spinner=False)
def cached_labels():
    return load_label_names()


model, model_name = cached_model()
labels = cached_labels()

with st.sidebar:
    st.title("ShadowVision AI")
    page = st.radio(
        "Navigation",
        ["Home", "Predict Image", "Live Webcam", "Analytics Dashboard", "Explainable AI", "About Project"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Model status")
    st.success(model_name)
    st.caption("Intel scene classes: Buildings, Forest, Glacier, Mountain, Sea, Street")


def render_feature_grid() -> None:
    st.markdown(
        """
        <div class="feature-grid">
          <div class="feature-card"><strong>Transfer Learning</strong><span>MobileNetV2 backbone with production preprocessing.</span></div>
          <div class="feature-card"><strong>Explainable AI</strong><span>Grad-CAM heatmaps show the image regions driving predictions.</span></div>
          <div class="feature-card"><strong>Analytics</strong><span>Accuracy, loss, class distribution, and confusion matrix views.</span></div>
          <div class="feature-card"><strong>Deployment Ready</strong><span>Docker, Render config, modular source files, and clean docs.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def home_page() -> None:
    st.markdown(
        """
        <section class="hero">
          <div>
            <h1>ShadowVision AI</h1>
            <p>Advanced image classification platform for natural scene recognition, real-time inference, model analytics, and explainable AI storytelling.</p>
            <div class="badge-row">
              <span class="badge">MobileNetV2</span>
              <span class="badge">Grad-CAM</span>
              <span class="badge">Webcam AI</span>
              <span class="badge">Streamlit SaaS UI</span>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    cols = st.columns(4)
    with cols[0]:
        metric_card("Target Accuracy", "92-96%", "After Intel dataset training")
    with cols[1]:
        metric_card("Prediction", "<1 sec", "Optimized image pipeline")
    with cols[2]:
        metric_card("Classes", "6", "Buildings to Street")
    with cols[3]:
        metric_card("Realtime", "15-25 FPS", "Hardware dependent")
    render_feature_grid()


def show_prediction_result(result) -> None:
    st.markdown(
        f"""
        <div class="prediction-card">
          <h2>{result.label}</h2>
          <p>Confidence: <strong>{result.confidence * 100:.2f}%</strong></p>
          <div class="confidence-bar"><div class="confidence-fill" style="width:{result.confidence * 100:.2f}%"></div></div>
          <p style="color:#a6adbb;margin-top:14px;">{result.model_mode}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    for item in result.top_predictions:
        st.progress(float(item["confidence"]), text=f'{item["label"]}: {float(item["confidence"]) * 100:.2f}%')


def predict_page() -> None:
    st.header("Image Prediction")
    uploaded = st.file_uploader("Upload a scene image", type=["jpg", "jpeg", "png", "webp"])
    if not uploaded:
        st.info("Upload an image to run classification and generate a downloadable report.")
        return

    image = pil_from_upload(uploaded)
    left, right = st.columns([0.9, 1.1])
    with left:
        st.image(image, caption=uploaded.name, use_column_width=True)
    with right:
        with st.spinner("ShadowVision is analyzing the image..."):
            result = predict_image(image, model, labels)
        add_prediction_history(uploaded.name, result)
        show_prediction_result(result)
        st.download_button(
            "Download Prediction Report",
            data=prediction_report(result, uploaded.name),
            file_name=f"{uploaded.name.rsplit('.', 1)[0]}_shadowvision_report.json",
            mime="application/json",
        )

    history = history_dataframe()
    if not history.empty:
        st.subheader("Prediction History")
        st.dataframe(history, use_container_width=True, hide_index=True)


def webcam_page() -> None:
    st.header("Live Webcam AI")
    st.caption("Start the camera and ShadowVision will classify incoming frames with FPS overlay.")
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_streamer(
        key="shadowvision-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=lambda: ShadowVisionVideoProcessor(model, labels),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def analytics_page() -> None:
    st.header("Analytics Dashboard")
    st.caption("Uses saved training history when available, with professional preview data in a fresh checkout.")
    render_dashboard(labels)


def explainable_page() -> None:
    st.header("Explainable AI")
    uploaded = st.file_uploader("Upload an image for Grad-CAM", type=["jpg", "jpeg", "png", "webp"], key="gradcam")
    if not uploaded:
        st.info("Upload an image to visualize where the neural network is focusing.")
        return
    image = pil_from_upload(uploaded)
    result = predict_image(image, model, labels)
    left, right = st.columns(2)
    with left:
        st.image(image, caption="Original image", use_column_width=True)
    with right:
        try:
            heatmap = make_gradcam_heatmap(model, image)
            st.image(overlay_heatmap(image, heatmap), caption=f"Grad-CAM: {result.label}", use_column_width=True)
        except Exception as exc:
            st.warning(f"Grad-CAM is available after loading a compatible convolutional model. Details: {exc}")
    show_prediction_result(result)


def about_page() -> None:
    st.header("About Project")
    report = read_text("reports/model_report.txt", "Train the model to generate a fresh evaluation report.")
    st.markdown(
        """
        ShadowVision AI is an internship-grade computer vision platform built around a clean ML workflow:
        dataset preparation, MobileNetV2 transfer learning, image prediction, webcam inference,
        analytics, explainability, and deployment packaging.
        """
    )
    st.subheader("Model Report")
    st.code(report, language="text")
    st.subheader("Recommended Dataset")
    st.write("Intel Image Classification Dataset with Buildings, Forest, Glacier, Mountain, Sea, and Street classes.")


pages = {
    "Home": home_page,
    "Predict Image": predict_page,
    "Live Webcam": webcam_page,
    "Analytics Dashboard": analytics_page,
    "Explainable AI": explainable_page,
    "About Project": about_page,
}

pages[page]()
