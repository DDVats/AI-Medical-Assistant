import streamlit as st
from PIL import Image
import requests
import base64
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced AI Medical Assistant",
    page_icon="🩺",
    layout="wide"
)

# --- Main UI ---
st.title("Advanced AI Medical Assistant 🩺")
st.markdown("This tool provides AI-powered diagnostic assistance with integrated clinical explanations. Please select a module to begin.")

# --- Select Diagnostic Module ---
disease_option = st.selectbox(
    "Which diagnostic module would you like to use?",
    ("Select a module...", "Heart Disease (ECG Analysis)", "Lung Disease (Chest X-Ray Analysis)")
)

# --- Heart Disease Module ---
if disease_option == "Heart Disease (ECG Analysis)":
    st.header("❤️ Heart Disease Diagnostic Module")
    st.markdown("Upload a patient's ECG data files (`.dat` and `.hea`) to get a diagnosis for Congestive Heart Failure (CHF).")

    dat_file_ecg = st.file_uploader("1. Select the patient's .dat file", type=['dat'], key="ecg_dat")
    hea_file_ecg = st.file_uploader("2. Select the patient's .hea file", type=['hea'], key="ecg_hea")

    if st.button("Diagnose Heart Condition"):
        if dat_file_ecg is not None and hea_file_ecg is not None:
            with st.spinner("Analyzing ECG signal..."):
                files = {
                    'dat_file': (dat_file_ecg.name, dat_file_ecg.getvalue()),
                    'hea_file': (hea_file_ecg.name, hea_file_ecg.getvalue())
                }
                api_url = "http://127.0.0.1:5000/predict_ecg"
                try:
                    response = requests.post(api_url, files=files)
                    st.subheader("Diagnosis Result")
                    if response.status_code == 200:
                        data = response.json()
                        prediction = data.get('prediction')
                        confidence = data.get('confidence_score', 0) * 100

                        # Display prediction with explanation
                        if prediction == "CHF Positive":
                            st.error(f"**Prediction:** {prediction}")
                            st.info("This suggests the patient may have signs of Congestive Heart Failure. Consult a cardiologist for confirmation.")
                        else:
                            st.success(f"**Prediction:** {prediction}")
                            st.info("This suggests no significant signs of Congestive Heart Failure based on the ECG segment provided.")

                        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                    else:
                        st.error(f"An error occurred: {response.json().get('error', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Connection Error: Could not connect to the API server. Is 'app.py' running?")
        else:
            st.warning("Please upload both the .dat and .hea files.")

# --- Lung Disease Module ---
elif disease_option == "Lung Disease (Chest X-Ray Analysis)":
    st.header("🫁 Lung Disease Diagnostic Module")
    st.markdown("Upload a patient's chest X-ray image to get a diagnosis for Pneumonia, complete with an AI-generated explanation.")

    xray_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpeg", "jpg", "png"], key="xray_img")

    if xray_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(xray_file)
            st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

        with col2:
            if st.button("Diagnose Lung Condition"):
                with st.spinner("Analyzing X-ray image with advanced model..."):
                    files = {'file': (xray_file.name, xray_file.getvalue())}
                    api_url = "http://127.0.0.1:5000/predict_xray"
                    try:
                        response = requests.post(api_url, files=files)
                        st.subheader("Diagnosis Result")
                        if response.status_code == 200:
                            data = response.json()
                            prediction = data.get('prediction')
                            confidence = data.get('confidence_score', 0) * 100

                            # Display prediction with explanation
                            if prediction.upper() == "PNEUMONIA":
                                st.error(f"**Prediction:** {prediction}")
                                st.info("The model indicates possible pneumonia. Please consult a pulmonologist or radiologist for confirmation.")
                            else:
                                st.success(f"**Prediction:** {prediction}")
                                st.info("No signs of pneumonia detected in the uploaded chest X-ray image.")

                            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

                            # --- Display Grad-CAM XAI Image ---
                            gradcam_b64 = data.get('gradcam_image_base64')
                            if gradcam_b64:
                                st.markdown("---")
                                st.subheader("AI Explanation (Grad-CAM Heatmap)")
                                st.markdown("Highlighted areas show where the AI model focused its attention for the diagnosis.")
                                img_bytes = base64.b64decode(gradcam_b64)
                                st.image(img_bytes, caption="Attention Heatmap", use_column_width=True)

                        else:
                            st.error(f"An error occurred: {response.json().get('error', 'Unknown error')}")
                    except requests.exceptions.ConnectionError:
                        st.error("Connection Error: Could not connect to the API server. Is 'app.py' running?")

# --- Sidebar Disclaimer ---
st.sidebar.title("About this Tool")
with st.sidebar.expander("⚠️ Important Disclaimer", expanded=True):
    st.warning("""
    - **This is not a medical device.** This tool is a proof-of-concept.
    - **Consult a healthcare professional.** The predictions are for research/demo purposes and not a substitute for professional medical advice.
    """)
