import streamlit as st
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🩺",
    layout="centered"
)

# --- UI Elements ---
st.title("🩺 AI Medical Assistant")
st.write(
    "Select a patient's ECG data files (.dat and .hea) to get an AI-powered diagnosis for Congestive Heart Failure (CHF)."
)

# File uploaders
dat_file = st.file_uploader("1. Select the patient's .dat file", type=['dat'])
hea_file = st.file_uploader("2. Select the patient's .hea file", type=['hea'])

# Diagnose button
if st.button("Diagnose Patient ECG"):
    if dat_file is not None and hea_file is not None:
        with st.spinner("Analyzing ECG signal... Please wait."):
            # Prepare files to send to the API
            files = {
                'dat_file': (dat_file.name, dat_file.getvalue(), 'application/octet-stream'),
                'hea_file': (hea_file.name, hea_file.getvalue(), 'text/plain')
            }

            # API endpoint URL
            api_url = "http://127.0.0.1:5000/predict"

            try:
                # Send request to the Flask API
                response = requests.post(api_url, files=files)

                # Display the results
                st.subheader("Diagnosis Result")
                if response.status_code == 200:
                    data = response.json()
                    prediction = data.get('prediction')
                    confidence = data.get('confidence_score', 0) * 100

                    if prediction == "CHF Positive":
                        st.error(f"**Prediction:** {prediction}")
                    else:
                        st.success(f"**Prediction:** {prediction}")

                    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

                    st.info(
                        "This is an AI-generated prediction and should be reviewed by a qualified medical professional."
                    )
                else:
                    error_data = response.json()
                    st.error(f"An error occurred: {error_data.get('error', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Connection Error: Could not connect to the API server. Please ensure 'app.py' is running."
                )
    else:
        st.warning("Please upload both the .dat and .hea files before diagnosing.")
