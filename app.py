import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

# --- Configuration & Setup ---
st.set_page_config(
    page_title="AI Based Heart Disease Prediction System", 
    page_icon="💓", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load saved model, scaler, and expected columns
@st.cache_resource
def load_assets():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_assets()

# --- Database Setup ---
DB_FILE = "users_db.json"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

db = load_db()

# --- Session State Initialization ---
if "auth_status" not in st.session_state:
    st.session_state["auth_status"] = "pending"
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "auth_error" not in st.session_state:
    st.session_state["auth_error"] = ""
if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False

# --- Callbacks ---
def set_guest_mode():
    st.session_state["auth_status"] = "guest"
    st.session_state["auth_error"] = ""

def process_login(username, password):
    if not username or not password:
        st.session_state["auth_error"] = "Please enter both username and password."
        return

    if username in db:
        if db[username]["password"] == password:
            st.session_state["auth_status"] = "logged_in"
            st.session_state["username"] = username
            st.session_state["auth_error"] = ""
        else:
            st.session_state["auth_error"] = "Incorrect password!"
    else:
        db[username] = {"password": password, "history": []}
        save_db(db)
        st.session_state["auth_status"] = "logged_in"
        st.session_state["username"] = username
        st.session_state["auth_error"] = ""

def process_logout():
    st.session_state["auth_status"] = "pending"
    st.session_state["username"] = ""
    st.session_state["analysis_complete"] = False

def clear_analysis():
    st.session_state["analysis_complete"] = False

# --- First Page Gatekeeper ---
if st.session_state["auth_status"] == "pending":
    st.markdown("<br><br><br>", unsafe_allow_html=True) 
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    
    with col2:
        st.markdown("<h1 style='text-align: center; color: #e74c3c;'>💓 AI Based Heart Disease Prediction System</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #7f8c8d;'>Professional AI Based Heart Disease Prediction System</p>", unsafe_allow_html=True)
        
        if st.session_state["auth_error"]:
            st.error(st.session_state["auth_error"])
            
        with st.container(border=True):
            tab1, tab2 = st.tabs(["🔐 Sign In / Register", "👤 Guest Mode"])
            with tab1:
                with st.form("login_form"):
                    user_input = st.text_input("Patient ID / Username")
                    pass_input = st.text_input("Password", type="password")
                    if st.form_submit_button("Access Portal", type="primary", use_container_width=True):
                        process_login(user_input, pass_input)
                        st.rerun()
            with tab2:
                st.write("Evaluate your risk without saving data.")
                st.button("Proceed as Guest", use_container_width=True, on_click=set_guest_mode)
    st.stop()


# --- Helper Functions & Visualizations ---
def calculate_bmi(weight, height):
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    if bmi < 18.5: category, color = "Underweight", "#f39c12"
    elif 18.5 <= bmi < 24.9: category, color = "Healthy Weight", "#27ae60"
    elif 25 <= bmi < 29.9: category, color = "Overweight", "#e67e22"
    else: category, color = "Obese", "#c0392b"
    return bmi, category, color

def get_ai_suggestions(patient_summary, risk_status):
    if GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            prompt = f"""
            You are the AI Based Heart Disease Prediction System, an elite AI cardiologist. 
            PATIENT PROFILE: {patient_summary}
            MODEL PREDICTION: {risk_status}
            
            TASK: Write a highly personalized, compassionate medical evaluation. 
            RULES: Explicitly mention their exact numbers. Explain why the model gave them this risk status. Write in 2 flowing paragraphs. NO bullet points.
            """
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400
            )
            return f"{response.choices[0].message.content}\n\n*(✨ Live AI Generation via Llama 3.1)*"
        except Exception as e:
            print(f"\n--- AI API OFFLINE: {str(e)} ---\n")
            pass 
            
    try:
        bmi_match = re.search(r"BMI:\s*([\d\.]+)", patient_summary)
        bmi = float(bmi_match.group(1)) if bmi_match else 25.0
        bp_match = re.search(r"BP:\s*(\d+)", patient_summary)
        bp = int(bp_match.group(1)) if bp_match else 120
        chol_match = re.search(r"Chol:\s*(\d+)", patient_summary)
        chol = int(chol_match.group(1)) if chol_match else 200
    except Exception:
        bmi, bp, chol = 25.0, 120, 200 
        
    narrative = f"Based on clinical parameters, the AI Based Heart Disease Prediction System indicates a {risk_status.lower()} profile. Specifically, your blood pressure is {bp} mmHg and cholesterol is {chol} mg/dL. Given your BMI of {bmi:.1f}, proactive lifestyle management is recommended. Please follow up with a certified cardiologist to monitor these trends."
    return f"{narrative}\n\n*(⚠️ Notice: Displaying Dynamic Medical Engine. Live AI is currently offline.)*"

def plot_professional_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        number = {'suffix': "%", 'font': {'size': 40}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Calculated Risk Probability", 'font': {'size': 18, 'color': 'gray'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "rgba(0,0,0,0)"}, 
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': "#a3e4d7", 'name': 'Low'},
                {'range': [30, 70], 'color': "#f9e79f", 'name': 'Moderate'},
                {'range': [70, 100], 'color': "#f5b7b1", 'name': 'High'}],
            'threshold': {'line': {'color': "#00bcd4", 'width': 6}, 'thickness': 0.75, 'value': probability * 100}
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def plot_clinical_metrics(bp, chol, hr):
    metrics = ['Resting BP', 'Cholesterol', 'Max HR']
    patient_vals = [bp, chol, hr]
    target_vals = [120, 200, 150]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=metrics, x=target_vals, name='Healthy Target Limit',
        orientation='h', marker=dict(color='rgba(189, 195, 199, 0.5)', line=dict(color='#bdc3c7', width=1)),
        hoverinfo='none'
    ))
    fig.add_trace(go.Bar(
        y=metrics, x=patient_vals, name='Patient Metrics',
        orientation='h', marker=dict(color='#00bcd4'),
        text=patient_vals, textposition='auto'
    ))
    fig.update_layout(
        barmode='overlay', height=280, margin=dict(l=20, r=20, t=30, b=10),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# --- SIDEBAR: Navigation & History ---
with st.sidebar:
    st.markdown("## 💓 AI Based Heart Disease Prediction System")
    if st.session_state["auth_status"] == "logged_in":
        st.success(f"👤 Patient ID: **{st.session_state['username']}**")
        st.button("Secure Logout", on_click=process_logout, use_container_width=True)
    else:
        st.info("👤 Mode: Guest")
        st.button("Sign In", on_click=process_logout, use_container_width=True)
        
    st.markdown("---")
    app_mode = st.radio("Navigation Menu", ["🩺 Heart Risk Predictor", "⚖️ BMI Clinical Calculator"], on_change=clear_analysis)
    st.markdown("---")
    
    st.markdown("<h4 style='color: #00bcd4;'>📂 Recent History</h4>", unsafe_allow_html=True)
    if st.session_state["auth_status"] != "logged_in":
        st.caption("Sign in to view your past assessments.")
    else:
        user_history = db[st.session_state["username"]].get("history", [])
        if not user_history:
            st.caption("No assessments recorded yet.")
        else:
            for idx, record in enumerate(user_history[:4]):
                with st.expander(f"Report: {record['date'].split(' ')[0]}", expanded=(idx==0)):
                    st.markdown(f"**Risk:** {record['probability_percent']}%")
                    st.markdown(f"**Status:** {record['prediction']}")

# --- MAIN CONTENT AREA ---
if app_mode == "🩺 Heart Risk Predictor":
    st.title("AI Based Heart Disease Prediction System")
    
    with st.container(border=True):
        st.markdown("<h4 style='color: #00bcd4;'>👤 1. Demographic & Basic Vitals</h4>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: 
            age = st.slider("Patient Age", 18, 100, 45)
            height = st.slider("Patient Height (cm)", 100.0, 250.0, 175.0, step=0.1)
            ft = int(height // 30.48); inc = int(round((height / 2.54) % 12))
            st.caption(f"*(Equivalent to: {ft}' {inc}\")*")
        with c2: 
            sex_input = st.selectbox("Biological Sex", ["Male", "Female"])
            weight = st.slider("Patient Weight (kg)", 30, 200, 75, step=1)

    with st.container(border=True):
        st.markdown("<h4 style='color: #00bcd4;'>🩸 2. Blood & Heart Rate Metrics</h4>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: 
            resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 125)
            max_hr = st.slider("Maximum Heart Rate Achieve (bpm)", 60, 220, 145)
        with c2: 
            cholesterol = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 180)
            fasting_bs_input = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"])

    with st.container(border=True):
        st.markdown("<h4 style='color: #00bcd4;'>🫀 3. Diagnostic Observations</h4>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: 
            chest_pain_input = st.selectbox("Chest Pain Type", ["Typical Angina (TA)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"], index=1)
            oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
        with c2: 
            resting_ecg_input = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
            exercise_angina_input = st.selectbox("Exercise-Induced Angina?", ["No", "Yes"])
            st_slope_input = st.selectbox("ST Slope", ["Upsloping (Up)", "Flat", "Downsloping (Down)"])

    if st.button("🔬 Execute Risk Model Analysis", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            sex = "M" if sex_input == "Male" else "F"
            cp_map = {"Typical Angina (TA)": "TA", "Atypical Angina (ATA)": "ATA", "Non-Anginal Pain (NAP)": "NAP", "Asymptomatic (ASY)": "ASY"}
            ecg_map = {"Normal": "Normal", "ST": "ST", "LVH": "LVH"}
            angina_map = {"Yes": "Y", "No": "N"}
            slope_map = {"Upsloping (Up)": "Up", "Flat": "Flat", "Downsloping (Down)": "Down"}
            
            bmi_val, bmi_cat, _ = calculate_bmi(weight, height)
            input_dict = {'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol, 'FastingBS': 1 if fasting_bs_input == "Yes" else 0, 'MaxHR': max_hr, 'Oldpeak': oldpeak}
            cat_features = {'Sex_' + sex: 1, 'ChestPainType_' + cp_map[chest_pain_input]: 1, 'RestingECG_' + ecg_map[resting_ecg_input]: 1, 'ExerciseAngina_' + angina_map[exercise_angina_input]: 1, 'ST_Slope_' + slope_map[st_slope_input]: 1}
            input_dict.update(cat_features)
            input_df = pd.DataFrame([input_dict])
            for col in expected_columns:
                if col not in input_df.columns: input_df[col] = 0
            input_df = input_df[expected_columns]
            
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)
            try: risk_prob = float(model.predict_proba(scaled_input)[0][1])
            except: risk_prob = 1.0 if prediction[0] == 1 else 0.0
            
            risk_status = "Elevated Risk Detected" if prediction[0] == 1 else "Normal / Low Risk"
            summary = f"Age: {age}, BMI: {bmi_val:.1f} ({bmi_cat}), BP: {resting_bp}, Chol: {cholesterol}, Max HR: {max_hr}"
            ai_advice = get_ai_suggestions(summary, risk_status)
            
            st.session_state["results"] = {"status": risk_status, "prob": risk_prob, "advice": ai_advice, "summary": summary, "flag": int(prediction[0])}
            st.session_state["analysis_complete"] = True

    if st.session_state["analysis_complete"]:
        res = st.session_state["results"]
        st.markdown("---")
        st.markdown("<h3 style='color: #00bcd4;'>📊 Inference Dashboard</h3>", unsafe_allow_html=True)
        if res["flag"] == 1:
            st.error(f"⚠️ Clinical Status: **{res['status']}**")
        else:
            st.success(f"✅ Clinical Status: **{res['status']}**")
            
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_professional_gauge(res["prob"]), use_container_width=True)
        with c2: st.plotly_chart(plot_clinical_metrics(resting_bp, cholesterol, max_hr), use_container_width=True)

        st.markdown("<h3 style='color: #00bcd4;'>🤖 AI Physician Narrative</h3>", unsafe_allow_html=True)
        st.info(res["advice"], icon="🩺")
        
        report_text = f"AI BASED HEART DISEASE PREDICTION SYSTEM REPORT\nDate: {datetime.now()}\n\nSUMMARY: {res['summary']}\nSTATUS: {res['status']}\nCONFIDENCE: {res['prob']*100:.1f}%\n\nADVICE:\n{res['advice']}"
        st.download_button("⬇️ Download Report (TXT)", data=report_text, file_name="AI_Heart_Report.txt", use_container_width=True)

elif app_mode == "⚖️ BMI Clinical Calculator":
    st.markdown("<h2 style='color: #00bcd4;'>Body Mass Index (BMI) Assessment</h2>", unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1: h = st.slider("Height (cm)", 50.0, 250.0, 175.0, step=0.1)
        with c2: w = st.slider("Weight (kg)", 20, 300, 75, step=1)
        if st.button("Calculate BMI", type="primary"):
            bmi_v, cat, color = calculate_bmi(w, h)
            st.markdown("---")
            st.metric("Calculated BMI", f"{bmi_v:.1f}")
            st.markdown(f"<h3 style='color: {color};'>Status: {cat}</h3>", unsafe_allow_html=True)
