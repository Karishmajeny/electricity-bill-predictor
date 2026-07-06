import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from reportlab.pdfgen import canvas
import io

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart Electricity AI", layout="wide")

st.title("⚡ Smart Electricity AI System (Industry Version)")

# Load model
model = joblib.load("model/bill_model.pkl")

# History file
DATA_FILE = "history.json"

# ---------------- HISTORY FUNCTIONS ----------------
def load_history():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🏠 Appliance Usage")

fan = st.sidebar.slider("Fan", 0, 10, 2)
ac = st.sidebar.slider("AC", 0, 5, 1)
fridge = st.sidebar.slider("Fridge", 0, 3, 1)
washing = st.sidebar.slider("Washing Machine", 0, 3, 1)
mixer = st.sidebar.slider("Mixer", 0, 5, 1)
hair = st.sidebar.slider("Hair Dryer", 0, 3, 0)
lights = st.sidebar.slider("Lights", 0, 20, 5)
ups = st.sidebar.slider("UPS", 0, 3, 1)
chargers = st.sidebar.slider("Chargers", 0, 10, 3)

hours = st.slider("Daily Usage Hours", 1, 24, 6)

appliances = ["Fan","AC","Fridge","Washing","Mixer","HairDryer","Lights","UPS","Chargers"]
usage = [fan, ac, fridge, washing, mixer, hair, lights, ups, chargers]

input_data = np.array([[fan, ac, fridge, washing, mixer,
                        hair, lights, ups, chargers, hours]])

# ---------------- PREDICTION ----------------
if st.button("Predict Bill ⚡"):

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Bill: ₹ {prediction:.2f}")

    # ---------------- SAVE HISTORY ----------------
    history = load_history()

    record = {
        "date": str(datetime.now().date()),
        "bill": float(prediction)
    }

    history.append(record)
    save_history(history)

    # ---------------- AI INSIGHTS ----------------
    st.subheader("💡 AI Insights")

    max_index = np.argmax(usage)
    st.warning(f"Highest usage: {appliances[max_index]}")

    if ac > 2:
        st.info("AC usage is high → consider reducing usage")

    if lights > 10:
        st.info("Lighting usage high → switch to LED")

    if sum(usage) > 25:
        st.error("Overall consumption is high")

    # ---------------- BAR CHART ----------------
    st.subheader("📊 Appliance Usage")
    fig, ax = plt.subplots()
    ax.bar(appliances, usage)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ---------------- PIE CHART ----------------
    st.subheader("🥧 Energy Share")
    fig2, ax2 = plt.subplots()
    ax2.pie(usage, labels=appliances, autopct="%1.1f%%")
    st.pyplot(fig2)

    # ---------------- PDF REPORT ----------------
    st.subheader("📄 Download Report")

    def generate_pdf():
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer)

        c.drawString(100, 800, "Electricity Bill Report")
        c.drawString(100, 780, f"Bill: ₹{prediction:.2f}")
        c.drawString(100, 760, f"Hours: {hours}")

        y = 720
        for i in range(len(appliances)):
            c.drawString(100, y, f"{appliances[i]}: {usage[i]}")
            y -= 20

        c.save()
        buffer.seek(0)
        return buffer

    pdf = generate_pdf()

    st.download_button(
        "Download PDF",
        pdf,
        file_name="bill_report.pdf",
        mime="application/pdf"
    )

# ---------------- HISTORY SECTION ----------------
st.subheader("📅 Monthly History")

history = load_history()

if history:
    df_hist = pd.DataFrame(history)
    st.dataframe(df_hist)

    df_hist["date"] = pd.to_datetime(df_hist["date"])
    df_hist = df_hist.sort_values("date")

    st.subheader("📈 Expense Trend")

    fig3, ax3 = plt.subplots()
    ax3.plot(df_hist["date"], df_hist["bill"], marker="o")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

else:
    st.info("No history yet")