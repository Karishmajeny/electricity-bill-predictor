import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------- LOAD MODEL ----------------
model = joblib.load("model/bill_model.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Electricity Bill Predictor",
    layout="centered"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 About This App")
st.sidebar.write("AI-based Electricity Bill Predictor")
st.sidebar.write("Built using Machine Learning + Streamlit")
st.sidebar.write("Generates bill + report")

# ---------------- TITLE ----------------
st.title("⚡ Smart Electricity Bill Predictor")
st.subheader("Enter household electricity usage")

st.markdown("---")

# ---------------- INPUT UI ----------------
col1, col2 = st.columns(2)

with col1:
    fans = st.number_input("Fans", 1, 10, 2)
    lights = st.number_input("Lights", 1, 20, 5)
    ac_hours = st.number_input("AC Hours/day", 0, 24, 2)
    fridge = st.number_input("Fridge", 1, 3, 1)

with col2:
    tv_hours = st.number_input("TV Hours/day", 0, 10, 2)
    washing_machine = st.number_input("Washing Machine", 0, 2, 1)
    water_heater = st.number_input("Water Heater", 0, 3, 1)
    tariff = st.number_input("Tariff per unit", 5.0, 15.0, 7.0)

st.markdown("---")

# ---------------- PDF FUNCTION ----------------
def generate_pdf(prediction):

    file_name = "Electricity_Report.pdf"
    c = canvas.Canvas(file_name, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "Electricity Bill Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Predicted Bill: ₹ {prediction:.2f}")

    c.drawString(50, 690, f"Fans: {fans}")
    c.drawString(50, 670, f"Lights: {lights}")
    c.drawString(50, 650, f"AC Hours: {ac_hours}")
    c.drawString(50, 630, f"Fridge: {fridge}")
    c.drawString(50, 610, f"TV Hours: {tv_hours}")
    c.drawString(50, 590, f"Washing Machine: {washing_machine}")
    c.drawString(50, 570, f"Water Heater: {water_heater}")
    c.drawString(50, 550, f"Tariff: {tariff}")

    c.drawString(50, 520, "Generated using AI Electricity Predictor")

    c.save()

    return file_name

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Bill"):

    features = np.array([[
        fans, lights, ac_hours, fridge,
        tv_hours, washing_machine, water_heater, tariff
    ]])

    prediction = model.predict(features)[0]

    st.success(f"💰 Estimated Monthly Bill: ₹ {prediction:.2f}")

    # ---------------- INSIGHTS ----------------
    st.markdown("## 📌 Insights")

    if prediction < 500:
        st.info("Low electricity usage ⚡ Efficient usage")
    elif prediction < 1000:
        st.warning("Moderate usage ⚠ Try saving energy")
    else:
        st.error("High usage 🔥 Reduce consumption")

    # ---------------- CHART DATA ----------------
    labels = ["Fans", "Lights", "AC", "Fridge", "TV", "WM", "Heater"]

    values = [
        fans * 5,
        lights * 2,
        ac_hours * 15,
        fridge * 10,
        tv_hours * 3,
        washing_machine * 20,
        water_heater * 25
    ]

    # ---------------- BAR CHART ----------------
    st.markdown("## 📊 Energy Usage (Bar Chart)")

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Energy Units")
    ax.set_title("Appliance-wise Consumption")
    plt.xticks(rotation=45)

    st.pyplot(fig)

    # ---------------- PIE CHART ----------------
    st.markdown("## 🥧 Energy Distribution")

    fig2, ax2 = plt.subplots()
    ax2.pie(values, labels=labels, autopct="%1.1f%%")

    st.pyplot(fig2)

    # ---------------- PDF DOWNLOAD ----------------
    pdf_file = generate_pdf(prediction)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name="Electricity_Bill_Report.pdf",
            mime="application/pdf"
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Machine Learning + Streamlit")