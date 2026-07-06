from pyngrok import ngrok
import os

# open tunnel to streamlit
url = ngrok.connect(8501)
print("Public URL:", url)

# run streamlit app
os.system("streamlit run app.py")
