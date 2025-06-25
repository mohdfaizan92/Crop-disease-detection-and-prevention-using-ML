import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model("plant_disease_model.h5")

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f]

# Class names must match your training labels
class_names = [
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Corn - Common Rust",
    "Corn - Northern Leaf Blight",
    "Corn - Healthy",
    "Pepper Bell - Bacterial Spot",
    "Pepper Bell - Healthy"
]

# Prevention tips for each class
prevention = {
    "Tomato - Bacterial Spot": "Use copper-based bactericides and avoid overhead watering.",
    "Tomato - Early Blight": "Remove infected leaves and use fungicides.",
    "Tomato - Healthy": "No disease detected. Keep monitoring your crop.",
    "Potato - Early Blight": "Use resistant varieties and practice crop rotation.",
    "Potato - Late Blight": "Apply appropriate fungicides and destroy infected plants.",
    "Potato - Healthy": "No disease detected. Maintain healthy soil.",
    "Corn - Common Rust": "Plant resistant hybrids and apply fungicides if necessary.",
    "Corn - Northern Leaf Blight": "Rotate crops and use resistant varieties.",
    "Corn - Healthy": "No disease detected. Keep up good farming practices.",
    "Pepper Bell - Bacterial Spot": "Avoid working in wet fields and use bactericides.",
    "Pepper Bell - Healthy": "No disease detected. Maintain good hygiene."
}

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Contact"])

st.sidebar.markdown("---")
st.sidebar.info("Developed by Mohd Faizan")

# Page 1: Home
if page == "Home":
    # Show image/banner

    st.title("🌿 Crop Disease Detection & Prevention")

    try:
        header_image = Image.open("home_page.jpeg")
        st.image(header_image, use_column_width=True)
    except:
        st.warning("Logo image not found. Please add 'logo.png' to the folder.")

    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

        # Preprocess image
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        prediction = model.predict(image_array)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        confidence = np.max(prediction) * 100

        st.success(f"✅ Predicted: **{class_name}**")
        # st.info(f"🧠 Confidence: {confidence:.2f}%")
        st.warning(f"🛡️ Prevention Tip: {prevention[class_name]}")

# Page 2: About
elif page == "About":
    st.title("About This Project")
    st.markdown("""
    ### 🌿 Crop Disease Detection and Prevention Using Machine Learning

    This project aims to help farmers, researchers, and agricultural officers detect crop diseases early and suggest prevention tips using deep learning techniques.

    #### 🔍 Problem Statement
    Crop diseases cause significant yield losses and affect food security. Manual disease identification is time-consuming, error-prone, and requires expert knowledge. This system addresses that gap by offering an automated tool using leaf images.

    #### 🧠 How It Works
    - A **Convolutional Neural Network (CNN)** model is trained on the **PlantVillage dataset**.
    - The model can detect diseases in **Tomato, Potato, Corn**, and **Bell Pepper**.
    - Users upload an image of a leaf, and the system returns the predicted disease and a relevant prevention tip.

    #### ⚙️ Technologies Used
    - **Python**
    - **TensorFlow/Keras** (Deep Learning)
    - **Streamlit** (Web interface)
    - **Pillow & NumPy** (Image processing)
    - **PlantVillage Dataset** (from Kaggle)

    #### 🏆 Key Features
    - Supports multiple crops and diseases
    - Easy-to-use web interface
    - Shows prediction confidence
    - Offers prevention suggestions

    #### 📊 Dataset Overview
    The model was trained on thousands of leaf images from the PlantVillage dataset, which contains labeled images for over 30 plant diseases. This project currently supports 11 classes across 4 crops.

    #### 🤝 Future Improvements
    - Add real-time camera support (mobile/webcam)
    - Include more crops and diseases
    - Integrate with a chatbot for guidance
    - Provide localized recommendations using weather and soil data  
    """)

# Page 3: Contact
elif page == "Contact":
    st.title("📬 Contact Us")

    st.markdown("""
    If you have questions, suggestions, or want to collaborate, feel free to reach out!

    ### 📧 Email
    - mohdfaizansaifi92@gmail.com.com

    ### 📱 Phone
    - +91-6397868995

    - 🌐 https://www.instagram.com/its_fai___zu_639/
    
    - Department of Computer Science  
      Shri Ram Murti Smarak College of Engineering Technology & Research  
      Bareill,Uttar Pradesh, india

    ---
    """)

    st.subheader("📣 Send Feedback")
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message")

    if st.button("Submit"):
        if name and email and message:
            st.success("✅ Thank you for your feedback! We’ll get back to you soon.")
        else:
            st.warning("⚠️ Please fill out all fields before submitting.")