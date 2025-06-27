import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import BertTokenizer, BertForSequenceClassification
import tempfile
import torch.nn.functional as F

# Load CNN model for image prediction
cnn_model = tf.keras.models.load_model("asd_cnn_model_Phase2_Test_ImageData.h5")

# Load BERT model for text prediction with extended features
text_model = BertForSequenceClassification.from_pretrained("as_bert_model")
tokenizer = BertTokenizer.from_pretrained("as_bert_model")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["ASD Screening"])

if page == "ASD Screening":
    st.title("🧠 ASD Screening")

    option = st.radio("Choose an option:", ["📤 Upload an Image", "📸 Take a Live Photo"])

    image = None

    if option == "📤 Upload an Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

    elif option == "📸 Take a Live Photo":
        capture_button = st.button("Open Camera and Capture")
        if capture_button:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot access camera.")
            else:
                st.info("📷 Press 'q' to capture the image.")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("⚠️ Failed to capture frame.")
                        break
                    cv2.imshow("Live Camera - Press 'q' to capture", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        img_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                        cv2.imwrite(img_path, frame)
                        break
                cap.release()
                cv2.destroyAllWindows()

                image = Image.open(img_path).convert("RGB")

    if image:
        # Display and preprocess image
        st.image(image, caption="Selected Image", use_column_width=True)
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = cnn_model.predict(img_array)[0][0]
        confidence = round(float(prediction) * 100, 2)
        result = "🟩 No ASD Detected" if prediction > 0.5 else "🟥 ASD Detected"

        st.subheader("📈 Image Prediction Result")
        st.write(f"**Prediction:** {result}")
        st.write(f"**Confidence Score:** {confidence}%")

elif page == "ASD Screening for Adults":
    st.title("ASD Screening for Adults ")

    st.subheader("Answer the following 10 screening questions:")
    questions = [
        "Do you often notice small sounds when others do not?",
        "Do you usually concentrate more on small details, rather than the whole picture?",
        "Do you find it difficult to multitask?",
        "You cannot switch back to work after an interruption",
        "Do you find it difficult to 'read between the lines' when someone is talking?",
        "You can’t tell if someone listening to you is getting bored?",
        "You can’t imagine the characters when you are reading a story?",
        "You never pick up on social cues?",
        "Do you find it difficult to work out what someone is thinking or feeling just by looking at their face?",
        "Is it difficult to interpret facial expressions?"
    ]

    responses = []
    for i, q in enumerate(questions):
        response = st.radio(q, ["Yes", "No"], key=f"q{i}")
        responses.append("1" if response == "Yes" else "0")

    #st.subheader("Additional Demographic Information:")
    age = st.number_input("Enter your age:", min_value=1, max_value=100)

    ethnicity = st.selectbox("Select your ethnicity:", [
        "White-European", "Black", "Asian", "Middle Eastern", "Latino", "South Asian", "Others"
    ])

    jaundice = st.radio("Have you had jaundice (at birth)?", ["yes", "no"])

    country = st.selectbox("Select your country of residence:", [
        "United States", "United Kingdom", "India", "Australia", "New Zealand", "Canada", "Others"
    ])

    if st.button("Predict from Text"):
        base_text = " ".join([f"A{i+1} score is {resp}" for i, resp in enumerate(responses)])
        full_text = f"{base_text}. Age is {age}. Ethnicity is {ethnicity}. Had jaundice: {jaundice}. Country: {country}."

        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = text_model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item() * 100

            label = "🟩 No ASD Detected" if prediction == 0 else "🟥 ASD Detected"

            st.success(f"✅ Text Prediction: {label}")
            st.write(f"**Raw Logits:** {logits.numpy()}")
            st.write(f"**Probabilities:** {probabilities.numpy()}")
            st.write(f"**Confidence:** {confidence:.2f}%")
