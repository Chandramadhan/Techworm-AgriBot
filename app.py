import os
import time
import gdown
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image as keras_image
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper, DuckDuckGoSearchAPIWrapper
from langdetect import detect
from deep_translator import GoogleTranslator
from dotenv import load_dotenv, find_dotenv

# Load env
load_dotenv(find_dotenv())

# Tools
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200))
duckduckgo_search = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(region="in-en", time="y", max_results=2))
tools = [wiki, arxiv, duckduckgo_search]

# Groq LLM
def load_llm():
    return ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

# Language utilities
def translate_to_english(text):
    try:
        detected_lang = detect(text)
        if detected_lang == "en":
            return text, "en"
        translated = GoogleTranslator(source=detected_lang, target="en").translate(text)
        return translated, detected_lang
    except:
        return text, "unknown"

def translate_back(text, lang):
    try:
        if lang == "en":
            return text
        return GoogleTranslator(source="en", target=lang).translate(text)
    except:
        return text


  # ✅ Add this import

def build_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(224, 224, 3)))  # ✅ Streamlit-safe
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(38, activation='softmax'))
    return model



def load_disease_model():
    model_path = "plant_disease.weights.h5"
    if not os.path.exists(model_path):
        with st.spinner("⬇️ Downloading model weights..."):
            gdown.download(id="1BaxOxurxwkTwQVxwodPmMBMeVvc5O6M8", output=model_path, quiet=False)

    model = build_model()
    model.load_weights(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_disease_model()


# Label map
label_map = {
    "Apple___Apple_scab": 0, "Apple___Black_rot": 1, "Apple___Cedar_apple_rust": 2, "Apple___healthy": 3,
    "Blueberry___healthy": 4, "Cherry_(including_sour)___Powdery_mildew": 5, "Cherry_(including_sour)___healthy": 6,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 7, "Corn_(maize)___Common_rust_": 8,
    "Corn_(maize)___Northern_Leaf_Blight": 9, "Corn_(maize)___healthy": 10,
    "Grape___Black_rot": 11, "Grape___Esca_(Black_Measles)": 12, "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 13,
    "Grape___healthy": 14, "Orange___Haunglongbing_(Citrus_greening)": 15, "Peach___Bacterial_spot": 16,
    "Peach___healthy": 17, "Pepper,_bell___Bacterial_spot": 18, "Pepper,_bell___healthy": 19,
    "Potato___Early_blight": 20, "Potato___Late_blight": 21, "Potato___healthy": 22,
    "Raspberry___healthy": 23, "Soybean___healthy": 24, "Squash___Powdery_mildew": 25,
    "Strawberry___Leaf_scorch": 26, "Strawberry___healthy": 27, "Tomato___Bacterial_spot": 28,
    "Tomato___Early_blight": 29, "Tomato___Late_blight": 30, "Tomato___Leaf_Mold": 31,
    "Tomato___Septoria_leaf_spot": 32, "Tomato___Spider_mites Two-spotted_spider_mite": 33,
    "Tomato___Target_Spot": 34, "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 35,
    "Tomato___Tomato_mosaic_virus": 36, "Tomato___healthy": 37
}
inv_label_map = {v: k for k, v in label_map.items()}

def predict_disease(img):
    try:
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction[0])
        return f"Prediction: {inv_label_map.get(class_index, 'Unknown')}"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Agent setup
def get_conversational_agent():
    llm = load_llm()
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=st.session_state.chat_memory,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=15,
        max_execution_time=60,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )

# App UI
def main():
    st.markdown("""
        <style>
        .stApp {
            background: url("https://en.reset.org/app/uploads/2020/06/india_farming.jpg") no-repeat center center fixed;
            background-size: cover;
            background-color: rgba(0, 50, 0, 0.3);
            background-blend-mode: darken;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🌾 Techworm (Multilingual + Disease Detection) 🌾")
    st.subheader("Smart Assistant for Farming + Plant Health")

    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button("Reset Chat"):
        st.session_state.chat_memory.clear()
        st.session_state.messages = []
        st.success("Chat history cleared!")

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    st.markdown("## 📷 Upload a plant image for disease detection")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image_pil = Image.open(uploaded_image)
        st.image(image_pil, caption='Uploaded Image', width=250)
        detection_result = predict_disease(image_pil)
        st.success(detection_result)

    prompt = st.chat_input("Ask your farming-related question in any language...")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            translated_query, original_lang = translate_to_english(prompt)
            st.write(f"🔍 *Detected Language:* {original_lang.upper()}")
            st.write(f"🔄 *Translated Query:* {translated_query}")

            agent = get_conversational_agent()

            def trim_chat_memory(max_length=5):
                history = st.session_state.chat_memory.load_memory_variables({})["chat_history"]
                if len(history) > max_length:
                    st.session_state.chat_memory.chat_memory.messages = history[-max_length:]
                return history

            chat_history = trim_chat_memory(max_length=5)

            full_prompt = f"""
You are a helpful and expert AI assistant for farming and agriculture questions.
IMPORTANT:
- If the question is not related to agriculture, respond: "Please ask questions related to agriculture only."
- Use tools like Wikipedia, Arxiv, DuckDuckGo only if needed (max 2 times).
- Be brief, clear, and accurate.

User's Question: {prompt}
"""

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = agent.invoke({"input": full_prompt})
                    break
                except Exception as e:
                    st.warning(f"⚠ Rate Limit - Retry {attempt + 1}/{max_retries}")
                    time.sleep(2)

            response_text = response["output"] if isinstance(response, dict) and "output" in response else str(response)
            final_response = translate_back(response_text, original_lang)

            st.chat_message("assistant").markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
