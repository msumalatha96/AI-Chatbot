from pathlib import Path
import io
import re
import tempfile
import hashlib

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from deep_translator import GoogleTranslator
from gtts import gTTS

import speech_recognition as sr


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Farmer Assistant",
    page_icon="🌾",
    layout="centered"
)


# ============================================================
# COMPACT CHAT UI
# ============================================================

st.markdown("""
<style>

/* -----------------------------------------------------------
   MAIN PAGE
----------------------------------------------------------- */

.block-container {
    padding-bottom: 90px !important;
}


/* -----------------------------------------------------------
   COMPACT COMPOSER
----------------------------------------------------------- */

.compact-composer {
    width: 100%;
}


/* -----------------------------------------------------------
   FILE UPLOADER (small icon)
----------------------------------------------------------- */

div[data-testid="stFileUploader"] {
    width: 34px !important;
    min-width: 34px !important;
    max-width: 34px !important;
    height: 34px !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

div[data-testid="stFileUploader"] section {
    width: 34px !important;
    height: 34px !important;
    min-height: 34px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
}

div[data-testid="stFileUploader"] section > div {
    width: 34px !important;
    height: 34px !important;
    padding: 0 !important;
    margin: 0 !important;
}

div[data-testid="stFileUploader"] button {
    width: 30px !important;
    height: 30px !important;
    min-height: 30px !important;
    padding: 0 !important;
    margin: 2px !important;
    border-radius: 50% !important;
    font-size: 12px !important;
}

div[data-testid="stFileUploader"] button svg {
    width: 14px !important;
    height: 14px !important;
}


/* -----------------------------------------------------------
   AUDIO INPUT (small icon)
----------------------------------------------------------- */

div[data-testid="stAudioInput"] {
    width: 34px !important;
    min-width: 34px !important;
    max-width: 34px !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

div[data-testid="stAudioInput"] button {
    width: 30px !important;
    height: 30px !important;
    min-height: 30px !important;
    padding: 0 !important;
    margin: 2px !important;
    border-radius: 50% !important;
}

div[data-testid="stAudioInput"] button svg {
    width: 14px !important;
    height: 14px !important;
}

div[data-testid="stAudioInput"] svg {
    width: 14px !important;
    height: 14px !important;
}


/* -----------------------------------------------------------
   TEXT INPUT
----------------------------------------------------------- */

.compact-text-input input {
    height: 40px !important;
    border-radius: 20px !important;
}


/* -----------------------------------------------------------
   SEND BUTTON
----------------------------------------------------------- */

.send-button button {
    width: 36px !important;
    height: 36px !important;
    min-height: 36px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    font-size: 16px !important;
}


/* -----------------------------------------------------------
   COLUMN SPACING
----------------------------------------------------------- */

div[data-testid="column"] {
    padding-left: 2px !important;
    padding-right: 2px !important;
}


/* -----------------------------------------------------------
   HIDE FILE UPLOADER EXTRA TEXT
----------------------------------------------------------- */

div[data-testid="stFileUploader"] small {
    display: none !important;
}


/* -----------------------------------------------------------
   COMPACT LANGUAGE SELECTOR (top-right, small)
----------------------------------------------------------- */

div[data-testid="stSelectbox"] {
    max-width: 130px !important;
    margin-left: auto !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    min-height: 30px !important;
    font-size: 12px !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    padding-top: 2px !important;
    padding-bottom: 2px !important;
    padding-left: 8px !important;
    font-size: 12px !important;
}

div[data-testid="stSelectbox"] svg {
    width: 14px !important;
    height: 14px !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT
    / "models"
    / "plant_disease_yolo"
    / "weights"
    / "best.pt"
)

KB_PATH = (
    PROJECT
    / "knowledge"
    / "agriculture_knowledge.csv"
)


# ============================================================
# LANGUAGES
# ============================================================

LANGUAGES = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Punjabi": "pa",
}

if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT / "best.pt"

KB_PATH = PROJECT / "agriculture_knowledge.csv"
# ============================================================
# SAFETY
# ============================================================

SAFETY = (
    "Always follow product-label instructions and consult "
    "your local agricultural officer before applying "
    "fertilizers, pesticides, or fungicides."
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_image_hash" not in st.session_state:
    st.session_state.processed_image_hash = ""

if "processed_audio_hash" not in st.session_state:
    st.session_state.processed_audio_hash = ""

if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0


# ============================================================
# TITLE + COMPACT LANGUAGE SELECTOR (tucked to one side)
# ============================================================

title_col, lang_col = st.columns([0.75, 0.25])

with title_col:

    st.title("🌾 AI Farmer Assistant")

    st.caption(
        "Crop disease detection, farming questions and voice assistance"
    )

with lang_col:

    language_name = st.selectbox(
        "🌐",
        list(LANGUAGES.keys()),
        label_visibility="collapsed"
    )

language_code = LANGUAGES[language_name]


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        return None

    try:

        return YOLO(str(MODEL_PATH))

    except Exception as e:

        print("YOLO MODEL ERROR:", e)

        return None


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_data
def load_knowledge_base():

    if not KB_PATH.exists():

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            KB_PATH,
            encoding="utf-8"
        ).fillna("")

        required_columns = [
            "Plant",
            "Disease",
            "Symptoms",
            "Cause",
            "Fertilizer",
            "Precautions",
            "Treatment"
        ]

        for column in required_columns:

            if column not in df.columns:

                df[column] = ""

        return df

    except Exception as e:

        print("KNOWLEDGE BASE ERROR:", e)

        return pd.DataFrame()


# ============================================================
# TRANSLATION
# ============================================================

def translate_text(text, target_language):

    if not text:
        return ""

    text = str(text)

    if target_language == "en":
        return text

    try:

        return GoogleTranslator(
            source="auto",
            target=target_language
        ).translate(text)

    except Exception as e:

        print("TRANSLATION ERROR:", e)

        return text


def translate_to_english(text, source_language):

    if not text:
        return ""

    text = str(text)

    if source_language == "en":
        return text

    try:

        return GoogleTranslator(
            source=source_language,
            target="en"
        ).translate(text)

    except Exception:

        try:

            return GoogleTranslator(
                source="auto",
                target="en"
            ).translate(text)

        except Exception:

            return text


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# KNOWLEDGE SEARCH
# ============================================================

@st.cache_resource
def create_vectorizer(texts):

    if not texts:
        return None, None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


def search_knowledge(question, top_k=3):

    df = load_knowledge_base()

    if df.empty:
        return []

    documents = []

    for _, row in df.iterrows():

        document = " ".join([
            str(row.get("Plant", "")),
            str(row.get("Disease", "")),
            str(row.get("Symptoms", "")),
            str(row.get("Cause", "")),
            str(row.get("Fertilizer", "")),
            str(row.get("Precautions", "")),
            str(row.get("Treatment", ""))
        ])

        documents.append(document)

    vectorizer, matrix = create_vectorizer(
        tuple(documents)
    )

    if vectorizer is None:
        return []

    query_vector = vectorizer.transform(
        [question]
    )

    scores = cosine_similarity(
        query_vector,
        matrix
    )[0]

    indices = scores.argsort()[::-1]

    results = []

    for index in indices[:top_k]:

        score = float(scores[index])

        if score < 0.10:
            continue

        row = df.iloc[index]

        results.append({
            "score": score,
            "Plant": str(row["Plant"]),
            "Disease": str(row["Disease"]),
            "Symptoms": str(row["Symptoms"]),
            "Cause": str(row["Cause"]),
            "Fertilizer": str(row["Fertilizer"]),
            "Precautions": str(row["Precautions"]),
            "Treatment": str(row["Treatment"])
        })

    return results


# ============================================================
# AGRICULTURE QUESTION CHECK
# ============================================================

def is_agriculture_question(question):

    keywords = [
        "plant",
        "crop",
        "farm",
        "farmer",
        "farming",
        "soil",
        "fertilizer",
        "fertiliser",
        "pesticide",
        "fungicide",
        "herbicide",
        "disease",
        "leaf",
        "leaves",
        "seed",
        "irrigation",
        "water",
        "rice",
        "wheat",
        "tomato",
        "potato",
        "apple",
        "grape",
        "corn",
        "maize",
        "soybean",
        "soyabean",
        "pepper",
        "chilli",
        "brinjal",
        "mango",
        "banana",
        "sugarcane",
        "groundnut",
        "papaya",
        "cultivation",
        "harvest",
        "root",
        "stem",
        "fruit",
        "flower",
        "insect",
        "pest",
        "fungus",
        "infection",
        "nutrition",
        "nutrient",
        "spray",
        "weed",
        "mite",
        "blight",
        "rust",
        "spot",
        "mold",
        "rot",
        "scab"
    ]

    question = normalize_text(question)

    return any(
        word in question
        for word in keywords
    )


# ============================================================
# TEXT QUESTION ANSWER
# ============================================================

def answer_question(question, language):

    english_question = translate_to_english(
        question,
        language
    )

    if not is_agriculture_question(
        english_question
    ):

        answer = (
            "I am the AI Farmer Assistant. "
            "Please ask me about crops, plant diseases, "
            "fertilizers, soil, irrigation, pests, "
            "or farming practices."
        )

        return translate_text(
            answer,
            language
        )

    results = search_knowledge(
        english_question
    )

    if not results:

        answer = (
            "Verified information for this question "
            "is not available in the agriculture knowledge "
            "base. I will not guess the answer."
        )

        return translate_text(
            answer,
            language
        )

    record = results[0]

    plant = translate_text(
        record["Plant"],
        language
    )

    disease = translate_text(
        record["Disease"],
        language
    )

    symptoms = translate_text(
        record["Symptoms"],
        language
    )

    cause = translate_text(
        record["Cause"],
        language
    )

    fertilizer = translate_text(
        record["Fertilizer"],
        language
    )

    treatment = translate_text(
        record["Treatment"],
        language
    )

    precautions = translate_text(
        record["Precautions"],
        language
    )

    safety = translate_text(
        SAFETY,
        language
    )

    return f"""
### 🌱 Crop Information

**Plant:** {plant}

**Disease:** {disease}

### 🔍 Symptoms

{symptoms}

### 🧬 Cause

{cause}

### 🌾 Fertilizer

{fertilizer}

### 💊 Treatment

{treatment}

### 🛡️ Precautions

{precautions}

---

⚠️ {safety}
""".strip()


# ============================================================
# READ IMAGE
# ============================================================

def read_image(uploaded_file):

    if uploaded_file is None:
        return None, None

    try:

        image_bytes = uploaded_file.getvalue()

        if not image_bytes:
            return None, None

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

        image = image.convert("RGB")

        return image, image_bytes

    except UnidentifiedImageError:

        return None, None

    except Exception as e:

        print("IMAGE ERROR:", e)

        return None, None


# ============================================================
# MATCH YOLO RESULT TO KNOWLEDGE BASE
# ============================================================

def match_prediction(prediction):

    df = load_knowledge_base()

    if df.empty:
        return None

    prediction = normalize_text(
        prediction
    )

    # Exact / partial match

    for _, row in df.iterrows():

        plant = normalize_text(
            row["Plant"]
        )

        disease = normalize_text(
            row["Disease"]
        )

        combined = (
            plant + " " + disease
        ).strip()

        if prediction == disease:
            return row.to_dict()

        if prediction == combined:
            return row.to_dict()

        if disease and disease in prediction:
            return row.to_dict()

        if prediction and prediction in combined:
            return row.to_dict()

    # Disease keywords

    keywords = [
        "early blight",
        "late blight",
        "bacterial spot",
        "septoria",
        "black rot",
        "rust",
        "mosaic virus",
        "yellow virus",
        "mold",
        "powdery mildew",
        "spider mites",
        "scab",
        "leaf spot"
    ]

    for keyword in keywords:

        if keyword in prediction:

            for _, row in df.iterrows():

                disease = normalize_text(
                    row["Disease"]
                )

                if keyword in disease:

                    return row.to_dict()

    return None


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image, language):

    model = load_model()

    if model is None:

        return (
            "The YOLO model could not be loaded. "
            "Please check this file:\n\n"
            f"{MODEL_PATH}"
        )

    try:

        results = model.predict(
            source=image,
            conf=0.10,
            imgsz=416,
            verbose=False
        )

    except Exception as e:

        print("YOLO ERROR:", repr(e))

        return (
            "The image was received, but image analysis "
            "failed.\n\n"
            f"Technical error: {e}"
        )

    if not results:

        return (
            "The image was received, but the model "
            "returned no result."
        )

    result = results[0]

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

        return translate_text(
            "No plant disease was detected. "
            "Please upload a clear image of the affected "
            "leaf with good lighting.",
            language
        )

    predictions = []

    for i in range(
        len(result.boxes)
    ):

        try:

            class_id = int(
                result.boxes.cls[i].item()
            )

            confidence = float(
                result.boxes.conf[i].item()
            )

            if isinstance(
                result.names,
                dict
            ):

                name = str(
                    result.names[class_id]
                )

            else:

                name = str(
                    result.names[class_id]
                )

            predictions.append({
                "name": name,
                "confidence": confidence
            })

        except Exception as e:

            print(
                "PREDICTION ERROR:",
                e
            )

    if not predictions:

        return (
            "The model returned a prediction, "
            "but it could not be read."
        )

    predictions.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    best = predictions[0]

    prediction_name = best["name"]

    confidence = (
        best["confidence"] * 100
    )

    translated_prediction = translate_text(
        prediction_name,
        language
    )

    # Match to verified knowledge

    record = match_prediction(
        prediction_name
    )

    # Low confidence

    if best["confidence"] < 0.40:

        return f"""
### 🔎 Image Analysis

**Prediction:** {translated_prediction}

**Confidence:** {confidence:.2f}%

⚠️ **Low confidence**

{translate_text(
    "The model is not confident enough to provide "
    "a reliable diagnosis. Please upload a clearer "
    "photo of the affected leaf.",
    language
)}
""".strip()

    # No KB match

    if record is None:

        return f"""
### 🔎 Image Analysis

**Prediction:** {translated_prediction}

**Confidence:** {confidence:.2f}%

⚠️ **Verification required**

{translate_text(
    "The model detected this condition, but a matching "
    "verified record was not found in the agriculture "
    "knowledge base. I will not guess treatment or "
    "fertilizer recommendations.",
    language
)}
""".strip()

    # Translate verified information

    plant = translate_text(
        record["Plant"],
        language
    )

    disease = translate_text(
        record["Disease"],
        language
    )

    symptoms = translate_text(
        record["Symptoms"],
        language
    )

    cause = translate_text(
        record["Cause"],
        language
    )

    fertilizer = translate_text(
        record["Fertilizer"],
        language
    )

    treatment = translate_text(
        record["Treatment"],
        language
    )

    precautions = translate_text(
        record["Precautions"],
        language
    )

    safety = translate_text(
        SAFETY,
        language
    )

    response = f"""
### 🔎 Crop Diagnosis

🌱 **Crop:** {plant}

🦠 **Disease:** {disease}

📊 **Confidence:** {confidence:.2f}%

---

### 📚 Verified Guidance

🔍 **Symptoms**

{symptoms}

🧬 **Cause**

{cause}

🌾 **Fertilizer**

{fertilizer}

💊 **Treatment**

{treatment}

🛡️ **Precautions**

{precautions}

---

⚠️ {safety}
"""

    if best["confidence"] < 0.70:

        response += f"""

> ⚠️ **Verification recommended**
>
> {translate_text(
    "The prediction has moderate confidence. "
    "Please verify the diagnosis with an agricultural "
    "professional before applying treatment.",
    language
)}
"""

    # Other predictions

    if len(predictions) > 1:

        response += "\n### 🔎 Other detected possibilities\n\n"

        for item in predictions[1:4]:

            name = translate_text(
                item["name"],
                language
            )

            conf = (
                item["confidence"] * 100
            )

            response += (
                f"- {name}: {conf:.2f}%\n"
            )

    return response.strip()


# ============================================================
# SPEECH TO TEXT
# ============================================================

def speech_to_text(
    audio_file,
    language_code
):

    if audio_file is None:

        return None, "No recording received."

    try:

        audio_bytes = audio_file.getvalue()

        if not audio_bytes:

            return None, "The recording is empty."

        recognizer = sr.Recognizer()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp:

            temp.write(audio_bytes)

            temp_path = temp.name

        try:

            with sr.AudioFile(
                temp_path
            ) as source:

                audio = recognizer.record(
                    source
                )

            text = recognizer.recognize_google(
                audio,
                language=language_code
            )

            return text.strip(), None

        finally:

            try:
                Path(temp_path).unlink()
            except Exception:
                pass

    except sr.UnknownValueError:

        return (
            None,
            "I could not understand the speech. "
            "Please speak clearly and try again."
        )

    except sr.RequestError:

        return (
            None,
            "Speech recognition is unavailable. "
            "Please check your internet connection."
        )

    except Exception as e:

        print("SPEECH ERROR:", repr(e))

        return (
            None,
            f"Could not process the recording: {e}"
        )


# ============================================================
# TEXT TO SPEECH
# ============================================================

def create_audio(text, language):

    try:

        # Remove markdown symbols

        clean = re.sub(
            r"\*\*(.*?)\*\*",
            r"\1",
            text
        )

        clean = re.sub(
            r"#+\s*",
            "",
            clean
        )

        clean = clean.replace(
            "---",
            ""
        )

        clean = re.sub(
            r"[🌱🦠🔍🧬🌾💊🛡️📊⚠️📷🎤🔎]",
            "",
            clean
        )

        clean = clean.strip()

        if not clean:
            return None

        clean = clean[:4500]

        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )

        temp.close()

        tts = gTTS(
            text=clean,
            lang=language,
            slow=False
        )

        tts.save(temp.name)

        with open(
            temp.name,
            "rb"
        ) as f:

            audio = f.read()

        try:
            Path(temp.name).unlink()
        except Exception:
            pass

        return audio

    except Exception as e:

        print("TTS ERROR:", e)

        return None


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        if message.get("image") is not None:

            try:

                st.image(
                    message["image"],
                    caption="Uploaded plant image"
                )

            except Exception:
                pass

        if message.get("content"):

            st.markdown(
                message["content"]
            )

        if message.get("audio"):

            st.audio(
                message["audio"],
                format="audio/mp3"
            )


# ============================================================
# COMPACT CHAT COMPOSER
# ============================================================

st.markdown(
    '<div class="compact-composer">',
    unsafe_allow_html=True
)

col_upload, col_voice, col_text, col_send = st.columns(
    [0.07, 0.07, 0.74, 0.12],
    vertical_alignment="center"
)


# ============================================================
# IMAGE UPLOAD BUTTON
# ============================================================

with col_upload:

    uploaded_image = st.file_uploader(
        "📎",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="plant_image",
        label_visibility="collapsed"
    )


# ============================================================
# VOICE BUTTON
# ============================================================

with col_voice:

    audio_recording = st.audio_input(
        "🎤",
        key=f"voice_{st.session_state.audio_key}",
        label_visibility="collapsed"
    )


# ============================================================
# TEXT INPUT
# ============================================================

with col_text:

    question = st.text_input(
        "",
        placeholder="Ask anything about your crop...",
        key="question_input",
        label_visibility="collapsed"
    )


# ============================================================
# SEND BUTTON
# ============================================================

with col_send:

    send_question = st.button(
        "↑",
        key="send_question",
        use_container_width=True
    )


st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_image is not None:

    image, image_bytes = read_image(
        uploaded_image
    )

    if image is None:

        st.error(
            "The uploaded file is not a valid image."
        )

    else:

        image_hash = hashlib.sha256(
            image_bytes
        ).hexdigest()

        if (
            image_hash
            != st.session_state.processed_image_hash
        ):

            st.session_state.processed_image_hash = (
                image_hash
            )

            with st.spinner(
                "🔎 Analyzing plant image..."
            ):

                response = analyze_image(
                    image,
                    language_code
                )

            st.session_state.messages.append({
                "role": "user",
                "content": "📷 Uploaded a plant image.",
                "image": image_bytes
            })

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

            st.rerun()


# ============================================================
# VOICE PROCESSING
# ============================================================

if audio_recording is not None:

    audio_bytes = audio_recording.getvalue()

    audio_hash = hashlib.sha256(
        audio_bytes
    ).hexdigest()

    if (
        audio_hash
        != st.session_state.processed_audio_hash
    ):

        st.session_state.processed_audio_hash = (
            audio_hash
        )

        with st.spinner(
            "🎤 Understanding your voice..."
        ):

            spoken_text, error = speech_to_text(
                audio_recording,
                language_code
            )

        if error:

            st.error(error)

        elif spoken_text:

            st.session_state.messages.append({
                "role": "user",
                "content": spoken_text
            })

            with st.spinner(
                "🌱 Preparing answer..."
            ):

                response = answer_question(
                    spoken_text,
                    language_code
                )

            with st.spinner(
                "🔊 Preparing voice answer..."
            ):

                audio_answer = create_audio(
                    response,
                    language_code
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "audio": audio_answer
            })

            st.session_state.audio_key += 1

            st.rerun()


# ============================================================
# TEXT QUESTION PROCESSING
# ============================================================

if send_question and question:

    question = question.strip()

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.spinner(
            "🌱 Preparing answer..."
        ):

            response = answer_question(
                question,
                language_code
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        st.rerun()


# ============================================================
# CLEAR CHAT
# ============================================================

if st.session_state.messages:

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.processed_image_hash = ""

        st.session_state.processed_audio_hash = ""

        st.session_state.audio_key += 1

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "⚠️ AI predictions should be verified with a qualified "
    "agricultural professional before treatment."
)