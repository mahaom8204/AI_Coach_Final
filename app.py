import streamlit as st
import cv2
import numpy as np
from models import adaptive_engine
from utils.session_state import init_state


from utils.session_state import init_state
from utils.roadmap_loader import flatten_roadmap
from models import adaptive_engine
from models.chatbot_service import TutorBot
from models.grammar_checker import correct_sentence, highlight_corrections
from models.speech_to_text import transcribe_file
from models.text_to_speech_service import synthesize_tts
from models.emotion_service import predict_emotion_from_frame

st.set_page_config(
    page_title="Adaptive English Coach",
    layout="wide",
    page_icon="🧠"
)

init_state(st)

# lazy init chatbot
if st.session_state.tutorbot is None:
    st.session_state.tutorbot = TutorBot()

st.title("Adaptive English Coach 🧠🇬🇧")
st.caption("Personalized English learning with speech, emotion, grammar, and mastery tracking.")

tabs = st.tabs([
    "My Learning Path",
    "Chat Tutor",
    "Grammar Checker",
    "Speak & Practice",
    "Mood / Engagement"
])

# ---------------- TAB 1: Learning Path ----------------
with tabs[0]:
    st.header("Your Personalized Path")

    # pick topic to inspect (for now list from roadmap)
    all_topics = flatten_roadmap()
    topic_names = [t["title"] for t in all_topics]
    chosen_topic = st.selectbox("Choose a topic you're studying:", topic_names)

    # adaptive difficulty insight from DKT mastery model
    info = adaptive_engine.get_topic_info(
        current_topic=chosen_topic,
        user_results=st.session_state.user_results
    )

    st.subheader("Recommendation")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Topic: {info['topic']}")
        st.write(f"Predicted mastery: {info['predicted_mastery']}")
        st.write(f"CEFR level (model guess): {info['model_level']}")
        st.write(f"Difficulty label: {info['difficulty_label']}")
    with col2:
        st.write(f"Roadmap stage: {info['roadmap_level']}")
        st.write("Description:")
        st.write(info["description"])

    st.markdown("---")
    st.write("Examples / phrases to practice:")
    for t in all_topics:
        if t["title"] == chosen_topic:
            for ex in t["examples"]:
                st.write(f"- {ex}")

    st.info(
        "Tip: If mastery is low and difficulty is Hard/Very Hard, we should revise fundamentals "
        "before moving to next level."
    )

# ---------------- TAB 2: Chat Tutor ----------------
with tabs[1]:
    st.header("Chat With Your English Tutor")

    user_msg = st.text_input("Ask something (grammar, usage, pronunciation, etc.)")
    if st.button("Send", key="send_btn"):
        if user_msg.strip():
            bot_reply = st.session_state.tutorbot.chat(user_msg.strip())
            st.session_state.chat_history.append(("You", user_msg.strip()))
            st.session_state.chat_history.append(("Tutor", bot_reply))

    # show chat history
    for speaker, msg in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Tutor:** {msg}")

# ---------------- TAB 3: Grammar Checker ----------------
with tabs[2]:
    st.header("Grammar Correction (T5 fine-tuned)")

    input_text = st.text_area("Write a sentence or paragraph in English:")
    if st.button("Correct Grammar"):
        if input_text.strip():
            corrected = correct_sentence(input_text.strip())
            diffview = highlight_corrections(input_text.strip(), corrected)
            st.subheader("Corrected Output")
            st.write(corrected)
            st.subheader("Changes (Removed [word], Added (word))")
            st.write(diffview)

# ---------------- TAB 4: Speak & Practice ----------------
with tabs[3]:
    st.header("Speech Practice (Whisper STT + gTTS TTS)")

    st.subheader("1. Upload your speaking audio (WAV/MP3)")
    audio_file = st.file_uploader("Upload recording", type=["wav","mp3","m4a"])
    lang_code = st.selectbox("Spoken language in the audio?", ["en","hi","mr"])
    if st.button("Transcribe Audio"):
        if audio_file is not None:
            audio_bytes = audio_file.read()
            transcript = transcribe_file(audio_bytes, language_code=lang_code)
            st.success("Transcription:")
            st.write(transcript)

            st.session_state.last_transcript = transcript
        else:
            st.warning("Please upload audio first.")

    st.markdown("---")
    st.subheader("2. Improve what you said")
    if "last_transcript" in st.session_state:
        if st.button("Fix my grammar from last transcript"):
            corrected = correct_sentence(st.session_state.last_transcript)
            st.write("Corrected version:")
            st.write(corrected)

            # TTS of corrected sentence
            mp3_path = synthesize_tts(corrected, lang="en")
            audio_file = open(mp3_path, "rb")
            st.audio(audio_file.read(), format="audio/mp3")
            audio_file.close()

# ---------------- TAB 5: Mood / Engagement ----------------
with tabs[4]:
    st.header("Emotion Check (for adaptive tone)")

    st.write("Upload a selfie or webcam snapshot (face visible).")
    img_file = st.file_uploader("Image", type=["jpg","jpeg","png"])

    if st.button("Analyze Emotion"):
        if img_file is not None:
            file_bytes = np.frombuffer(img_file.read(), np.uint8)
            bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            label = predict_emotion_from_frame(bgr)
            if label is None:
                st.warning("No face detected.")
            else:
                st.success(f"Detected emotion: {label}")
                if label in ["sad","fear","angry","disgust"]:
                    st.info("Coach suggestion: We'll go slower and review basics calmly.")
                elif label in ["happy","surprise","neutral"]:
                    st.info("Coach suggestion: You're doing great. We can try a slightly harder exercise next!")
        else:
            st.warning("Please upload an image first.")
