# app.py
import io
import base64
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from pydub import AudioSegment

from utils.session_state import init_state
from models.chatbot_service import TutorBot
from models import adaptive_engine
from models.emotion_service import predict_emotion_from_frame
from models.grammar_checker import correct_sentence, highlight_corrections
from models.speech_to_text import transcribe_file
from models.text_to_speech_service import synthesize_tts


st.set_page_config(
    page_title="Adaptive English Coach",
    layout="wide",
    page_icon="🧠",
)

init_state(st)

# lazy init chatbot instance
if st.session_state.tutorbot is None:
    st.session_state.tutorbot = TutorBot()

# helper function: browser mic recorder widget
def mic_recorder_ui():
    html_code = """
    <script>
    let chunks = [];
    let mediaRecorder;
    let isRecording = false;

    async function startRec(){
        chunks = [];
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
        mediaRecorder.onstop = e => {
            const blob = new Blob(chunks, { type: 'audio/webm' });
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64data = reader.result.split(',')[1];
                const pyInput = document.getElementById("audio_data");
                pyInput.value = base64data;
            };
            reader.readAsDataURL(blob);
        };
        mediaRecorder.start();
        isRecording = true;
        document.getElementById("status").innerText = "Recording...";
    }

    function stopRec(){
        if (isRecording){
            mediaRecorder.stop();
            isRecording = false;
            document.getElementById("status").innerText = "Stopped. Click 'Process Speech' below.";
        }
    }
    </script>

    <div>
      <p id="status">Idle</p>
      <button onclick="startRec()">Start Recording</button>
      <button onclick="stopRec()">Stop Recording</button>
      <input type="hidden" id="audio_data" name="audio_data" />
    </div>
    """
    components.html(html_code, height=200)

# tabs
tabs = st.tabs([
    "Assessment",
    "Learning Path",
    "Chat Tutor",
    "Speak & Practice",
    "Grammar Checker",
])

########################################
# TAB 1: ASSESSMENT (LLM quiz + emotion + adaptivity)
########################################
with tabs[0]:
    st.header("Quick Assessment")

    col_left, col_right = st.columns([2,1])

    with col_left:
        st.subheader("1. Answer these questions")

        # generate quiz if not already
        if st.session_state.quiz_data is None:
            # choose a topic prompt you want to test (can be dynamic later)
            quiz_topic = "basic English grammar, tense correctness, and sentence choice"
            st.session_state.quiz_data = st.session_state.tutorbot.generate_quiz(
                topic=quiz_topic,
                num_q=5
            )
            st.session_state.quiz_answers = [None] * len(st.session_state.quiz_data)

        # render each question
        for i, q in enumerate(st.session_state.quiz_data):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            # show options as "1. text", "2. text", etc.
            options_labels = [f"{idx+1}. {opt}" for idx, opt in enumerate(q["options"])]
            current_val = st.session_state.quiz_answers[i]
            st.session_state.quiz_answers[i] = st.radio(
                "Choose one:",
                options=options_labels,
                index=options_labels.index(current_val) if current_val in options_labels else 0,
                key=f"quiz_q_{i}"
            )

        if st.button("Submit Answers"):
            results = []
            for i, q in enumerate(st.session_state.quiz_data):
                chosen_label = st.session_state.quiz_answers[i]
                chosen_idx = int(chosen_label.split(".")[0]) - 1
                correct_flag = 1 if chosen_idx == q["answer_index"] else 0
                results.append(correct_flag)

            st.session_state.user_results = results
            score = sum(results)
            total = len(results)

            st.success(f"Your score: {score} / {total}")

    with col_right:
        st.subheader("2. Update Mood")
        st.caption("Take a quick snapshot so we can adapt pacing & tone.")

        img_data = st.camera_input("Tap 'Take Photo' to detect mood")
        if img_data is not None:
            file_bytes = np.asarray(bytearray(img_data.getvalue()), dtype=np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            emo_label = predict_emotion_from_frame(frame_bgr)
            if emo_label:
                st.session_state.current_emotion = emo_label
                st.info(f"Emotion: {emo_label}")
            else:
                st.warning("No face detected.")

        st.markdown("---")
        st.subheader("3. Personalized Recommendation")

        # Pick a focus topic to evaluate.
        # In future: choose dynamically based on what the quiz tested.
        focus_topic = "Grammar Expansion"

        info = adaptive_engine.get_topic_info(
            current_topic=focus_topic,
            user_results=st.session_state.user_results,
            emotion=st.session_state.current_emotion
        )

        st.write("Coach Plan:")
        st.json(info)

########################################
# TAB 2: LEARNING PATH (display current adaptive state)
########################################
with tabs[1]:
    st.header("Your Learning Path")

    st.write("This reflects your latest quiz performance and current engagement mood.")
    st.write("We adapt difficulty, CEFR level guess, and give targeted practice ideas.")

    focus_topic = "Grammar Expansion"

    info = adaptive_engine.get_topic_info(
        current_topic=focus_topic,
        user_results=st.session_state.user_results,
        emotion=st.session_state.current_emotion
    )

    colA, colB = st.columns(2)
    with colA:
        st.write(f"Topic: {info['topic']}")
        st.write(f"Estimated Mastery: {info['predicted_mastery']}")
        st.write(f"CEFR Guess: {info['model_level']}")
        st.write(f"Difficulty Plan: {info['base_difficulty']}")
        st.write(f"Mood right now: {info['emotion']}")
    with colB:
        st.write(f"Roadmap Level: {info['roadmap_level']}")
        st.write("Description:")
        st.write(info["description"])
        st.write("Examples / practice:")
        for ex in info["examples"]:
            st.write(f"- {ex}")

    st.markdown("---")
    st.info(info["coach_message"])

########################################
# TAB 3: CHAT TUTOR
########################################
with tabs[2]:
    st.header("Chat With Your English Tutor")

    user_msg = st.text_input("Ask about grammar, vocabulary, pronunciation, etc.")
    if st.button("Send", key="chat_send_btn"):
        if user_msg.strip():
            reply = st.session_state.tutorbot.chat(user_msg.strip())
            st.session_state.chat_history.append(("You", user_msg.strip()))
            st.session_state.chat_history.append(("Tutor", reply))

    for speaker, msg in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Tutor:** {msg}")

########################################
# TAB 4: SPEAK & PRACTICE
########################################
with tabs[3]:
    st.header("Speak & Practice (Mic → Whisper → Correction → TTS)")
    st.write("1. Record yourself speaking in English.")
    mic_recorder_ui()

    st.write("2. Paste recorded audio data below (auto-filled after Stop Recording).")
    audio_b64 = st.text_area(
        "Hidden audio base64 from browser recorder",
        value="",
        help="After you click Stop Recording, the recorder script fills this "
             "hidden field in the DOM, but Streamlit can't read it automatically. "
             "For now, you can manually paste from the browser console if needed."
    )

    lang_code = st.selectbox("Your spoken language?", ["en", "hi", "mr"], index=0)

    if st.button("Process Speech"):
        if audio_b64.strip():
            # Convert base64 webm audio -> wav bytes using pydub
            raw_bytes = base64.b64decode(audio_b64.strip())
            audio_segment = AudioSegment.from_file(io.BytesIO(raw_bytes), format="webm")
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_bytes = wav_io.getvalue()

            transcript = transcribe_file(wav_bytes, language_code=lang_code)
            st.session_state.last_transcript = transcript

            st.success("Transcript:")
            st.write(transcript)

            corrected = correct_sentence(transcript)
            st.write("Corrected version:")
            st.write(corrected)

            # TTS the corrected sentence
            mp3_path = synthesize_tts(corrected, lang="en")
            with open(mp3_path, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
        else:
            st.warning("No audio captured yet. Please record first.")

########################################
# TAB 5: GRAMMAR CHECKER
########################################
with tabs[4]:
    st.header("Grammar Checker (T5 fine-tuned)")

    text_in = st.text_area("Write a sentence / paragraph in English:")
    if st.button("Correct Grammar"):
        if text_in.strip():
            corrected = correct_sentence(text_in.strip())
            diff_view = highlight_corrections(text_in.strip(), corrected)

            st.subheader("Corrected Output")
            st.write(corrected)

            st.subheader("Changes (Removed [word], Added (word))")
            st.write(diff_view)
        else:
            st.warning("Please type something.")
