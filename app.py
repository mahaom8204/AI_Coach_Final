# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av, cv2

from utils.session_state import init_state, save_game_state
from models.chatbot_service import TutorBot
from models import adaptive_engine
from models.emotion_service import predict_emotion_from_frame
from models.grammar_checker import correct_sentence, highlight_corrections
from models.speech_to_text import record_audio, transcribe_file
from models.text_to_speech_service import synthesize_tts

st.set_page_config(page_title="Adaptive English Coach", page_icon="🧠", layout="wide")
init_state(st)

# Tutor singleton
if st.session_state.tutorbot is None:
    st.session_state.tutorbot = TutorBot()

# ===== Live Emotion (auto-playing) =====
class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_emotion = None
    def transform(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        label, box = predict_emotion_from_frame(img)
        if label:
            self.last_emotion = label
            if box:
                x,y,w,h = box
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,180,0), 2)
            cv2.putText(img, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2, cv2.LINE_AA)
        return img

# Try to start automatically (user will still need to allow camera once)
ctx = webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    desired_playing_state=True,             # <-- auto-start
    media_stream_constraints={"video": True, "audio": False},
    video_transformer_factory=EmotionTransformer,
    async_processing=True,
    video_html_attrs={"autoPlay": True, "muted": True, "playsInline": True}
)

def get_live_emotion():
    if ctx and ctx.video_transformer and ctx.video_transformer.last_emotion:
        st.session_state.current_emotion = ctx.video_transformer.last_emotion
    return st.session_state.current_emotion

# ===== Helpers =====
def update_gamification(correct, total):
    gs = st.session_state.game_state
    gs["xp"] += int(correct) * 10
    if total > 0 and (correct/total) >= 0.6:
        gs["streak_days"] += 1
    # sync "You" in leaderboard
    you = next((p for p in gs["leaderboard"] if p["name"].lower()=="you"), None)
    if you: you["xp"] = gs["xp"]
    else: gs["leaderboard"].append({"name":"You","xp":gs["xp"]})
    save_game_state(gs)

def generate_quiz_now():
    info = adaptive_engine.get_topic_info(
        current_topic=st.session_state.current_topic,
        user_results=st.session_state.user_results,
        emotion=get_live_emotion()
    )
    diff = info["base_difficulty"]
    q = st.session_state.tutorbot.generate_quiz(
        topic=st.session_state.current_topic,
        difficulty_hint=diff,
        num_q=5
    )
    st.session_state.quiz_data = q
    st.session_state.quiz_answers = [None]*len(q)
    return info

def refresh_teaching_block():
    info = adaptive_engine.get_topic_info(
        current_topic=st.session_state.current_topic,
        user_results=st.session_state.user_results,
        emotion=get_live_emotion()
    )
    st.session_state.teaching_block = st.session_state.tutorbot.generate_teaching_block(
        topic=st.session_state.current_topic,
        mood=st.session_state.current_emotion,
        level_hint=info["model_level"],
    )

# ===== Sidebar =====
with st.sidebar:
    gs = st.session_state.game_state
    st.metric("Streak (days)", gs["streak_days"])
    st.metric("XP", gs["xp"])
    st.caption(f"Emotion: {get_live_emotion() or '—'}")

# ===== Tabs =====
tabs = st.tabs(["📘 Learn", "📝 Assessment", "💬 Chat", "🎙 Speak", "🛠 Grammar", "🌐 Translate"])

# --- Learn ---
with tabs[0]:
    st.subheader(f"Current Topic • {st.session_state.current_topic} (starts A1)")
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("Load/Refresh Lesson", use_container_width=True):
            refresh_teaching_block()
        st.write(st.session_state.teaching_block or "Click to load lesson content.")
    with col2:
        st.write("Live Emotion:", get_live_emotion() or "Detecting…")

# --- Assessment (auto new quiz always) ---
with tabs[1]:
    st.subheader("Adaptive Quiz (emotion-aware)")
    if st.session_state.quiz_data is None:
        generate_quiz_now()
    if st.session_state.quiz_data:
        for i, q in enumerate(st.session_state.quiz_data):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            opts = [f"{j+1}. {opt}" for j, opt in enumerate(q["options"])]
            cur = st.session_state.quiz_answers[i]
            st.session_state.quiz_answers[i] = st.radio(
                "Choose one:", opts,
                index=(opts.index(cur) if cur in opts else 0),
                key=f"q_{i}"
            )
            st.divider()

    # Submit always regenerates a new quiz (right OR wrong)
    if st.button("Submit & Next Quiz", use_container_width=True):
        total = len(st.session_state.quiz_data or [])
        correct = 0
        res = []
        for i, q in enumerate(st.session_state.quiz_data or []):
            sel = st.session_state.quiz_answers[i]
            idx = int(sel.split(".")[0]) - 1
            ok = 1 if idx == q["answer_index"] else 0
            res.append(ok); correct += ok
        st.session_state.user_results = res
        update_gamification(correct, total)
        info = generate_quiz_now()      # ← ALWAYS regenerate
        refresh_teaching_block()        # refresh learn content
        st.success(f"Round score: {correct}/{total}")
        st.info(info["coach_message"])

# --- Chat ---
with tabs[2]:
    msg = st.text_input("Ask your tutor:")
    if st.button("Send"):
        if msg.strip():
            ans = st.session_state.tutorbot.chat(msg.strip())
            st.write("**Tutor:**", ans)

# --- Speak ---
with tabs[3]:
    st.subheader("Speak & Practice")
    secs = st.slider("Record seconds:", 3, 15, 5)
    lang_in = st.selectbox("You will speak in:", ["en","hi","mr"], index=0)
    if st.button("Record Now"):
        wav = record_audio(secs)
        text = transcribe_file(wav, language_code=lang_in)
        st.write("Transcript:", text or "—")
        if text.strip():
            corr = correct_sentence(text)
            st.write("Corrected:", corr)
            tts_lang = st.selectbox("Listen in:", ["en","hi","mr"], index=0, key="tts1")
            if corr.strip():
                p = synthesize_tts(corr, lang=tts_lang)
                with open(p, "rb") as f: st.audio(f.read(), format="audio/mp3")
        else:
            st.warning("No speech detected. Try again closer to the mic.")

# --- Grammar ---
with tabs[4]:
    txt = st.text_area("Enter English text:")
    if st.button("Correct Grammar"):
        if txt.strip():
            corr = correct_sentence(txt.strip())
            st.subheader("Corrected")
            st.write(corr)
            st.subheader("Changes")
            st.write(highlight_corrections(txt.strip(), corr))
        else:
            st.warning("Please type something.")

# --- Translate ---
with tabs[5]:
    src = st.selectbox("From", ["English","Hindi","Marathi"], index=0)
    tgt = st.selectbox("To",   ["English","Hindi","Marathi"], index=1)
    ttxt = st.text_area("Text:")
    if st.button("Translate"):
        tr = st.session_state.tutorbot.translate(ttxt, src, tgt)
        st.subheader("Translation")
        st.write(tr)
        tts_lang2 = st.selectbox("Speak result in:", ["en","hi","mr"], index=0, key="tts2")
        if tr.strip():
            p2 = synthesize_tts(tr, lang=tts_lang2)
            with open(p2, "rb") as f: st.audio(f.read(), format="audio/mp3")
