# utils/session_state.py
import json, os
GAME_STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "game_state.json")

def load_game_state():
    with open(GAME_STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_game_state(state):
    with open(GAME_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def init_state(st):
    if "game_state" not in st.session_state:
        st.session_state.game_state = load_game_state()
    if "tutorbot" not in st.session_state:
        st.session_state.tutorbot = None
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = None
    if "user_results" not in st.session_state:
        st.session_state.user_results = []
    if "current_emotion" not in st.session_state:
        st.session_state.current_emotion = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # Start from basic A1 topic on first run
    if "current_topic" not in st.session_state:
        # Use a guaranteed A1 item if present in your roadmap; else fallback text
        st.session_state.current_topic = st.session_state.game_state.get("current_topic", "A1: Greetings and Introductions")
    if "teaching_block" not in st.session_state:
        st.session_state.teaching_block = None
