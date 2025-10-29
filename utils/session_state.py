def init_state(st):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_results" not in st.session_state:
        # store quiz correctness history for adaptive engine
        st.session_state.user_results = [1,0,1,1,0,1]  # demo defaults
    if "tutorbot" not in st.session_state:
        st.session_state.tutorbot = None
