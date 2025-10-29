# models/chatbot_service.py
import os
import re
import json
from dotenv import load_dotenv
from google import genai  # google-genai client

class TutorBot:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("MY_API_KEY")
        self.client = genai.Client(api_key=api_key)

        # Your safety/style/system prompt
        self.sys_prompt = (
            "You are an English learning tutor. "
            "You must ONLY help with English language learning: grammar, vocabulary, usage, pronunciation, fluency. "
            "You should respond in plain English, no special symbols like *, **, #, @, etc. "
            "Use numbered or roman numbered lists, not bullet points. "
            "Be supportive, encouraging, and clear."
        )

        self.history = []  # list of {"user":..., "bot":...}

    def chat(self, user_msg: str) -> str:
        # Build conversation context from system + history
        contents = [self.sys_prompt]
        for turn in self.history:
            contents.append(turn["user"])
            contents.append(turn["bot"])
        contents.append(user_msg)

        resp = self.client.models.generate_content(
            model="gemma-3-27b-it",
            contents=contents,
        )
        answer = resp.text

        # update history
        self.history.append({"user": user_msg, "bot": answer})
        return answer

    def generate_quiz(self, topic: str, num_q: int = 5):
        """
        Ask LLM for MCQs. We force strict JSON for parsing.
        Output shape:
        [
          {"question": "...",
           "options": ["A","B","C","D"],
           "answer_index": 2},
          ...
        ]
        """
        prompt = f"""
        Create {num_q} multiple choice questions to test English skills on topic: {topic}.
        Difficulty: mixed beginner to intermediate.
        For each question include:
        1. "question": the question text
        2. "options": an array of 4 answer choices (strings)
        3. "answer_index": index (0-3) of the correct answer.
        Respond ONLY as valid JSON list, no commentary, no markdown, no extra text.
        """

        resp = self.client.models.generate_content(
            model="gemma-3-27b-it",
            contents=[self.sys_prompt, prompt],
        )
        raw = resp.text.strip()

        # Cleanup for ```json ... ``` style responses
        raw = re.sub(r"^```json", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()

        try:
            quiz = json.loads(raw)
        except json.JSONDecodeError:
            quiz = []

        cleaned = []
        for q in quiz:
            if (
                isinstance(q, dict)
                and "question" in q
                and "options" in q
                and "answer_index" in q
                and isinstance(q["options"], list)
                and len(q["options"]) == 4
            ):
                cleaned.append(q)

        return cleaned[:num_q]
