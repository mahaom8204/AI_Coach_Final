# models/chatbot_service.py
import os, re, json
from dotenv import load_dotenv
from google import genai

class TutorBot:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("MY_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.sys = (
            "You are an English learning tutor. "
            "Use plain text. When generating a quiz, output ONLY valid JSON."
        )
        self.history = []

    def _gen(self, contents):
        resp = self.client.models.generate_content(
            model="gemma-3-27b-it",
            contents=contents,
        )
        return resp.text.strip()

    def chat(self, msg: str) -> str:
        convo = [self.sys]
        for t in self.history:
            convo += [t["user"], t["bot"]]
        convo.append(msg)
        ans = self._gen(convo)
        self.history.append({"user": msg, "bot": ans})
        return ans

    def generate_quiz(self, topic: str, difficulty_hint: str, num_q: int = 5):
        prompt = f"""
Create {num_q} MCQ questions for topic "{topic}".
Style: {difficulty_hint}.
Return ONLY a JSON array where each item has:
- "question": string
- "options": array of 4 strings
- "answer_index": integer 0..3
No commentary.
"""
        raw = self._gen([self.sys, prompt])
        raw = re.sub(r"^```json", "", raw, flags=re.I).strip()
        raw = re.sub(r"```$", "", raw).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        out = []
        for q in data:
            if isinstance(q, dict) and "question" in q and "options" in q and "answer_index" in q:
                if isinstance(q["options"], list) and len(q["options"]) == 4:
                    out.append(q)
        return out[:num_q]

    def generate_teaching_block(self, topic: str, mood: str | None, level_hint: str):
        mood_line = f"Learner emotion: {mood}." if mood else "Learner emotion: unknown."
        prompt = f"""
Topic: {topic}
Level hint: {level_hint}
{mood_line}
Provide:
1. Simple definition
2. Usage steps (numbered)
3. 3 short example sentences
4. A tiny practice exercise
Plain text only.
"""
        return self._gen([self.sys, prompt])

    def translate(self, text: str, src_lang: str, tgt_lang: str):
        prompt = f"Translate from {src_lang} to {tgt_lang}. Only the translation:\n{text}"
        return self._gen([self.sys, prompt])
