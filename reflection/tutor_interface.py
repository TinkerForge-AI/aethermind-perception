
"""Simple tutor interface stub."""

class TutorInterface:
    def ask(self, prompt: str) -> str:
        # TODO: Integrate an LLM API
        return f"[Tutor] {prompt}"
