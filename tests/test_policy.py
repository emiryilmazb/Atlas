import unittest

from app.agent.answer_generator import build_answer_prompt, generate_answer, is_needs_human_response
from app.agent.types import Intent, NormalizedQuestion
from app.storage.memory import MemoryContext


class AnswerFlowTests(unittest.TestCase):
    def test_needs_human_when_llm_missing(self) -> None:
        memory = MemoryContext(
            cv_profile={"summary": "", "experience": []},
            personal_facts={},
            past_answers={"entries": []},
        )
        normalized = NormalizedQuestion("Why do you want to work here?", Intent.MOTIVATION_COMPANY)
        response = generate_answer(normalized, None, memory, "ApplyWise", "Engineer")
        self.assertTrue(is_needs_human_response(response))

    def test_prompt_includes_matching_past_answers(self) -> None:
        memory = MemoryContext(
            cv_profile={"summary": "Product-minded engineer.", "experience": ["Led API redesign."]},
            personal_facts={"LOCATION": {"value": "I am in Berlin.", "source": "human"}},
            past_answers={
                "entries": [
                    {
                        "intent": Intent.MOTIVATION_COMPANY.value,
                        "question": "Why ApplyWise?",
                        "answer": "I align with the mission.",
                        "submitted_at": "2026-01-05T13:21:00Z",
                    },
                    {
                        "intent": Intent.EXPERIENCE_TECH.value,
                        "question": "Tech stack?",
                        "answer": "Python and Postgres.",
                        "submitted_at": "2026-01-05T13:21:00Z",
                    },
                ]
            },
        )
        normalized = NormalizedQuestion("Why this company?", Intent.MOTIVATION_COMPANY)
        prompt = build_answer_prompt(normalized, memory, "ApplyWise", "Engineer")
        self.assertIn("Product-minded engineer.", prompt)
        self.assertIn("Led API redesign.", prompt)
        self.assertIn("Why ApplyWise?", prompt)
        self.assertNotIn("Tech stack?", prompt)


if __name__ == "__main__":
    unittest.main()
