import asyncio
from datetime import datetime, timezone
import json
import time
import unittest
from unittest.mock import patch

from app.agent.state import AgentContext, HumanReply
from app.pc_agent import llm_planner
from app.pc_agent.controller import run_task_with_approval


class PCApprovalTests(unittest.IsolatedAsyncioTestCase):
    async def _await_pending(self, agent_context: AgentContext, *, category: str, timeout: float = 2.0):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            pending = agent_context.pending_request
            if pending and pending.category == category:
                return pending
            await asyncio.sleep(0.01)
        raise AssertionError(f"Pending request ({category}) not created.")

    async def test_high_risk_approval_flow_accepts_run(self) -> None:
        agent_context = AgentContext()
        agent_context.start()
        context = {
            "steps": [
                {
                    "name": "run_executable",
                    "description": "Run test app",
                    "risk": "HIGH",
                    "reversible": False,
                    "payload": {"path": "C:\\\\test.exe"},
                }
            ]
        }
        with patch("app.pc_agent.controller.execute_action", return_value=None):
            task = asyncio.create_task(
                run_task_with_approval(
                    agent_context=agent_context,
                    telegram_app=None,
                    chat_id=None,
                    task="Run test app",
                    context=context,
                    user_id="test-user",
                    llm_client=None,
                )
            )
            pending = await self._await_pending(agent_context, category="approval")
            approval_id = pending.question_id
            await agent_context.human_reply_queue.put(
                HumanReply(
                    question_id=approval_id,
                    text="YES",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
            await self._await_pending(agent_context, category="approval_high_risk")
            await agent_context.human_reply_queue.put(
                HumanReply(
                    question_id=approval_id,
                    text="RUN",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
            result = await asyncio.wait_for(task, timeout=2)
            self.assertTrue(result.approved)
            self.assertTrue(result.completed)


class PCPlannerTests(unittest.TestCase):
    def test_parse_steps_supports_scroll_drag_wait(self) -> None:
        payload = {
            "steps": [
                {"name": "scroll", "description": "Scroll down", "payload": {"direction": "down"}},
                {
                    "name": "drag",
                    "description": "Drag slider",
                    "payload": {"start_x": 10, "start_y": 20, "end_x": 30, "end_y": 40},
                },
                {"name": "wait", "description": "Wait", "payload": {"seconds": 1.5}},
            ]
        }
        steps = llm_planner._parse_steps(json.dumps(payload))
        self.assertIsNotNone(steps)
        names = [step["name"] for step in steps]
        self.assertEqual(names, ["scroll", "drag", "wait"])

    def test_parse_steps_supports_form_upload(self) -> None:
        payload = {
            "steps": [
                {
                    "name": "fill_form",
                    "description": "Fill application form",
                    "payload": {"fields": [{"label": "Email", "value": "me@example.com"}]},
                },
                {
                    "name": "upload_file",
                    "description": "Upload CV",
                    "payload": {"selector": "input[type=file]", "path": "C:\\\\cv.pdf"},
                },
            ]
        }
        steps = llm_planner._parse_steps(json.dumps(payload))
        self.assertIsNotNone(steps)
        names = [step["name"] for step in steps]
        self.assertEqual(names, ["fill_form", "upload_file"])


if __name__ == "__main__":
    unittest.main()
