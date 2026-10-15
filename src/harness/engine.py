import os
import json
from typing import Optional
import uuid
from groq import Groq  # hoặc OpenAI
from .registry import PluginRegistry
from .session import SessionManager
from src.agentic.prompts import SUPERAGENT_SYSTEM_PROMPT


MAX_STEP = 5

class HarnessEngine:
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.session_manager = SessionManager()

    def loop(
        self,
        thread_id: str,
        user_id: str,
        message: str,
    ):
        turn_id = f"turn_{uuid.uuid4().hex[:8]}"
        self.session_manager.get_or_create_session(thread_id, user_id)
        self.session_manager.append_event(
            thread_id,
            turn_id,
            1,
            "user/message",
            {"content": message},
        )

        for step in range(1, MAX_STEP + 1):
            history = self.session_manager.get_trajectory(thread_id)

            messages_for_llm = [
                {"role": "system", "content": SUPERAGENT_SYSTEM_PROMPT}
            ] + history

            # Gọi LLM cùng danh sách schema các Tools
            resp = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages_for_llm,
                tools=PluginRegistry.get_tools_for_llm(),
                max_tokens=1500,
                temperature=0.2,
            )
            
            choice = resp.choices[0].message

            if choice.tool_calls:
                tool_calls_payload = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in choice.tool_calls
                ]

                # Store tool event choice assistant/tool_call
                self.session_manager.append_event(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    step_index=step,
                    event_type="assistant/tool_call",
                    payload={"tool_calls": tool_calls_payload}
                )

                for tc in choice.tool_calls:
                    func_name = tc.function.name
                    args = json.loads(tc.function.arguments)

                    output = PluginRegistry.execute(func_name, **args)

                    self.session_manager.append_event(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        step_index=step,
                        event_type="tool/result",
                        payload={
                            "tool_call_id": tc.id,
                            "function_name": func_name,
                            "arguments": args,
                            "output": output
                        }
                    )

            else:
                self.session_manager.append_event(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    step_index=step,
                    event_type="assistant/message",
                    payload={"content": choice.content}
                )
                return choice.content
        return "Xin lỗi, tác vụ vượt quá số bước suy luận tối đa cho phép."
                

