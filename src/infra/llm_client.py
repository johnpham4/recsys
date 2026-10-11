import os
from typing import List, Dict, Any, Optional
from openai import OpenAI

class LLMClient:
    """
    OpenAI-compatible LLM Client Adapter.
    Mặc định trỏ sang Groq Cloud (Free, siêu nhanh cho Agent).
    Có thể đổi sang Ollama, LiteLLM Gateway hoặc OpenAI chỉ bằng đổi biến môi trường.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=20.0
        )

    def is_live(self) -> bool:
        return bool(self.api_key)

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
    ):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            
        return self.client.chat.completions.create(**kwargs)
