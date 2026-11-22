"""Native Gemini transport for the AGE-ML matched benchmark.

Uses the same client family as AGE-ML, with seed and temperature explicitly
forwarded. Backend determinism is not guaranteed. No search/reward changes.
"""
import os
import json
import urllib.request
from langchain_google_genai import ChatGoogleGenerativeAI
from .base_chat import BaseAssistantChat
from .. import bench_hooks

class AssistantChatGemini(ChatGoogleGenerativeAI, BaseAssistantChat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def assistant_chat(self, message: str) -> str:
        response = BaseAssistantChat.assistant_chat(self, message)
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            parts = []
            for block in response:
                if isinstance(block, str):
                    parts.append(block)
                elif (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                ):
                    parts.append(block["text"])
            text = "".join(parts)
            if text.strip():
                return text
        raise ValueError("Gemini response has no usable text.")

    def describe(self):
        return {"model": self.model, "provider": "gemini", "seed": self.seed}


def get_gemini_models():
    key=os.environ["OPENAI_API_KEY"]  # harness injects the configured key only here
    req=urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",headers={"x-goog-api-key":key})
    with urllib.request.urlopen(req,timeout=60) as response:
        return [m["name"].removeprefix("models/") for m in json.load(response).get("models",[])]


def create_gemini_chat(config, session_name):
    if "seed" not in ChatGoogleGenerativeAI.model_fields:
        raise RuntimeError("Gemini client does not support seed; refusing silent fallback")
    kwargs={"model":config.model,"google_api_key":os.environ["OPENAI_API_KEY"],"session_name":session_name,"seed":int(config.seed),"temperature":float(config.temperature)}
    bench_hooks.chat_created(session_name,{"provider":"gemini","model":config.model,"seed":int(config.seed),"temperature":float(config.temperature),"base_url":"https://generativelanguage.googleapis.com/v1beta","seed_support":"requested; backend determinism not guaranteed"})
    return AssistantChatGemini(**kwargs)
