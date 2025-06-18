"""Simple model selection based on goal, prompt and optional logs."""

import os
import yaml
from typing import Optional, Tuple

class ModelSelector:
    """Selects an OpenAI GPT model based on keywords."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.model_map = {
            "analysis": "gpt-4",
            "code": "gpt-4",
            "default": "gpt-3.5-turbo"
        }
        self.assistant_map = {}
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

    def _load_config(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.model_map.update(data.get("model_map", {}))
        self.assistant_map.update(data.get("assistant_map", {}))

    def select(self, goal: str, prompt: str, logs: str = "") -> Tuple[str, Optional[str]]:
        """Return model name and optional assistant id."""
        text = f"{goal} {prompt} {logs}".lower()
        for key, model in self.model_map.items():
            if key != "default" and key in text:
                return model, self.assistant_map.get(model)
        model = self.model_map.get("default", "gpt-3.5-turbo")
        return model, self.assistant_map.get(model)
