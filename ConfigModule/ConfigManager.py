import json
import os
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        config_path = Path(__file__).parent.parent / "config.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = self._get_default_config()
            print(f"⚠️ 配置文件未找到，已使用默认配置")
        except json.JSONDecodeError:
            self.config = self._get_default_config()
            print(f"⚠️ 配置文件格式错误，已使用默认配置")
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "enable_rag": True,
            "llm_model": "gemma3:27b",
            "temperature": 0.7,
            "max_tokens": 4096,
            "debug_mode": False
        }
    
    def get(self, key: str, default=None) -> Any:
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def reload(self):
        self._load_config()

# 单例实例
config = ConfigManager()