import os
import sys
import requests
import json
from typing import Dict, Generator


class LLMmodule:
    def __init__(self, llmModel="gemma3:27b", maxTokens=4096, temperature=0.7):
        self.ollamaBaseUrl = "http://localhost:11434"
        # 默认模型名
        self.llmModel = llmModel
        self.maxTokens = maxTokens
        self.temperature = temperature
    
    def GenerateWithOllama(self, prompt: str, stream: bool = False) -> Dict:
        """
        调用本地Ollama生成文本
        参数说明：
        - prompt: 输入提示词
        - stream: 是否使用流式传输
        返回：包含生成结果的字典
        """
        try:
            payload = {
                "model": self.llmModel,
                "prompt": prompt,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.maxTokens
                },
                "stream": stream
            }
            
            response = requests.post(
                f"{self.ollamaBaseUrl}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=300
            )
            
            if response.status_code == 200:
                if stream:
                    # 处理流式响应
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line.decode())
                            full_response += chunk.get("response", "")
                    return {"success": True, "response": full_response}
                else:
                    result = response.json()
                    return {"success": True, "response": result.get("response", "")}
            else:
                return {
                    "success": False,
                    "error": f"API Error: {response.status_code} - {response.text}"
                }
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"连接异常: {str(e)}"}
        except json.JSONDecodeError:
            return {"success": False, "error": "响应解析失败"}
    
    def GetContentFromDict(self, data: Dict) -> str:
        if not isinstance(data, dict):
            return "Invalid response format"
        
        if not data.get("success", False):
            return f"Error: {data.get('error', 'Unknown error')}"
        
        response = data.get("response", "").strip()
        return response if response else "Empty model response"


if __name__ == "__main__":
    llm = LLMmodule()
    result = llm.GenerateWithOllama("说一首李白的诗", stream=False)
    content = llm.GetContentFromDict(result)
    print(content)
