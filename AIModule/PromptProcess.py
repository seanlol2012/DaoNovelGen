import os
import sys
from DataCache.NovelInfoModule import NovelInfoModule
from AIModule.LLMmodule import LLMmodule


class PromptProcess:
    def __init__(self):
        self.LLMmodule = LLMmodule()

    def GeneratePrompt(self, novelInfoModule: NovelInfoModule) -> str:
        prompt = ""
        prompt += "你是一个修仙小说创作者大师，现在你将按照以下信息，完成小说:\n"
        if novelInfoModule.novelName != "":
            prompt += "小说的名字: " + novelInfoModule.novelName + "\n"
        prompt += "小说的主题: " + novelInfoModule.novelTheme + "\n"
        prompt += "小说主人公的名字: " + novelInfoModule.novelMainChar
    
    def GenerateTitleByTheme(self, novelTheme) -> str:
        prompt = ""
        if novelTheme != "":
            prompt += "你是一个修仙小说创作者大师，现在你将按照以下提供的小说主题信息，生成一个符合主题得的小说名字。（请注意，只需要小说名字，不需要额外的信息，也无需提供推理解释）\n"
            prompt += "小说的主题: " + novelTheme + "\n"
        else:
            prompt += "你是一个修仙小说创作者大师，现在请你随机生成一个小说的名字。（请注意，只需要小说名字，不需要额外的信息，也无需提供推理解释）\n"
        result = self.LLMmodule.GenerateWithOllama(prompt)
        content = self.LLMmodule.GetContentFromDict(result)
        return content

if __name__ == "__main__":
    promptProcess = PromptProcess()
    result = promptProcess.GeneratePrompt("诗人身处异乡的思乡之情，非常感人")
    print(result)
