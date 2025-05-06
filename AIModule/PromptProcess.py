import os
import sys
import json
import logging
from DataCache.NovelInfoModule import NovelInfoModule
from ConfigModule.ConfigManager import config
from AIModule.LLMmodule import LLMmodule
from AIModule.RAGmodule import RAGProcessor


class PromptProcess:
    def __init__(self, novelInfoModule: NovelInfoModule):
        self.isRagOn = config.get("enable_rag", False)
        if self.isRagOn:
            print("RAG on")
            self.RAGModule = RAGProcessor()
        else:
            print("RAG off")
            self.LLMmodule = LLMmodule()
        self.novelInfoModule = novelInfoModule

        self.promptCommon = "背景：你是一个修仙小说创作者大师，熟悉当前几乎所有热门网文的创作灵感和创作方法。" \
                            "你善于设置悬念，设计的故事剧情引人入胜。如果没有特别说明的话，你的输出语言应该是中文。" \
                            "你只需要完成任务，不会说出什么【希望这个大纲能帮助你创作出一部精彩的修仙小说！】等这种结尾祝福式的对话。" \
                            "现在你的任务是："
        self.resultPlot = ""
        self.resultTitle = ""

    # 推理性的模型可能有<think></think>这样的标记，需要整个剔除掉，不需要思考的部分
    def StripFromResponse(self, response: str):
        if "</think>" in response and "<think>" in response:
            return response.split("</think>")[-1].strip()
        else:
            return response

    def GeneratePrompt(self) -> str:
        prompt = self.promptCommon
        prompt += "你将按照以下信息，完成小说:\n"
        if self.novelInfoModule.novelName != "":
            prompt += "小说的名字: " + self.novelInfoModule.novelName + "\n"
        prompt += "小说的主题: " + self.novelInfoModule.novelTheme + "\n"
        prompt += "小说主人公的名字: " + self.novelInfoModule.novelMainChar
    
    def GenerateTitleByTheme(self, novelTheme) -> str:
        prompt = self.promptCommon
        if novelTheme != "":
            prompt += "按照以下提供的小说主题信息，生成一个符合主题的小说名字。（请注意，只需要小说名字，不需要额外的信息，也无需提供推理解释）\n"
            prompt += "小说的主题: " + novelTheme + "\n"
        else:
            prompt += "请你随机生成一个小说的名字。（请注意，只需要小说名字，不需要额外的信息，也无需提供推理解释）\n"
        if self.isRagOn:
            result = self.RAGModule.GenerateWithOllama(prompt)
        else:
            result = self.LLMmodule.GenerateWithOllama(prompt)
        result = self.StripFromResponse(result)
        return result

    def GenerateFullPlot(self, filepath) -> str:
        try:
            # 读取JSON文件
            with open(filepath, 'r', encoding='utf-8') as f:
                novel_data = json.load(f)
            
            # 校验必要字段
            required_fields = ['theme', 'title', 'protagonist', 'background', 'chapters']
            if not all(key in novel_data for key in required_fields):
                errorInfo = f"JSON文件缺少必要字段，缺失字段：{set(required_fields) - set(novel_data.keys())}"
                print(errorInfo)
                logging.error(errorInfo)
                return ""

            # 构建提示词
            prompt = f"{self.promptCommon}\n"
            prompt += "基于以下完整设定生成小说的大纲：\n"
            prompt += f"作品名称：《{novel_data['title']}》\n"
            prompt += f"核心主题：{novel_data['theme']}\n"
            prompt += f"主角设定：{novel_data['protagonist']}（{novel_data['background']}）\n"
            prompt += "生成要求：\n"
            prompt += "1. 构建宏大而独特的世界：创造一个具有独特地理、历史、文化和修炼体系的世界，让读者能够沉浸在其中。" \
                      "设定清晰的修炼体系：明确修炼的阶段、方法、资源和限制，确保体系逻辑自洽，如角色能力提升的境界划分。" \
                      "融入多元元素：结合神话、传说、历史或现代元素，增加世界的丰富性和吸引力。" \
                      "考虑世界规则：设定世界的运行规则，如因果报应、天道法则等(包括不局限于列举的几种)，为故事发展提供基础。\n"
            prompt += "2. 主角设定：主角应有明确的目标和动机，性格鲜明，具备成长空间和独特的技能或天赋。" \
                      "配角塑造：配角应有各自的性格、背景和动机，避免脸谱化，使他们与主角产生互动和冲突。" \
                      "反派设计：反派应有合理的动机和背景，避免单纯为了作恶而存在，使正邪冲突更具深度。" \
                      "角色成长：角色应随着故事发展而成长和变化，体现其在修仙道路上的心境提升和实力增长。\n"
            prompt += "3. 语言流畅：使用流畅、生动的语言，避免过于晦涩或冗长的句子，使读者能够轻松阅读。" \
                      "描写细腻：对场景、人物和情感进行细腻的描写，增强故事的画面感和代入感。" \
                      "叙事结构：选择合适的叙事结构，如线性叙事、多线叙事等，使故事更具层次感和吸引力。" \
                      "幽默与搞怪：根据小说风格，适当加入幽默、搞怪或现代元素，增加故事的趣味性\n"
            prompt += "4. 创新设定：在传统修仙元素的基础上，加入独特的创新设定，如特殊的修炼方法、新颖的法宝等。" \
                      "反映现实：通过故事反映人性的复杂，增加故事的深度和内涵。" \
                      "文化传承：融入中国传统文化元素，如道家思想、神话传说等，使小说更具文化底蕴。\n"

            print(prompt)

            if self.isRagOn:
                resultPlot = self.RAGModule.GenerateWithOllama(prompt)
            else:
                resultPlot = self.LLMmodule.GenerateWithOllama(prompt)
            resultPlot = self.StripFromResponse(resultPlot)
            print(resultPlot)
            self.resultPlot = resultPlot

            return resultPlot

        except FileNotFoundError:
            logging.error(f"配置文件不存在：{filepath}")
            return f"错误：找不到配置文件 {os.path.basename(filepath)}"
        except json.JSONDecodeError:
            logging.error("JSON文件解析失败")
            return "错误：配置文件格式不正确"
        except Exception as e:
            logging.error(f"生成大纲时发生异常：{str(e)}")
            return f"生成失败：{str(e)}"
    
    def GenerateSpecificTitles(self, filepath, startIndex, endIndex):
        try:
            # 读取JSON文件
            with open(filepath, 'r', encoding='utf-8') as f:
                novel_data = json.load(f)
            
            # 校验必要字段
            required_fields = ['theme', 'title', 'protagonist', 'background', 'chapters']
            if not all(key in novel_data for key in required_fields):
                errorInfo = f"JSON文件缺少必要字段，缺失字段：{set(required_fields) - set(novel_data.keys())}"
                print(errorInfo)
                logging.error(errorInfo)
                return ""
            
            prompt = f"{self.promptCommon}\n"
            prompt += f"作品名称：《{novel_data['title']}》\n"
            prompt += f"核心主题：{novel_data['theme']}\n"
            prompt += f"主角设定：{novel_data['protagonist']}（{novel_data['background']}）\n"

            if self.resultPlot != "":
                prompt += f"世界观：{self.resultPlot}\n"
            prompt += f"\n"
            prompt += f"基于以下要求生成小说第{startIndex}章到第{endIndex}章的章节标题（严格按照此要求的起始和节数章节来，不能多不能少，不可省略，不可概括，每一章的标题都需要如实输出）：\n"
            prompt += "注意：" \
                      "主线清晰：确定故事的主线，如主角的修仙之路、寻找神器、拯救世界等，确保情节围绕主线展开。" \
                      "冲突与挑战：设置丰富的冲突和挑战，如修炼瓶颈、正邪斗争、资源争夺等，推动情节发展。" \
                      "悬念与反转：在情节中设置悬念和反转，增加故事的吸引力和不确定性，如隐藏的身份、未解的秘密等。" \
                      "节奏把控：合理安排情节的节奏，避免过于拖沓或紧凑，使读者能够保持阅读兴趣\n"
            prompt += "输出要求：\n列举清晰，首先不同的剧情阶段涉及的章节数，其次每个剧情阶段包含的章节标题，标题后面用括号记录本章会发生的剧情，" \
                      "记住不要记录无聊的流水账，不要重复。每个剧情阶段都需要设置多个伏笔，并在此剧情阶段陆续解决回收伏笔。\n" \
                      "输出不需要你说任何额外的内容，按下述要求即可。（一定不要输出下述格式之外的内容）"
            prompt += "输出举例：\n第1章：《XXXXX》（XXXX,XXXXX）\n第2章：《XXXXX》（XXXX,XXXXX）\n第3章：《XXXXX》（XXXX,XXXXX）\n"

            print(prompt)

            if self.isRagOn:
                resultTitle = self.RAGModule.GenerateWithOllama(prompt)
            else:
                resultTitle = self.LLMmodule.GenerateWithOllama(prompt)
            resultTitle = self.StripFromResponse(resultTitle)
            print(resultTitle)
            self.resultTitle = resultTitle

            return resultTitle

        except FileNotFoundError:
            logging.error(f"配置文件不存在：{filepath}")
            return f"错误：找不到配置文件 {os.path.basename(filepath)}"
        except json.JSONDecodeError:
            logging.error("JSON文件解析失败")
            return "错误：配置文件格式不正确"
        except Exception as e:
            logging.error(f"生成大纲时发生异常：{str(e)}")
            return f"生成失败：{str(e)}"
    
    def GenerateChapter(self, filepath, chapterIndex):
        try:
            # 读取JSON文件
            with open(filepath, 'r', encoding='utf-8') as f:
                novel_data = json.load(f)
            
            # 校验必要字段
            required_fields = ['theme', 'title', 'protagonist', 'background', 'chapters']
            if not all(key in novel_data for key in required_fields):
                errorInfo = f"JSON文件缺少必要字段，缺失字段：{set(required_fields) - set(novel_data.keys())}"
                print(errorInfo)
                logging.error(errorInfo)
                return ""

            prompt = f"{self.promptCommon}\n"
            prompt += f"作品名称：《{novel_data['title']}》\n"
            prompt += f"核心主题：{novel_data['theme']}\n"
            prompt += f"主角设定：{novel_data['protagonist']}（{novel_data['background']}）\n"
            if self.resultPlot != "":
                prompt += f"世界观：{self.resultPlot}\n"
            prompt += "要求：\n" \
                      "语言流畅：使用流畅、生动的语言，避免过于晦涩或冗长的句子，使读者能够轻松阅读。" \
                      "描写细腻：对场景、人物和情感进行细腻的描写，增强故事的画面感和代入感。" \
                      "叙事结构：选择合适的叙事结构，如线性叙事、多线叙事等，使故事更具层次感和吸引力。" \
                      "幽默与搞怪：根据小说风格，适当加入幽默、搞怪或现代元素，增加故事的趣味性\n"
            prompt += "特别注意：\n" \
                      "一定要注意扩充细节，描写更像网文大师一样，不要用一些很草率的话来代替。一定不能存在记流水账的现象，前一段还在村里默默无闻，后面就突然拿到宝物了。" \
                      "要注意因果逻辑关系，剧情不能突兀，也不能仓促。现在是长篇小说，有充足的篇幅去说清楚每一件事情，要注意逻辑符合常理，要注意细节。"
            prompt += f"现在请根据剧情走向，后续将要发生的内容，以及上述的各种信息和要求，编写第{chapterIndex}章的内容，要求不少于4200字，请严格按照此字数限制输出。如果不够，重新生成扩写，直到满足要求。"
            
            print(prompt)

            if self.isRagOn:
                resultContent = self.RAGModule.GenerateWithOllama(prompt)
            else:
                resultContent = self.LLMmodule.GenerateWithOllama(prompt)
            resultContent = self.StripFromResponse(resultContent)
            print(resultContent)

            return resultContent

        except FileNotFoundError:
            logging.error(f"配置文件不存在：{filepath}")
            return f"错误：找不到配置文件 {os.path.basename(filepath)}"
        except json.JSONDecodeError:
            logging.error("JSON文件解析失败")
            return "错误：配置文件格式不正确"
        except Exception as e:
            logging.error(f"生成大纲时发生异常：{str(e)}")
            return f"生成失败：{str(e)}"


if __name__ == "__main__":
    promptProcess = PromptProcess()
    result = promptProcess.GeneratePrompt("诗人身处异乡的思乡之情")
    print(result)
