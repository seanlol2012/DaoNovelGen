import os
import sys
import json
from pathlib import Path
from datetime import datetime
from flask import Flask
from flask import render_template, request, jsonify
from DataCache.NovelInfoModule import NovelInfoModule
from AIModule.PromptProcess import PromptProcess


class DaoWritingAgent:
    def __init__(self):
        self.app = Flask(__name__, template_folder='WebUI')
        self.handldProcess = NovelInfoModule()
        self.RegisterRoutes()
        self.promptProcess = PromptProcess(self.handldProcess)

    def RegisterRoutes(self):
        @self.app.route('/')
        def Home():
            return render_template('index.html', ready=True, status="就绪")

        @self.app.route('/creation_workspace')
        def CreationWorkspace():
            work_type = request.args.get('type', 'new')
            return render_template('workspace.html', 
                                work_type=work_type)
        
        @self.app.route('/creation_content')
        def CreationContent():
            work_type = request.args.get('type', 'new')
            return render_template('contentcreate.html', 
                                work_type=work_type)

        @self.app.route('/api/generate-novel-setting', methods=['POST'])
        def HandleGeneration():
            return self.handldProcess.GatherUserInput()

        @self.app.route('/api/summarize-novel-theme', methods=['POST'])
        def SummarizeTheme():
            print("summarize theme ...")
            try:
                # 获取请求数据
                data = request.get_json()
                if not data or 'theme' not in data:
                    return jsonify({
                        "success": False,
                        "error": "缺少必要参数: theme"
                    }), 400
                novelName = self.promptProcess.GenerateTitleByTheme(data['theme'])
                print(novelName)
                return jsonify({
                    "success": True,
                    "generatedTitle": novelName,
                    "originalTheme": data['theme']
                })
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"服务器内部错误: {str(e)}"
                }), 500
        
        # 新增路由
        @self.app.route('/api/get-novel-files')
        def GetNovelFiles():
            novel_dir = Path("DataCache/novels")
            files = []
            
            if novel_dir.exists():
                for f in novel_dir.glob("*.json"):
                    stat = f.stat()
                    files.append({
                        "name": f.stem,
                        "path": str(f),
                        "time": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M"),
                        "isLatest": False  # 将在后续处理
                    })
                
                # 标记最新文件
                if files:
                    latest = max(files, key=lambda x:x['time'])
                    latest['isLatest'] = True
            
            return jsonify({"files": files})

        @self.app.route('/api/get-novel-content')
        def GetNovelContent():
            filepath = request.args.get('path')
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return jsonify({
                        "title": data.get('title'),
                        "theme": data.get('theme'),
                        "protagonist": data.get('protagonist'),
                        "background": data.get('background'),
                        "chapters": data.get('chapters', 0)
                    })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
            
        @self.app.route('/api/generate-plot', methods=['POST'])
        def GenerateFullPlot():
            try:
                data = request.get_json()
                # 调用AI模块生成主线逻辑
                plot = self.promptProcess.GenerateFullPlot(data['filepath'])
                return jsonify({
                    "success": True,
                    "plot": plot
                })
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500

        @self.app.route('/api/generate-chapter', methods=['POST'])
        def GenerateChapterContent():
            try:
                data = request.get_json()
                # 调用AI模块生成章节内容
                content = self.promptProcess.GenerateChapter(
                    data['filepath'], 
                    data['chapter']
                )
                return jsonify({
                    "success": True,
                    "content": content
                })
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500


if __name__ == "__main__":
    print("start Dao writing ...")
    test = DaoWritingAgent()
    test.app.run(host='localhost', port=8888)
