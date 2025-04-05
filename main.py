import os
import sys
from flask import Flask
from flask import render_template, request, jsonify
from DataCache.NovelInfoModule import NovelInfoModule
from AIModule.PromptProcess import PromptProcess


class DaoWritingAgent:
    def __init__(self):
        self.app = Flask(__name__, template_folder='WebUI')
        self.handldProcess = NovelInfoModule()
        self.RegisterRoutes()
        self.promptProcess = PromptProcess()

    def RegisterRoutes(self):
        @self.app.route('/')
        def Home():
            return render_template('index.html', ready=True, status="就绪")

        @self.app.route('/creation_workspace')
        def CreationWorkspace():
            work_type = request.args.get('type', 'new')
            return render_template('workspace.html', 
                                work_type=work_type,
                                theme="修仙")

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


if __name__ == "__main__":
    print("start Dao writing ...")
    test = DaoWritingAgent()
    test.app.run(host='localhost', port=8888)
