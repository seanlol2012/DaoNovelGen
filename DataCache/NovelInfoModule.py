import json
from datetime import datetime
from flask import request, jsonify
from pathlib import Path


class NovelInfoModule:
    def __init__(self):
        self.novelName = ""
        self.novelTheme = ""
        self.novelMainCharacterName = ""
        self.novelCharacterList = []
        self.novelChapters = 0

    def GatherUserInput(self):
        try:
            # 获取前端数据
            data = request.get_json()
            
            # 数据校验
            required_fields = ['theme', 'title', 'protagonist', 'background', 'chapters']
            if not all(field in data for field in required_fields):
                return jsonify({
                    "success": False,
                    "error": "params missing"
                }), 400
            
            self.novelTheme = data['theme']
            self.novelName = data['title']
            self.novelMainCharacterName = data['protagonist']
            self.novelCharacterList = data['background']
            self.novelChapters = data['chapters']
            print(self.novelTheme, self.novelName, self.novelMainCharacterName, self.novelCharacterList, self.novelChapters)

            # 创建存储目录
            cache_dir = Path("datacache/novels")
            cache_dir.mkdir(exist_ok=True)
            
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"novel_{timestamp}.json"
            filepath = cache_dir / filename
            
            # 保存原始数据
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 这里可以添加后续处理逻辑
            # ...
            
            return jsonify({
                "success": True,
                "message": "saved successfully",
                "filepath": str(filepath)
            })
        
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"exception: {str(e)}"
            }), 500
