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

            # 新增校验：检查title是否有效
            novel_title = data.get('title', '').strip()
            if not novel_title:
                return jsonify({
                    "success": False,
                    "error": "必须填写小说名称"
                }), 400
            
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

            def sanitize_filename(name):
                # 替换非法字符为下划线
                keep_chars = (' ', '_', '-')
                return "".join(c if c.isalnum() or c in keep_chars else '_' for c in name).strip()

            safe_title = sanitize_filename(novel_title)[:50]  # 限制最大长度
            filename = f"{safe_title}.json"  # 添加时间戳保证唯一性

            # 创建存储目录
            cache_dir = Path("datacache/novels")
            cache_dir.mkdir(exist_ok=True)
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
