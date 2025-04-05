@echo off
REM 启动 Python 脚本
start python main.py

REM 打开浏览器访问 Web 界面
timeout /t 5 >nul 2>&1
start http://localhost:8888
