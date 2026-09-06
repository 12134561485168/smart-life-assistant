@echo off
chcp 65001 >nul
setlocal

rem ============================================================
rem  万象 · 生活智能助手 —— 一键启动脚本（开源模板）
rem  用法：复制本文件为 start.bat 使用；或直接运行本文件。
rem  如需指定本机 Python，请把下面的 python 改为具体路径，例如：
rem      set PY=C:/Users/xxx/miniconda3/python.exe
rem ============================================================

set PY=python
set ROOT=%~dp0

rem 本地天气工具（端口 8000）
start cmd /k "cd /d "%ROOT%tool" && "%PY%" weather.py"

rem 本地美食工具（端口 8002）
start  cmd /k "cd /d "%ROOT%tool" && set food_port=8002 && "%PY%" food.py"

timeout /t 3 /nobreak >nul

rem 后端（端口 8080，SSE 流式）
start cmd /k "cd /d "%ROOT%backend" && "%PY%" -m uvicorn api:app --host 127.0.0.1 --port 8080"

timeout /t 3 /nobreak >nul

rem 前端（Vite dev，端口 5173）
start cmd /k "cd /d "%ROOT%front" && (if not exist node_modules (call npm install)) && npm run dev"

endlocal