@echo off
chcp 65001 >nul
setlocal

echo ==================================================
echo   VibeMouse 一键打包
echo ==================================================
echo.

:: ---- Locate project root (where this .bat lives) ----
cd /d "%~dp0"

:: ---- Check Python ----
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    echo   https://www.python.org/downloads/
    pause
    exit /b 1
)

:: ---- Create venv if needed ----
if not exist .venv\Scripts\python.exe (
    echo [1/3] 创建虚拟环境 ...
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
) else (
    echo [1/3] 虚拟环境已存在
)

:: ---- Install / update dependencies ----
echo [2/3] 安装依赖 ...
.venv\Scripts\pip.exe install -q -e . "pyinstaller>=6.0"
if errorlevel 1 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)

:: ---- Build ----
echo [3/3] 开始打包 ...
.venv\Scripts\python.exe scripts\build_exe.py
if errorlevel 1 (
    echo [错误] 打包失败
    pause
    exit /b 1
)

:: ---- Done ----
echo.
echo 打包完成!
echo   输出: %cd%\dist\VibeMouse.exe
echo.
pause
