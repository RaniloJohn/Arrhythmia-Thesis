@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   Arrhythmia Thesis Dual-Agent Pipeline
echo ===================================================
set /p USER_PROMPT="Enter task for Claude & Antigravity: "

echo.
echo [1/2] Claude (Sonnet) is generating implementation steps...
claude -p "%USER_PROMPT%. Output a concise checklist in PLAN.md."

echo.
echo [2/2] Antigravity is executing the tasks...
agy "Read PLAN.md and implement the requested changes directly in the workspace."

echo.
echo ===================================================
echo   Execution Finished. Review PLAN.md and git diff.
echo ===================================================
pause