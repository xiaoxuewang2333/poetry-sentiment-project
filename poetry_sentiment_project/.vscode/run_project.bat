@echo off
call conda activate tf2
cd /d C:\poetry_sentiment_project_complete
set PYTHONPATH=.
python -m poetry_sentiment_project.app.main
pause
