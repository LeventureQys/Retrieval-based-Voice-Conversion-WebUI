@echo off
call conda activate rvc
python infer-web.py --port 7897
pause
