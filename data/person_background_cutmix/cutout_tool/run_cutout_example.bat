@echo off
chcp 65001 >nul

REM 先进入脚本所在目录后再执行
python batch_cutout_anime.py ^
  --real-dir "C:\Users\86156\Desktop\new_inpaint\real" ^
  --fake-dir "C:\Users\86156\Desktop\new_inpaint\fake" ^
  --output-root "C:\Users\86156\Desktop\new_inpaint\cutout_test" ^
  --limit-per-class 60 ^
  --model isnet-anime

pause
