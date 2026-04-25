@echo off
echo Installing PyTorch with CUDA 12.4...
venv\Scripts\pip.exe install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo Installing fine-tuning dependencies...
venv\Scripts\pip.exe install transformers peft bitsandbytes accelerate trl qwen-vl-utils datasets Pillow anthropic

echo Done! Activate venv with: venv\Scripts\activate
