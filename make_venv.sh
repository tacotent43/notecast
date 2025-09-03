# python -m venv .venv

# pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

conda init -c speech-to-conspect

conda install pytorch torchvision torchaudio -c pytorch -c nvidia
conda install accelerate transformers ffmpeg -c conda-forge 