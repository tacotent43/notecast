# Notecast

**Notecast** is a simple and minimalistic utilite to transcribe audiofiles to text and create a conspect.

Requires nvcc v12.8

You can create environment running
```
conda env create -f environment.yml
```
or for CUDA/Nvidia:
```
conda create -n notecast -c pytorch -c nvidia pytorch torchvision torchaudio transformers python=3.12
conda install ffmpeg customtkinter -c conda-forge -c bioconda
```