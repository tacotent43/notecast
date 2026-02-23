import tkinter as tk
import torch

class State:
    def __init__(self, root):
        # transcription
        self.model = tk.StringVar(root, "openai/whisper-large-v3-turbo")
        self.batch = tk.StringVar(root, "32")
        self.chunk = tk.StringVar(root, "30")
        self.dtype = tk.StringVar(root, "torch.float16")
        self.language = tk.StringVar(root, "ru")

        # llm
        self.api_key = tk.StringVar(root, "")
        self.api_model = tk.StringVar(root, "")
        self.base_url = tk.StringVar(root, "")
        self.conspect_lang = tk.StringVar(root, "Russian")

        # flags
        self.create_conspect = tk.BooleanVar(root, False)
        self.remove_transcription = tk.BooleanVar(root, False)

        # files
        self.input_file = tk.StringVar(root)
        self.output_file = tk.StringVar(root)

        # device
        devices = []
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        devices.append("cpu")

        self.device_opts = devices
        self.device = tk.StringVar(root, devices[0])