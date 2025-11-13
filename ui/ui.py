import os
import queue
import sys
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk

import customtkinter as ctk
import torch

from transcription.audio_transcription import AudioTranscription
from transcription.device_configuration import DeviceConfiguration
from transcription.torch_checker import check_torch
from ui.tooltip import ToolTip
from ui.ui_log_handler import setup_ui_logger
from utils.requests_to_api import LLMrequest

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 725


class TranscriberApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Notecast")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # TODO: fix this stuff        
        base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        icon_path = os.path.join(base_path, "assets", "logo.ico")
        self.iconbitmap(icon_path)

        # states
        self.progress_queue = queue.Queue()
        self.transcribe_thread = None
        self.stop_flag = threading.Event()

        # USER VARIABLES
        # transcription model settings
        self.model_var = tk.StringVar(value="openai/whisper-large-v3-turbo")
        self.batch_var = tk.StringVar(value="32")
        self.chunk_var = tk.StringVar(value="30")
        self.dtype_var = tk.StringVar(value="torch.float16")
        self.transcription_lang_var = tk.StringVar(value="ru")
        # llm settings
        self.conspect_transcription_lang_var = tk.StringVar(value="Russian")
        self.api_key_var = tk.StringVar(value="")
        self.base_url_var = tk.StringVar(value="")
        self.api_model_var = tk.StringVar(value="")
        # checkboxes
        self.create_conspect = tk.BooleanVar(value=False)
        self.remove_transcription = tk.BooleanVar(value=False)

        # settings device options
        device_opts = []
        if torch.cuda.is_available():
            device_opts.append("cuda")
        if torch.backends.mps.is_available():
            device_opts.append("mps")
        device_opts.append("cpu")
        self.device_var = tk.StringVar(value=device_opts[0])
        self.device_opts = device_opts

        # input & output file variables
        self.input_file_var = tk.StringVar()
        self.output_file_var = tk.StringVar()

        # tabs packing
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(expand=True, fill="both", padx=10, pady=10)

        self.transcript_tab = self.tabview.add("Transcription")
        self.settings_tab = self.tabview.add("Settings")

        self._build_transcription_tab()
        self._build_settings_tab()

        # logger
        self.ui_logger = setup_ui_logger(self.log_box)

    ### TRANSCRIPTION TAB
    def _build_transcription_tab(self):
        # file selectors
        file_frame = ctk.CTkFrame(self.transcript_tab, corner_radius=10)
        file_frame.pack(padx=20, pady=10, fill="x")

        ctk.CTkLabel(file_frame, text="Input file:").pack(side="left", padx=5, pady=5)
        ctk.CTkEntry(file_frame, textvariable=self.input_file_var, width=400).pack(
            side="left", padx=5, pady=5, expand=True, fill="x"
        )
        ctk.CTkButton(file_frame, text="Browse", command=self._browse_input).pack(
            side="left", padx=5, pady=5
        )

        out_frame = ctk.CTkFrame(self.transcript_tab, corner_radius=10)
        out_frame.pack(padx=20, pady=5, fill="x")
        ctk.CTkLabel(out_frame, text="Output file:").pack(side="left", padx=5, pady=5)
        ctk.CTkEntry(out_frame, textvariable=self.output_file_var, width=400).pack(
            side="left", padx=5, pady=5, expand=True, fill="x"
        )
        ctk.CTkButton(out_frame, text="Browse", command=self._browse_output).pack(
            side="left", padx=5, pady=5
        )

        # controls
        ctrl_frame = ctk.CTkFrame(self.transcript_tab, corner_radius=10)
        ctrl_frame.pack(padx=20, pady=10, fill="x")

        ctk.CTkButton(ctrl_frame, text="Check Torch", command=self._check_torch).pack(
            side="left", padx=10, pady=5
        )
        self.start_button = ctk.CTkButton(
            ctrl_frame, text="Start", command=self._start_transcription
        )
        self.start_button.pack(side="left", padx=10, pady=5)

        self.stop_button = ctk.CTkButton(
            ctrl_frame, text="Stop", command=self._stop_transcription, state="disabled"
        )
        self.stop_button.pack(side="left", padx=10, pady=5)

        self.create_conspect_checkbox = ctk.CTkCheckBox(
            ctrl_frame,
            text="Create conspect",
            variable=self.create_conspect,
            onvalue=True,
            offvalue=False,
        )
        self.create_conspect_checkbox.pack(side="left", padx=10, pady=5)
        
        self.remove_transcription_checkbox = ctk.CTkCheckBox(
            ctrl_frame,
            text="Remove transcription file after",
            variable=self.remove_transcription,
            onvalue=True,
            offvalue=False,
        )
        self.remove_transcription_checkbox.pack(side="left", padx=10, pady=5)

        # TODO: add unload model button here

        # log box
        self.log_box = scrolledtext.ScrolledText(
            self.transcript_tab,
            wrap="word",
            height=20,
            font=("Consolas", 16),
        )
        self.log_box.pack(padx=20, pady=10, expand=True, fill="both")

    ### SETTINGS TAB
    def _build_settings_tab(self):
        pad = 20

        def add_setting(parent, row, col, text, tooltip, variable, values: list | None):
            frame = ctk.CTkFrame(parent)
            frame.grid(row=row, column=col, padx=pad, pady=(pad, 5), sticky="nsew")

            label = ctk.CTkLabel(frame, text=text)
            label.grid(row=0, column=0, sticky="w")
            help_icon = ctk.CTkLabel(frame, text="?", width=20, cursor="question_arrow")
            help_icon.grid(row=0, column=1, sticky="w", padx=(5, 0))
            ToolTip(help_icon, tooltip)

            if values:
                ctk.CTkOptionMenu(frame, variable=variable, values=values).grid(
                    row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0)
                )
            else:
                ctk.CTkEntry(frame, textvariable=variable).grid( 
                    row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0)
                )

            frame.grid_columnconfigure(0, weight=1)

        grid = ctk.CTkFrame(self.settings_tab)
        grid.pack(fill="both", expand=True)

        # first row
        ### Model setting
        add_setting(
            parent=grid,
            row=0,
            col=0,
            text="Model:",
            tooltip="Choose model for speed recognition",
            variable=self.model_var,
            values=[
                "openai/whisper-large-v3-turbo",
                "openai/whisper-large-v2",
                "openai/whisper-large",
                "openai/whisper-medium",
                "openai/whisper-small",
                "openai/whisper-tiny",
            ],
        )
        ### Batch size setting
        add_setting(
            parent=grid,
            row=0,
            col=1,
            text="Batch size:",
            tooltip="Chunks count for one iteration",
            variable=self.batch_var,
            values=["32", "16", "8", "4", "2"],
        )

        # second row
        ### Data type setting
        add_setting(
            parent=grid,
            row=1,
            col=0,
            text="Data type:",
            tooltip="Data type for calculations",
            variable=self.dtype_var,
            values=["torch.float16", "torch.float32", "torch.bfloat16"],
        )
        ### Chunk Length setting
        add_setting(
            parent=grid,
            row=1,
            col=1,
            text="Chunk length (s):",
            tooltip="Maximum length of processing audio fragment",
            variable=self.chunk_var,
            values=["30", "24", "20", "14", "10", "6"],
        )

        # third row
        ### Device setting
        add_setting(
            parent=grid,
            row=2,
            col=0,
            text="Device:",
            tooltip="Choose device\n- CUDA for CUDA & ROCm\n- MPS for Apple Silicon  \n- CPU for CPU-only mode",
            variable=self.device_var,
            values=self.device_opts,
        )
        ### Transcription language setting
        add_setting(
            parent=grid,
            row=2,
            col=1,
            text="Transcription language:",
            tooltip="Choose the transcription language",
            variable=self.transcription_lang_var,
            values=["ru", "en"],
        )

        # fourth row
        ### OpenAI API key setting
        add_setting(
            parent=grid,
            row=3,
            col=0,
            text="Insert OpenAI API key here:",
            tooltip="Give this programm access to LLM that would create a fully prepared conspect with AI overviews",
            variable=self.api_key_var,
            values=None,
        )
        ### Model name setting
        add_setting(
            parent=grid,
            row=3,
            col=1,
            text="Model name:",
            tooltip="Name of the model that you are going to use",
            variable=self.api_model_var,
            values=None,
        )

        # fifth row
        ### Base URL setting
        add_setting(
            parent=grid,
            row=4,
            col=0,
            text="Base URL:",
            tooltip="OpenAI base URL. Blank for None.",
            variable=self.base_url_var,
            values=None,
        )
        ### Output (conspect) language setting
        add_setting(
            parent=grid,
            row=4,
            col=1,
            text="Conspect language:",
            tooltip="Conspect language. Blank for English (default)",
            variable=self.conspect_transcription_lang_var,
            values=None,
        )

        ### Custom Prompt setting
        customPromptFrame = ctk.CTkFrame(grid)
        customPromptFrame.grid(
            row=5, column=0, columnspan=2, padx=20, pady=(20, 5), sticky="nsew"
        )

        label = ctk.CTkLabel(customPromptFrame, text="Custom Prompt:")
        label.grid(row=0, column=0, sticky="w")

        help_icon = ctk.CTkLabel(
            customPromptFrame, text="?", width=20, cursor="question_arrow"
        )
        help_icon.grid(row=0, column=1, sticky="w", padx=(5, 0))
        ToolTip(
            help_icon,
            "Enter your custom prompt for model.",
        )

        self.custom_prompt_textbox = ctk.CTkTextbox(
            customPromptFrame, width=400, height=150
        )
        self.custom_prompt_textbox.grid(
            row=1, column=0, columnspan=2, sticky="nsew", pady=(5, 0)
        )

        customPromptFrame.grid_columnconfigure(0, weight=1)
        customPromptFrame.grid_rowconfigure(1, weight=1)

        grid.grid_columnconfigure((0, 1), weight=1)

    # action buttons
    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select input audio file",
            filetypes=[
                ("Media files", "*.wav *.mp3 *.m4a *.flac *.ogg *.mp4 *.mkv *.avi"),
                ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg"),
                ("Video files", "*.mp4 *.mkv *.avi"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.input_file_var.set(path)

    def _browse_output(self):
        # TODO: add custom filename here
        directory = filedialog.askdirectory(
            title="Select output directory",
        )
        if directory:
            input_name = os.path.basename(self.input_file_var.get())
            name, _ = os.path.splitext(input_name)
            # TODO: redo output_name logic maybe?
            output_name = f"{"".join(name.split(".").pop())}.txt"
            path = os.path.join(directory, output_name)

            self.output_file_var.set(path)

    def _check_torch(self):
        check_torch(self.ui_logger)
        
        self.ui_logger.info(f"==== Transcription ====")
        self.ui_logger.info(f"Transcription model: {self.model_var.get()}")
        self.ui_logger.info(f"Batch size (in chunks): {self.batch_var.get()}")
        self.ui_logger.info(f"Chunk size (in seconds): {self.chunk_var.get()}")
        self.ui_logger.info(f"Data type: {self.dtype_var.get()}")
        self.ui_logger.info(f"Transcription language: {self.transcription_lang_var.get()}")
        self.ui_logger.info(f"=======================")
        
        debug_api_key_var: str = self.api_key_var.get() if self.api_key_var.get() else "Not set"
        debug_api_model_var: str = self.api_model_var.get() if self.api_model_var.get() else "Not set"
        debug_base_url_var: str = self.base_url_var.get() if self.base_url_var.get() else "Not set"
        debug_transcription_lang_var = self.transcription_lang_var.get() if self.transcription_lang_var.get() else "Not set"
        debug_custom_prompt_var: str = self.custom_prompt_textbox.get() if self.custom_prompt_textbox.get() else "Not set"
        
        self.ui_logger.info(f"===== LLM Setting =====")
        self.ui_logger.info(f"API key: {debug_api_key_var}")
        self.ui_logger.info(f"Model name setting: {debug_api_model_var}")
        self.ui_logger.info(f"Base URL setting: {debug_base_url_var}")
        # self.ui_logger.info(f"Custom prompt: {debug_transcription_lang_var}") # <-- issue here
        self.ui_logger.info(f"=======================")
        

    def _start_transcription(self):
        infile = self.input_file_var.get().strip()
        if not infile or not os.path.isfile(infile):
            messagebox.showerror("Error", "Please select a valid input file.")
            return

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.ui_logger.info("Starting transcription...")

        self.stop_flag.clear()
        self.transcribe_thread = threading.Thread(
            target=self._transcribe_worker, args=(infile,), daemon=True
        )
        self.transcribe_thread.start()

    def _stop_transcription(self, Audio: AudioTranscription):
        self.stop_flag.set()
        self.ui_logger.info("Stopping transcription...")
        self.ui_logger.info("Unloading model...")
        Audio._unload_model()
        self.stop_button.configure(state="disabled")

    def _transcribe_worker(self, infile: str):
        try:
            config = DeviceConfiguration(
                device=self.device_var.get(),
                model_name=self.model_var.get(),
                batch_size=int(self.batch_var.get()),
                chunk_length_s=int(self.chunk_var.get()),
                data_type=self.dtype_var.get(),
            )
            Audio = AudioTranscription(
                filepath=infile,
                device_configuration=config,
                logger=self.ui_logger,
                language=self.transcription_lang_var.get(),
            )
            transcription = Audio.transcribe_audio()
            
            outfile = self.output_file_var.get().strip()
            if not outfile:
                outfile = infile
            
            if not self.remove_transcription.get():
                outfile += ".txt"

                with open(outfile, "w", encoding="utf-8") as f:
                    f.write(transcription)
                self.ui_logger.info(f"Transcription saved to {outfile}.")
            
            if self.create_conspect.get():
                # TODO: add custom prompt ability here
                # TODO: add logging here
                # TODO: add progressbar instead of tqdm
                self.ui_logger.info(f"Starting creating conspect via {self.api_model_var.get()}...")
                with open("utils/default_prompt.txt", "r", encoding="utf-8") as f:
                    default_prompt = "\n".join(f.readlines())
                
                # if self.custom_prompt_textbox.get(): # <-- issue here
                #     prompt = transcription + "\n" + self.custom_prompt_textbox.get()
                # else:
                #     prompt = transcription + "\n" + default_prompt
                
                prompt = transcription + "\n" + default_prompt

                request = LLMrequest(
                    api_key=self.api_key_var.get(),
                    model_name=self.api_model_var.get(),
                    base_url=self.base_url_var.get(),
                )
                response = request.get_response(prompt=prompt)
                outfile += ".md"
                with open(outfile, "w", encoding="utf-8") as f:
                    f.write(response)
                
                self.ui_logger.info(f"Conspect saved to {outfile}.")

        except Exception as e:
            self.ui_logger.error(f"Error: {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
