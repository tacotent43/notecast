import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import customtkinter as ctk
import torch

from transcription.audio_transcription import AudioTranscription
from transcription.device_configuration import DeviceConfiguration
from transcription.torch_checker import check_torch
from ui.ui_log_handler import setup_ui_logger
from ui.tooltip import ToolTip

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 650


class TranscriberApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Notecast")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # states
        self.progress_queue = queue.Queue()
        self.transcribe_thread = None
        self.stop_flag = threading.Event()

        # user variables
        self.model_var = tk.StringVar(value="openai/whisper-large-v2")
        self.batch_var = tk.StringVar(value="32")
        self.dtype_var = tk.StringVar(value="torch.float16")
        self.chunk_var = tk.StringVar(value="30")

        # settings device options
        device_opts = []
        if torch.cuda.is_available():
            device_opts.append("cuda")
        if torch.backends.mps.is_available():
            device_opts.append("mps")
        device_opts.append("cpu")
        self.device_var = tk.StringVar(value=device_opts[0])
        self.device_opts = device_opts

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

    # main transcription tab
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
        self.start_btn = ctk.CTkButton(
            ctrl_frame, text="Start transcription", command=self._start_transcription
        )
        self.start_btn.pack(side="left", padx=10, pady=5)

        self.stop_btn = ctk.CTkButton(
            ctrl_frame, text="Stop", command=self._stop_transcription, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=10, pady=5)
        
        # TODO: add unload model button here

        # log box
        self.log_box = scrolledtext.ScrolledText(
            self.transcript_tab,
            wrap="word",
            height=20,
            font=("Consolas", 16),
        )
        self.log_box.pack(padx=20, pady=10, expand=True, fill="both")

    def _build_settings_tab(self):
        # TODO: add tooltips here
        pad = 20

        ### Model
        ctk.CTkLabel(self.settings_tab, text="Model:").pack(
            anchor="w", padx=pad, pady=(pad, 5)
        )
        ctk.CTkOptionMenu(
            self.settings_tab,
            variable=self.model_var,
            values=[
                "openai/whisper-large-v2",
                "openai/whisper-large",
                "openai/whisper-medium",
                "openai/whisper-small",
                "openai/whisper-tiny",
            ],
        ).pack(fill="x", padx=pad, pady=5)

        ### Batch size
        ctk.CTkLabel(self.settings_tab, text="Batch size:").pack(
            anchor="w", padx=pad, pady=(pad, 5)
        )
        ctk.CTkOptionMenu(
            self.settings_tab,
            variable=self.batch_var,
            values=["32", "16", "8", "4", "2"],
        ).pack(fill="x", padx=pad, pady=5)

        ### Data type
        ctk.CTkLabel(self.settings_tab, text="Data type:").pack(
            anchor="w", padx=pad, pady=(pad, 5)
        )
        ctk.CTkOptionMenu(
            self.settings_tab,
            variable=self.dtype_var,
            values=["torch.float16", "torch.float32", "torch.bfloat16"],
        ).pack(fill="x", padx=pad, pady=5)

        ### Chunk length
        ctk.CTkLabel(self.settings_tab, text="Chunk length (s):").pack(
            anchor="w", padx=pad, pady=(pad, 5)
        )
        ctk.CTkOptionMenu(
            self.settings_tab,
            variable=self.chunk_var,
            values=["30", "25", "20", "15", "10", "5"],
        ).pack(fill="x", padx=pad, pady=5)

        ### Device
        ctk.CTkLabel(self.settings_tab, text="Device:").pack(
            anchor="w", padx=pad, pady=(pad, 5)
        )
        ctk.CTkOptionMenu(
            self.settings_tab,
            variable=self.device_var,
            values=self.device_opts,
        ).pack(fill="x", padx=pad, pady=5)

    # action buttons
    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select input audio file",
            filetypes=[
                ("Media files", "*.wav *.mp3 *.m4a *.flac *.ogg *.mp4 *.mkv *.avi"),
                ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg"),
                ("Video files", "*.mp4 *.mkv *.avi"),
                ("All files", "*.*"),
            ]
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

    def _start_transcription(self):
        infile = self.input_file_var.get().strip()
        if not infile or not os.path.isfile(infile):
            messagebox.showerror("Error", "Please select a valid input file.")
            return

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.ui_logger.info("Starting transcription...")

        self.stop_flag.clear()
        self.transcribe_thread = threading.Thread(
            target=self._transcribe_worker, args=(infile,), daemon=True
        )
        self.transcribe_thread.start()

    def _stop_transcription(self):
        self.stop_flag.set()
        self.ui_logger.info("Stopping transcription...")
        self.stop_btn.configure(state="disabled")

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
            )
            transcription = Audio.transcribe_audio()

            outfile = self.output_file_var.get().strip()
            if not outfile:
                outfile = infile + ".txt"

            with open(outfile, "w", encoding="utf-8") as f:
                f.write(transcription)

            self.ui_logger.info(f"Transcription saved to {outfile}")
        except Exception as e:
            self.ui_logger.error(f"Error: {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
