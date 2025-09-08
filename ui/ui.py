import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from ui.ui_log_handler import UILogHandler, setup_ui_logger
from transcription.torch_checker import check_torch
from transcription.device_configuration import DeviceConfiguration
from transcription.audio_transcription import AudioTranscription

def main():
    root = tk.Tk()
    root.title("Audio Transcriptor")
    root.geometry("800x600")

    for col in range(4):
        root.grid_columnconfigure(col, weight=1)
    root.grid_rowconfigure(6, weight=1)

    ### Buttons selector
    check_torch_baton = tk.Button(root, text="Check Torch")
    check_torch_baton.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

    # TODO: implement saving function
    save_configuration_baton = tk.Button(root, text="Save configuration")
    save_configuration_baton.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    # TODO: implement deleting function
    delete_configuration_baton = tk.Button(root, text="Delete configuration")
    delete_configuration_baton.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

    start_transcription_baton = tk.Button(root, text="Transcript")
    start_transcription_baton.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

    ### Model options selector
    model_options = [
        "openai/whisper-large-v2",
        "openai/whisper-large",
        "openai/whisper-medium",
        "openai/whisper-small",
        "openai/whisper-tiny"
    ]
    selected_model = tk.StringVar(value=model_options[0])
    label_model = tk.Label(root, text="Model name:")
    label_model.grid(row=1, column=0, sticky="w", pady=5, padx=5)
    dropdown_model_selection = tk.OptionMenu(root, selected_model, *model_options)
    dropdown_model_selection.grid(row=1, column=1, sticky="ew", pady=5, padx=5)

    ### Batch size selector
    batch_sizes = ["32", "16", "8", "4", "2"]
    selected_batch_size = tk.StringVar(value=batch_sizes[0])
    label_batch_size = tk.Label(root, text="Batch size:")
    label_batch_size.grid(row=1, column=2, sticky="w", pady=5, padx=5)
    dropdown_batch_size_selection = tk.OptionMenu(root, selected_batch_size, *batch_sizes)
    dropdown_batch_size_selection.grid(row=1, column=3, sticky="ew", pady=5, padx=5)

    ### Data type selector
    data_types = ["torch.float16", "torch.float32", "torch.bfloat16"]
    selected_data_type = tk.StringVar(value=data_types[0])
    label_data_type = tk.Label(root, text="Data type:")
    label_data_type.grid(row=2, column=0, sticky="w", pady=5, padx=5)
    dropdown_data_type_selection = tk.OptionMenu(root, selected_data_type, *data_types)
    dropdown_data_type_selection.grid(row=2, column=1, sticky="ew", pady=5, padx=5)

    ### Chunk length selector
    chunk_lengths = ["30", "25", "20", "15", "10", "5"]
    selected_chunk_length = tk.StringVar(value=chunk_lengths[0])
    label_chunk_length = tk.Label(root, text="Chunk length:")
    label_chunk_length.grid(row=2, column=2, sticky="w", pady=5, padx=5)
    dropdown_chunk_length_selection = tk.OptionMenu(root, selected_chunk_length, *chunk_lengths)
    dropdown_chunk_length_selection.grid(row=2, column=3, sticky="ew", pady=5, padx=5)
    
    # TODO: add device selector (cuda/mps/cpu)

    ### Filepath (input)
    # TODO: add checker if path is valid/invalid (i think in utils or something)
    label_file_path = tk.Label(root, text="Input filepath:")
    label_file_path.grid(row=3, column=0, sticky="w", pady=5, padx=5)
    file_path = tk.Text(root, height=1)
    file_path.grid(row=3, column=1, columnspan=3, sticky="ew", pady=5, padx=5)

    ### Filepath (output)
    # TODO: add question mark here with tip while mouse is on it
    label_output_file_path = tk.Label(root, text="Output filepath:")
    label_output_file_path.grid(row=4, column=0, sticky="w", pady=5, padx=5)
    output_file_path = tk.Text(root, height=1)
    output_file_path.grid(row=4, column=1, columnspan=3, sticky="ew", pady=5, padx=5)

    def show_selections():
        ui_logger.info(f"Selected model: {selected_model.get()}")
        ui_logger.info(f"Selected batch size: {selected_batch_size.get()} chunks")
        ui_logger.info(f"Selected data type: {selected_data_type.get()}")
    
    show_selections_baton = tk.Button(root, text="Show Selections", command=show_selections)
    show_selections_baton.grid(row=5, column=0, columnspan=4, pady=5, sticky="ew")

    log_box = ScrolledText(root, wrap="word")
    log_box.grid(row=6, column=0, columnspan=4, sticky="nsew", padx=10, pady=5)
    ui_logger = setup_ui_logger(log_box)
    
    def transcribe():
        current_device_config = DeviceConfiguration(
            device="cuda",
            model_name=selected_model.get(),
            batch_size=int(selected_batch_size.get()),
            chunk_length_s=30,
            data_type=selected_data_type.get()
        )
        Audio = AudioTranscription(
            filepath=file_path.get("1.0", "end-1c"),
            device_configuration=current_device_config,
            logger=ui_logger
        )
        transcription = Audio.transcribe_audio()
        with open(f"{file_path.get('1.0', 'end-1c')}.txt", "w") as output_file:
            output_file.write(transcription)
    
    check_torch_baton.config(command=lambda: check_torch(ui_logger))
    start_transcription_baton.config(command=transcribe)
    
    root.mainloop()



if __name__ == "__main__":
    main()
