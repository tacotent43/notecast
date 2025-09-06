import tkinter as tk

root = tk.Tk()
root.title("Audio Transcriptor")
root.geometry("800x600")

path_entry = tk.Entry(root, width=40)
path_entry.pack(padx=10, pady=(5, 10))

transcribe_button = tk.Button(text="Transcribe")
transcribe_button.pack(anchor="e", rely=25, relx=15)

root.mainloop()