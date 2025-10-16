import customtkinter as ctk


class ToolTip(ctk.CTkToplevel):
    def __init__(self, widget, text):
        super().__init__()
        self.withdraw()
        self.overrideredirect(True)
        self.attributes("-topmost", True)

        self.label = ctk.CTkLabel(
            self, text=text, fg_color="gray20", corner_radius=6, padx=10, pady=5
        )
        self.label.pack()

        self.widget = widget
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.geometry(f"+{x}+{y}")
        self.deiconify()

    def hide_tooltip(self, event=None):
        self.withdraw()
