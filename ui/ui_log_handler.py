import logging
import tkinter as tk


class UILogHandler(logging.Handler):
    def __init__(self, text_widget: tk.Text):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        log_entry = self.format(record)
        self.text_widget.insert(tk.END, log_entry + "\n")
        self.text_widget.see(tk.END)


# TODO: status should be here
def setup_ui_logger(text_widget: tk.Text, level=logging.INFO):
    logger = logging.getLogger("UI_LOGGER")
    logger.setLevel(level)
    logger.handlers.clear()

    handler = UILogHandler(text_widget)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    return logger
