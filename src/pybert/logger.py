import logging

def setup_logger(filename: str, console_debug:bool = False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a file logger to log all debug messages
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # Create a console logger to only show the user error messages.
    ch = logging.StreamHandler()
    log_level = logging.DEBUG if console_debug else logging.ERROR
    ch.setLevel(log_level)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)


class TraitsUiConsoleHandler(logging.Handler):
    """Create a custom logging handler that appends each record to the TraitsUi TextArea."""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.setLevel(logging.INFO)

    def emit(self, record):
        """Append the record to the console."""
        self.app.console_log  += self.format(record) + "\n"
