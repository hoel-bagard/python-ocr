from dataclasses import dataclass
from pathlib import Path
from logging import StreamHandler, handlers
from typing import Optional
import logging
import os
import sys


@dataclass(frozen=True, slots=True)
class ConsoleColor:
    """Simple shortcut to use colors in the console."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    ORANGE = '\033[93m'
    RED = '\033[91m'
    ENDCOLOR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ColoredFormatter(logging.Formatter):
    """Formatter adding colors to levelname."""
    def format(self, record):
        levelno = record.levelno
        if logging.ERROR == levelno:
            levelname_color = ConsoleColor.RED + record.levelname + ConsoleColor.ENDCOLOR
        elif logging.WARNING == levelno:
            levelname_color = ConsoleColor.ORANGE + record.levelname + ConsoleColor.ENDCOLOR
        elif logging.INFO == levelno:
            levelname_color = ConsoleColor.GREEN + record.levelname + ConsoleColor.ENDCOLOR
        elif logging.DEBUG == levelno:
            levelname_color = ConsoleColor.BLUE + record.levelname + ConsoleColor.ENDCOLOR
        else:
            levelname_color = record.levelname
        record.levelname = levelname_color
        return logging.Formatter.format(self, record)


def create_logger(name: str, log_dir: Optional[Path] = None, stdout: bool = True) -> logging.Logger:
    """Create a logger.

    Note: If the logger has already been created once, it can be retrieved with logging.getLogger(<name>).
          For example, within a module, by using __name__ as the logger's name.

    Args:
        name: Name of the logger.
        log_dir: Folder where the logs will be saved. If None then logs do not get saved.
        stdout: If true then the logger will output to the stdout.

    Returns:
        The logger instance.
    """
    assert log_dir is not None or stdout, "You need to use at least one of log_dir or stdout"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        # Add a (rotating) file handler to the logging system
        file_log_handler = handlers.RotatingFileHandler(log_dir / (name + ".log"), maxBytes=500000, backupCount=2)
        file_log_handler.setFormatter(log_formatter)
        file_log_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_log_handler)

    if stdout:
        # Add an handler to the logging system (default has none) : outputing in stdout
        terminal_log_handler = StreamHandler(sys.stdout)
        terminal_log_handler.setLevel(logging.DEBUG)
        if os.name != 'nt':
            # Fancy color for non windows
            colored_log_formatter = ColoredFormatter("%(levelname)s - %(message)s")
            terminal_log_handler.setFormatter(colored_log_formatter)
        else:
            log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            terminal_log_handler.setFormatter(log_formatter)
        logger.addHandler(terminal_log_handler)

    return logger
