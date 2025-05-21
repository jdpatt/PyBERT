"""Debug functions that should not be used in production."""
import builtins
import logging
from typing import Any

logger = logging.getLogger(__name__)

def setattr(obj: object, name: str, value: Any, /) -> None:
    """Wrapper around builtins.setattr to log the changes from the GUI."""
    builtins.setattr(obj, name, value)
    logger.debug(f"Set {name} to {value}")
