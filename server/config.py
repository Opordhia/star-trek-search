import os
import pathlib

# Constants
ALLOWED_DIRECTORIES = [
    str(pathlib.Path(os.path.expanduser("/data")).resolve())
]