import os
import pathlib

# Only list the root directory, it will only use the first one in the list
ALLOWED_DIRECTORIES = [
    str(pathlib.Path(os.path.expanduser("/data")).resolve())
]
