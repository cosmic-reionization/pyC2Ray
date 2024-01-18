import numpy as np
import pyc2ray as p2c

from .load_extensions import load_c2ray, load_asora
from .asora_core import cuda_is_init

# Load extension modules
libc2ray = load_c2ray()
libasora = load_asora()