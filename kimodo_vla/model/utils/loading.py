import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
from safetensors.torch import load_file as load_safetensors


def load_checkpoint_state_dict(ckpt_path: Union[str, Path]) -> dict:
    ckpt_path = os.path.join(str(ckpt_path), "model.safetensors")
    state_dict = load_safetensors(ckpt_path)
    return {key: val.detach().cpu() for key, val in state_dict.items()}
