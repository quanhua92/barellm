import torch

MODEL_ID = "Qwen/Qwen3-0.6B"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

if DEVICE == "cuda":
    DTYPE = torch.bfloat16
elif DEVICE == "mps":
    DTYPE = torch.float16
else:
    DTYPE = torch.float32
