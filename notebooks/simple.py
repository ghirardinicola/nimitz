# %%
import sys
import subprocess
print(sys.executable)
sys.path.append("../src")
version = subprocess.run([sys.executable, "--version"], capture_output=True, text=True).stdout.strip()
print(version)

#%%
from main import quick_analysis

#%%
import numpy as np
import torch

print("Torch version:", torch.__version__)

#%%
import clip

clip.available_models()
#%% Analisi super-rapida
yamato = quick_analysis('../images')
# %%
