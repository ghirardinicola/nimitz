# %%
import sys
import subprocess
print(sys.executable)
sys.path.append("../src")
version = subprocess.run([sys.executable, "--version"], capture_output=True, text=True).stdout.strip()
print(version)

#%%
from main import quick_analysis, run_nimitz_pipeline

#%%
import numpy as np
import torch

print("Torch version:", torch.__version__)
#%% Analisi super-rapida
results = quick_analysis('../images')

# %% 
# Custom characteristics
custom_chars = {
    "style": ["photography", "painting", "illustration"],
    "mood": ["happy", "sad", "neutral"]
}
results = run_nimitz_pipeline("../images", characteristics=custom_chars)

# # Minimal functional approach
# from embed import extract_image_features
# features, paths = extract_image_features("./images", model, preprocess, device)
# %%
