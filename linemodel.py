import torch
import torch.nn as nn
import matplotlib.pyplot as plt

what_im_building = {1: "data (prep and load)",
                    2: "build model",
                    3: "train",
                    4: "inference",
                    5: "saving and loading a model",
                    6: "put it together"
                    }

# 0. housekeeping
# check pytorch version
print(torch.__version__)
