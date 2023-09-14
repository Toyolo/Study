"""
contains a bunch a utility functions for training and saving
"""
import torch
from pathlib import Path
def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """saves a model to a target directory
    Args:
        model: a pytorch model to save
        target_dir: a directory for saving the model to
        model_name: a filename for the saved model sould include either .pth or .pt as extension.
    
    example:
        save_model(model_0, models, tinyvgg_model.pth)
    """
    #create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(True, True)

    #create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "wrong extension, pth or pt pls"
    model_save_path = target_dir_path / model_name

    #save state dict
    print(f'[INFO] saving model to: {model_save_path}')
    torch.save(model.state_dict, model_save_path)
