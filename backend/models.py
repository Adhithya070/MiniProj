import sys
sys.path.append("D:/miniproj/stylegan2_ada_pytorch")
import torch
from torch.serialization import add_safe_globals
from torch_utils.persistence import _reconstruct_persistent_obj
from numpy import dtype  # <-- Add this import
from numpy.core.multiarray import scalar  # <-- Keep this import

# Allow required globals for torch.load()
add_safe_globals([
    _reconstruct_persistent_obj,
    scalar,
    dtype  # <-- Add numpy dtype to allowed list
])

from training import networks

def get_pretrained_discriminator(checkpoint_path, resolution=256):
    # Load the checkpoint with weights_only=True
    ckpt = torch.load(
        checkpoint_path, 
        map_location="cuda", 
        weights_only=False
    )
    
    # Extract the discriminator state
    D_state = ckpt['D']
    
    # If D_state is a persistent object, convert it to a state_dict
    if hasattr(D_state, 'state_dict'):
        D_state = D_state.state_dict()
    
    # Instantiate the discriminator
    D = networks.Discriminator(c_dim=0, img_resolution=resolution, img_channels=3)
    
    # Load the state_dict
    D.load_state_dict(D_state, strict=False)
    return D

def modify_discriminator_for_deepfake(D):
    import torch.nn as nn
    # Wrap the existing final fully connected layer (D.b4.out) with a Sigmoid activation.
    D.b4.out = nn.Sequential(
        D.b4.out,
        nn.Sigmoid()
    )
    return D

if __name__ == "__main__":
    checkpoint = "backends/models/stylegan2-ada-pretrained.pt"
    discriminator = get_pretrained_discriminator(checkpoint, resolution=256)
    discriminator = modify_discriminator_for_deepfake(discriminator)
    print("Discriminator loaded successfully:")
    print(discriminator)