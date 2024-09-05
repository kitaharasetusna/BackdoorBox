# test Spatial Transformer Network (STN) + encoder -> New encoder 
import torch
import sys
sys.path.append('..')
from myutils import utils_attack

# Example usage
input_batch = torch.randn(8, 3, 32, 32)  # (batch_size, channels, height, width)

# Initialize the model
model = utils_attack.EncoderWithFixedTransformation(input_channels=3)

# Forward pass through the model
output = model(input_batch)

# Output should have the same shape as the input
print(f"Input shape: {input_batch.shape}")
print(f"Output shape: {output.shape}")
