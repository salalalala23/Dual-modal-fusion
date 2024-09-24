import torch
import numpy as np

def mutual_information(tensor1, tensor2, bins=64):
    """
    Calculate the mutual information between two tensors.

    :param tensor1: First tensor (BCHW)
    :param tensor2: Second tensor (BCHW)
    :param bins: Number of bins to use for histogram
    :return: Mutual information value
    """

    tensor1 = torch.mean(tensor1, dim=1)
    tensor2 = torch.mean(tensor2, dim=1)
    # Flatten the tensors along the spatial dimensions (HW)
    t1_flat = tensor1.view(tensor1.size(0) * tensor1.size(1), -1)
    t2_flat = tensor2.view(tensor2.size(0) * tensor2.size(1), -1)

    # Calculate the joint histogram
    joint_hist = np.histogram2d(t1_flat.cpu().numpy().flatten(), t2_flat.cpu().numpy().flatten(), bins=bins)[0]

    # Calculate the probability distribution
    pxy = joint_hist / float(np.sum(joint_hist))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x

    # Broadcast to match dimensions
    px_py = px[:, None] * py[None, :]

    # Calculate the mutual information
    nzs = pxy > 0 # Non-zero elements
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    return mi

# Example tensors
tensor1 = torch.randn(1, 3, 64, 64)  # Random tensor of size BCHW
tensor2 = torch.randn(1, 1, 64, 64)  # Another random tensor of size BCHW

# Calculate mutual information
mi = mutual_information(tensor1, tensor2)
print(mi)
mi = mutual_information(tensor1, tensor1)
print(mi)
mi = mutual_information(tensor2, tensor2)
print(mi)
