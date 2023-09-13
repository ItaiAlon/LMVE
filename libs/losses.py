import torch

def cov_mse_loss(pred, y):
    if y.ndim == 2: # replace by outer if needed
        y = torch.einsum('...i,...j->...ij', y, y)
    return torch.mean((y - pred).pow(2).sum(dim=(1, 2)))

def cov_gaussian_loss(pred, y, reg=1.):
    eye = torch.eye(y.shape[-1], dtype=torch.float32, device=pred.device)
    coverage = torch.mean(torch.einsum('...i,...ij,...j->...', y, torch.linalg.inv(pred + 1e-3 * eye), y))
    volume = torch.mean(torch.sqrt(torch.linalg.det(pred + 1e-3 * eye))) if reg > 0 else 0.
    return coverage + reg * volume
