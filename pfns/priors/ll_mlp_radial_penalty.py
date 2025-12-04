
import torch

from pfns.priors import Batch
from pfns.priors.simple_mlp import get_batch as get_batch_for_mlp

@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, hyperparameters, device, num_outputs=1, n_targets_per_input=1, **kwargs):
    assert len(hyperparameters) == 0

    hyperparameters = {
            "mlp_num_layers": torch.randint(15, 35, (1,)),
            "mlp_num_hidden": torch.randint(128, 512, (1,)),
            "mlp_init_std": torch.rand(1).item() / 5,
            "mlp_sparseness": torch.rand(1).item() / 10,
            "mlp_input_sampling": "normal",
            "mlp_output_noise": 0.0,
            "mlp_noisy_targets": False,
            "mlp_preactivation_noise_std": 0.0,
    }
    batch = get_batch_for_mlp(batch_size, seq_len, num_features, hyperparameters, device, num_outputs, n_targets_per_input, **kwargs)

    batch_size = batch.x.shape[1]
    num_features = batch.x.shape[2]
    
    # α > 0
    alpha = torch.rand(batch_size) * 2.0 + 0.5  # α in [0.5, 2.5]
    alpha = alpha.view(1, batch_size, 1)  # shape: (1, batch_size, 1)
    
    p = torch.rand(batch_size) * 4.0 + 1.1  # p in [1.1, 4.1]
    p = p.view(1, batch_size, 1) 
    
    # ε > 0 small
    epsilon = torch.rand(batch_size) * 1e-4 + 1e-6  # ε in [1e-6, 1.0001e-4]
    epsilon = epsilon.view(1, batch_size, 1)  # shape: (1, batch_size, 1)
    
    b = torch.randint(0, 3, (1,))
    mu_tilde = torch.randn(batch_size, num_features)  
    mu = b * torch.tanh(mu_tilde)
    mu = mu.view(1, batch_size, num_features)
    
    x_minus_mu = batch.x - mu.to(device)  # subtract mean (broadcasts correctly)
    squared_norm = torch.sum(x_minus_mu ** 2, dim=2, keepdim=True)  # ||x - μ||²
    radial_penalty = alpha.to(device) * (squared_norm + epsilon.to(device)) ** (p.to(device) / 2)
    batch.y = torch.exp(batch.y - radial_penalty.to(device))
    batch.y = torch.log(batch.y + 1)

    batch = Batch(
        x=batch.x.transpose(0, 1),
        y=batch.y.transpose(0, 1),
        target_y=batch.y.transpose(0, 1),
    )
    return batch