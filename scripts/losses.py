import torch
import torch.nn as nn
from functools import partial

def neg_log(x):
    return -torch.log(x + 1e-5)

def bernoulli_entropy(p):
    return p * neg_log(p) + (1 - p) * neg_log(1 - p)

def _get_inds(target):
    batch_size = target.size(0)
    device = target.device
    inds = torch.arange(batch_size, device=device)

    return inds

def compute_components(loss_pos, loss_rand, target, flip_sign=True):
    """
    Compute loss components and total loss.
    flip_sign should be True for entropy-based losses.
    
    Parameters:
        loss_pos (Tensor): loss computed for the main branch.
        loss_rand (Tensor): loss computed for the randomized branch.
        target (Tensor): ground-truth class indices with shape [B, 1].
        
    Returns:
        total_loss (Tensor): scalar loss.
        components (dict): dictionary with individual loss components.
    """
    inds = _get_inds(target)
    
    # Create a mask that identifies target positions.
    target_mask = torch.zeros_like(loss_pos, dtype=torch.bool)
    target_mask.scatter_(1, target, True)

    # Compute components
    non_target = loss_pos[~target_mask].mean()
    background = loss_rand.mean()

    if flip_sign:
        non_target = -non_target
        background = -background

    components = {
        'target': loss_pos[target_mask].mean(),
        'non_target': non_target,
        'background': background
    }

    total_loss = loss_pos.mean() + loss_rand.mean()
    return total_loss, components

def an_full_loss(logits, rand_logits, target, pos_weight):
    """
    Compute the 'an_full' loss.
    """
    inds = _get_inds(target)
    
    # Loss for all classes.
    loss_pos = neg_log(1.0 - logits)
    # For the target class, scale the loss using pos_weight.
    loss_pos[inds, target.squeeze(-1)] = pos_weight * neg_log(logits[inds, target.squeeze(-1)])
    
    loss_rand = neg_log(1 - rand_logits)
    
    return compute_components(loss_pos, loss_rand, target, flip_sign=False)

def an_full_weighted_loss(logits, rand_logits, target, species_weights):
    """
    Compute the 'an_full' loss using species-specific inverse weights.
    
    Parameters:
        logits (Tensor): Logits for each class (batch_size, num_classes).
        rand_logits (Tensor): Logits for random classes (batch_size,).
        target (Tensor): Target class indices (batch_size, 1).
        species_weights (Tensor): Precomputed species weights (num_classes,).
    
    Returns:
        Tensor: Weighted loss.
    """
    inds = _get_inds(target)
    target_indices = target.squeeze(-1)
    species_weights = species_weights.to(target_indices.device)
    non_target_weights = species_weights / (species_weights - 1)

    # Compute the base loss for non-target locations: use -log(1 - logits)
    # Multiply each class's loss by its corresponding negative weight.
    loss_pos = non_target_weights.unsqueeze(0) * neg_log(1.0 - logits)

    # For the target (positive) class locations, override with the species weight and
    # a different loss formulation: -log(logit)
    loss_pos[inds, target_indices] = species_weights[target_indices] * neg_log(logits[inds, target_indices])

    # Random logits loss component
    loss_rand = neg_log(1 - rand_logits)

    return compute_components(loss_pos, loss_rand, target, flip_sign=False)


def max_entropy_loss(logits, rand_logits, target, pos_weight):
    """
    Compute the maxent poisson loss
    """
    batch_size, classes = logits.shape
    device = target.device
    inds = torch.arange(batch_size, device=device)
    
    # Normalize predicted intensities (i.e., λ) across species for each location
    lambda_sum = logits.sum(dim=1, keepdim=True) + 1e-5  # Avoid divide-by-zero
    lambda_norm = logits / lambda_sum  # Shape: [B, C]

    # Log of normalized λ
    loss_pos = neg_log(lambda_norm)

    loss_pos[inds, target.squeeze(-1)] *= pos_weight[target.squeeze(-1)]

    # same for random background
    rand_lambda_sum = rand_logits.sum(dim=1, keepdim=True) + 1e-5
    rand_lambda_norm = rand_logits / rand_lambda_sum
    log_rand_lambda_norm = neg_log(rand_lambda_norm)
    loss_rand = log_rand_lambda_norm
    
    return compute_components(loss_pos, loss_rand, target, flip_sign=True)

def max_entropy_weighted_loss(*args, **kwargs):
    # TODO
    raise NotImplementedError("max_entropy_weighted_loss is not implemented yet.")

def get_losses(loss_type, pos_weight=None, species_weights=None):
    """
    Return a loss function based on the loss_type string.
    
    Supported loss types:
        - 'an_full'
        - 'an_full_weighted'
        - 'max_entropy'
        - 'max_entropy_weighted'
    
    The returned function should have the signature:
        loss_fn(logits, rand_logits, target, pos_weight) -> (total_loss, components)
    """
    if loss_type == 'an_full':
        return partial(an_full_loss, pos_weight=pos_weight)
    elif loss_type == 'an_full_weighted':
        return partial(an_full_weighted_loss, species_weights=species_weights)
    elif loss_type == 'max_entropy':
        return max_entropy_loss
    elif loss_type == 'max_entropy_weighted':
        return max_entropy_weighted_loss
    else:
        raise ValueError("Unsupported loss type: {}".format(loss_type))
