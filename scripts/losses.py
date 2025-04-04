import torch
import torch.nn as nn

def neg_log(x):
    return -torch.log(x + 1e-5)

def bernoulli_entropy(p):
    return p * neg_log(p) + (1 - p) * neg_log(1 - p)

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
    batch_size = target.size(0)
    device = target.device
    inds = torch.arange(batch_size, device=device)
    
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
    batch_size = target.size(0)
    device = target.device
    inds = torch.arange(batch_size, device=device)
    
    # Loss for all classes.
    loss_pos = neg_log(1.0 - logits)
    # For the target class, scale the loss using pos_weight.
    loss_pos[inds, target.squeeze(-1)] = pos_weight * neg_log(logits[inds, target.squeeze(-1)])
    
    loss_rand = neg_log(1 - rand_logits)
    
    return compute_components(loss_pos, loss_rand, target, flip_sign=False)

def an_full_weighted_loss(*args, **kwargs):
    # TODO
    raise NotImplementedError("an_full_weighted_loss is not implemented yet.")

def max_entropy_loss(logits, rand_logits, target, pos_weight):
    # TODO
    """
    Compute the 'max_entropy' loss.
    """
    batch_size = target.size(0)
    device = target.device
    inds = torch.arange(batch_size, device=device)
    
    loss_pos = -bernoulli_entropy(1.0 - logits)
    loss_pos[inds, target.squeeze(-1)] = pos_weight * bernoulli_entropy(logits[inds, target.squeeze(-1)])
    
    loss_rand = -bernoulli_entropy(1 - rand_logits)
    
    return compute_components(loss_pos, loss_rand, target, flip_sign=True)

def max_entropy_weighted_loss(*args, **kwargs):
    # TODO
    raise NotImplementedError("max_entropy_weighted_loss is not implemented yet.")

def get_losses(loss_type):
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
        return an_full_loss
    elif loss_type == 'an_full_weighted':
        return an_full_weighted_loss
    elif loss_type == 'max_entropy':
        return max_entropy_loss
    elif loss_type == 'max_entropy_weighted':
        return max_entropy_weighted_loss
    else:
        raise ValueError("Unsupported loss type: {}".format(loss_type))
