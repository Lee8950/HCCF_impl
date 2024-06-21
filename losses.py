import torch
import torch.nn.functional

def contrast_loss(embedding1, embedding2, nodes, temperture):
    # Normalizing embedding, adding 1e-8 to avoid value turning zero,
    # in paper they use temperture as softmax normalize factor.
    embedding1 = torch.nn.functional.normalize(embedding1 + 1e-8, p=2)
    embedding2 = torch.nn.functional.normalize(embedding2 + 1e-8, p=2)
    
    return -torch.log(
        (torch.exp(torch.sum(embedding1[nodes] * embedding2[nodes], dim=-1) / temperture)) /
        (torch.exp(embedding1[nodes] @ embedding2[nodes].T / temperture).sum(-1) + 1e-8)
    ).mean()