import torch
import torch.nn.functional as F


def _squash(x):
    norm = x.norm(p=2, dim=-1, keepdim=True)
    scale = norm / (1 + norm ** 2)
    return scale * input

def _dynamic_routing(input, bias, num_iter, squash):
    """
    Parameters
    ----------
    input : ``torch.LongTensor``
        The input capsules of shape ``(batch_size, out_capsules, in_capsules, out_dim)`` 
    bias : ``torch.LongTensor``
        bias for the output capsules, having shape of ``(batch_size, out_capsules, out_dim)``
    Returns
    -------
    out: ``torch.LongTensor``
        The output capsule of shape: ``(batch_size, out_capsules, out_dim)``
    """
    b_ij = torch.zeros_like(input)
    for i in range(num_iter):
        c_ij = F.softmax(b_ij, dim=-3)

        # out:   [batch_size, out_capsules, 1,           out_dim] <-
        # c_ij:  [batch_size, out_capsules, in_capsules, out_dim] *
        # input: [batch_size, out_capsules, in_capsules, out_dim]
        out = (c_ij * input).sum(dim=-2) + bias

        # out: [batch_size, out_capsules, 1, out_dim] 
        out = _squash(out)

        # b_ij: [batch_size, out_capsules, in_capsules, 1]
        b_ij = (out * input).sum(dim=-1, keepdim=True)

        # original implementation in the paper
        # b_ij = b_ij + (out * input).sum(dim=-1, keepdim=True)
        
    return out    
    

def _k_means_routing():
    """
    Parameters
    ----------
    input: ``torch.LongTensor``
        A tensor of shape ``(batch_size, out_capsules, in_capsules, out_dim)`` 
    Returns
    -------
        out: ``torch.LongTensor``
        The output capsule of shape: ``(batch_size, out_capsules, out_dim)``
    """


class CapsuleLinear(nn.Module):
    """
    Parameters
    ----------
    in_capsules : ``int``, optional
        number of input capsules
    in_dim : ``int``, required
        dimension of input capsule
    out_capsules : ``int``, required
        number of output capsules
    out_dim : ``int``, required
        dimension of output capsule
    share_weight: ``bool``, (default=``True``)
        if True, weight will be shared between input capsules
    routing_type: ``str``, (default='k_means')
        routing algorithm type
        -- options: ['dynamic', 'k_means']
    num_iter: ``int``, (default = 3)
        number of routing iterations
    dropout (float, optional): if non-zero, introduces a dropout layer on the inputs
    bias (bool, optional):  if True, adds a learnable bias to the output
    kwargs (dict, optional): other args:
    - similarity (str, optional): metric of similarity between capsules, it only works for 'k_means' routing
        -- options: ['dot', 'cosine', 'tonimoto', 'pearson']
    - squash (bool, optional): squash output capsules or not, it works for all routing
    - return_prob (bool, optional): return output capsules' prob or not, it works for all routing
    - softmax_dim (int, optional): specify the softmax dim between capsules, it works for all routing

    
    """

    def __init__(self, 
                 in_capsules: int,
                 in_dim: int,
                 out_capsules: int,
                 out_dim: int,
                 share_weight: bool,
                 routing_type: str,
                 num_iter: int):

        if share_weight = True:
            self.weight = Parameter(torch.Tensor(out_capsules, out_dim, in_dim))
        else:
            self.weight = Parameter(torch.Tensor(out_capsules, in_capsules, out_dim, in_dim))   

        self.bias = Parameter(torch.Tensor(out_capsules, out_dim))

        self.in_capsules = in_capsules
        self.out_capsules = out_capsules

        self.in_dim = in_dim    
        self.out_dim = out_dim

        self.routing_type = routing_type
        self.num_iter = num_iter

        nn.init.xavier_uniform_(self.bias)

    def forward(self, input):
    """
    Parameters
    ----------
    input : Dict[str, torch.LongTensor], required
        The input embedded text vectors of shape ``(batch_size, in_capsules, in_dim)``
    Returns
    -------
    out : ``torch.LongTensor``
        A tensor of shape ``(batch_size, out_capsules, in_capsules, out_dim)`` 
    """

    if share_weight：
        # [batch_size, out_capsules, in_capsules, out_dim] =
        # [None,       out_capsules, None,        out_dim, in_dim] @ 
        # [batch_size, None,         in_capsules, in_dim,  None]
        transformed_input = (self.weight[None, :, None, :, :] @ input[:, None, :, :, None]).squeeze(dim=-1)
    else:
        # [batch_size, out_capsules, in_capsules, out_dim] =
        # [None,       out_capsules, in_capsules, out_dim, in_dim] @ 
        # [batch_size, None,         in_capsules, in_dim,  None]
        transformed_input = (self.weight[None, :, :, :, :] @ input[:, None, :, :, None]).squeeze(dim=-1)

    # bias is of shape [1, out_capsules, 1, out_dim]
    self.bias = self.bias.reshape(1, 1, *bias.size()).permute(0, 2, 1, 3).contiguous()

    if self.routing_type = 'dynamic':
        # out is of shape [batch_size, out_capsules, out_dim]
        out = _dynamic_routing(transformed_input, self.num_iter)
    
    return out    


