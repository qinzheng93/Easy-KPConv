import torch

def index_select(data, index, dim):
    r"""Advanced index select.
    
    Returns a tensor `output` which indexes the `data` tensor along dimension `dim` using the entries in `index`
    which is a `LongTensor`.

    Different from `torch.index_select`, `index` does not has to be 1-D.
    The `dim`-th dimension of `data` will be expanded to the number of dimensions in `index`.

    For example, suppose the shape `data` is $(a_0, a_1, ..., a_{n-1})$, the shape of `index` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.

    Args:
        data: torch.Tensor, (a_0, a_1, ..., a_{n-1})
        index: torch.LongTensor, (b_0, b_1, ..., b_{m-1})
        dim: int

    Returns:
        output: torch.Tensor, (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    output = data.index_select(dim, index.view(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)

    return output

