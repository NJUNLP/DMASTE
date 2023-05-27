import torch 
def batched_index_select(target, indices):
    """
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A tensor of shape (batch_size, ...), where each element is an index into the
        `sequence_length` dimension of the `target` tensor.
    """
    
    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))
    flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))
    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets

def flatten_and_batch_shift_indices(indices, sequence_length):
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        print(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
        exit()
    offsets = torch.arange(start=0, end=indices.size(0), dtype=torch.long).to(indices.device) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices