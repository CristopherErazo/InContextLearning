import torch


def get_attention_patterns(model, sub_batch, path, device):
    """ 
    Get the attention patterns of the model on the sub_batch of size n_test. This function is used for evaluation during training.    
    """
    sequence = sub_batch['sequence'].to(device) # shape (n_test, seq_len + 1)
    input = sequence[:, :-1] # shape (n_test, seq_len)
    mask = sub_batch['mask'].to(device) # shape (n_test, seq_len, seq_len)

    with torch.no_grad():
        output = model.full_output(input,mask, path = path)
        attn1 = output.get('A1', None) # shape (n_test, seq_len, seq_len)
        attn2 = output.get('A2', None) # shape (n_test, seq_len, seq_len)
    return {'attn1': attn1, 'attn2': attn2}
    

