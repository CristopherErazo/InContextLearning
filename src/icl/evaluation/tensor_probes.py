import torch



def get_attention_patterns(model, test_batch, path, device, n_test = 5):
    """ 
    Get the attention patterns of the model on the given batch. This function is used for evaluation during training.    
    """
    sequence = test_batch['sequence'][:n_test].to(device) # shape (n_test, seq_len + 1)
    input = sequence[:, :-1] # shape (n_test, seq_len)
    mask = test_batch['mask'][:n_test].to(device) # shape (n_test, seq_len, seq_len)

    with torch.no_grad():
        output = model.full_output(input,mask, path = path)
        attn1 = output.get('A1', None) # shape (n_test, seq_len, seq_len)
        attn2 = output.get('A2', None) # shape (n_test, seq_len, seq_len)
    return {'attn1': attn1, 'attn2': attn2}
    