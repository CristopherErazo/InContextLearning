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
        scores1 = output.get('S1', None) # shape (n_test, seq_len, seq_len)
        scores2 = output.get('S2', None) # shape (n_test, seq_len, seq_len)
        logits = output.get('logits', None) # shape (n_test, seq_len, vocab_size)
    return {'attn1': attn1, 'attn2': attn2, 'scores1': scores1, 'scores2': scores2, 'logits': logits}
    

def get_logits(model, sub_batch, path, device):
    """ 
    Get the attention patterns of the model on the sub_batch of size n_test. This function is used for evaluation during training.    
    """
    sequence = sub_batch['sequence'].to(device) # shape (n_test, seq_len + 1)
    input = sequence[:, :-1] # shape (n_test, seq_len)
    mask = sub_batch['mask'].to(device) # shape (n_test, seq_len, seq_len)

    with torch.no_grad():
        output = model.full_output(input,mask, path = path)
        logits = output.get('logits', None) # shape (n_test, seq_len, vocab_size)
    return logits
    



def get_order_parameters(model):
    with torch.no_grad():
        # Parameters
        P = model.embed.P.weight.data # shape (L, d)
        E = model.embed.E.weight.data # shape (V, d)

        WQ1 = model.attn1.WQ.weight.data # shape (r, d)
        WK1 = model.attn1.WK.weight.data # shape (r, d)
        WV1 = model.attn1.WV.weight.data # shape (r, d)
        WO1 = model.attn1.WO.weight.data # shape (d, r)

        WQ2 = model.attn2.WQ.weight.data # shape (r, d)
        WK2 = model.attn2.WK.weight.data # shape (r, d)
        WV2 = model.attn2.WV.weight.data # shape (r, d)
        WO2 = model.attn2.WO.weight.data # shape (d, r)

        U = model.unembed.U.weight.data # shape (V, d)

        # Parameters containing WOV2

        u_ov2_p = torch.linalg.multi_dot([U, WO2, WV2, P.T]) # shape (V, L)
        u_ov2_e = torch.linalg.multi_dot([U, WO2, WV2, E.T]) # shape (V, V)

        u_ov2_ov1_p = torch.linalg.multi_dot([U, WO2, WV2, WO1, WV1, P.T]) # shape (V, L)
        u_ov2_ov1_e = torch.linalg.multi_dot([U, WO2, WV2, WO1, WV1, E.T]) # shape (V, V)

        # Parameters containing WQK2
        e_qk2_e = torch.linalg.multi_dot([E, WQ2.T, WK2, E.T]) # shape (V, V)
        p_qk2_p = torch.linalg.multi_dot([P, WQ2.T, WK2, P.T]) # shape (L, L)

        e_qk2_ov1_e = torch.linalg.multi_dot([E, WQ2.T, WK2, WO1, WV1, E.T]) # shape (V, V)
        p_qk2_ov1_p = torch.linalg.multi_dot([P, WQ2.T, WK2, WO1, WV1, P.T]) # shape (L, L)

        e_ov1_qk2_e = torch.linalg.multi_dot([E, WO1, WV1, WQ2.T, WK2, E.T]) # shape (V, V)
        p_ov1_qk2_p = torch.linalg.multi_dot([P, WO1, WV1, WQ2.T, WK2, P.T]) # shape (L, L)

        e_ov1_qk2_ov1_e = torch.linalg.multi_dot([E, WO1, WV1, WQ2.T, WK2, WO1, WV1, E.T]) # shape (V, V)
        p_ov1_qk2_ov1_p = torch.linalg.multi_dot([P, WO1, WV1, WQ2.T, WK2, WO1, WV1, P.T]) # shape (L, L)

        # Parameters containing WQK1
        e_qk1_e = torch.linalg.multi_dot([E, WQ1.T, WK1, E.T]) # shape (V, V)
        p_qk1_p = torch.linalg.multi_dot([P, WQ1.T, WK1, P.T]) # shape (L, L)

        return {
            'u_ov2_p': u_ov2_p.cpu().numpy(),
            'u_ov2_e': u_ov2_e.cpu().numpy(),
            'u_ov2_ov1_p': u_ov2_ov1_p.cpu().numpy(),
            'u_ov2_ov1_e': u_ov2_ov1_e.cpu().numpy(),
            'e_qk2_e': e_qk2_e.cpu().numpy(),
            'p_qk2_p': p_qk2_p.cpu().numpy(),
            'e_qk2_ov1_e': e_qk2_ov1_e.cpu().numpy(),
            'p_qk2_ov1_p': p_qk2_ov1_p.cpu().numpy(),
            'e_ov1_qk2_e': e_ov1_qk2_e.cpu().numpy(),
            'p_ov1_qk2_p': p_ov1_qk2_p.cpu().numpy(),
            'e_ov1_qk2_ov1_e': e_ov1_qk2_ov1_e.cpu().numpy(),
            'p_ov1_qk2_ov1_p': p_ov1_qk2_ov1_p.cpu().numpy(),
            'e_qk1_e': e_qk1_e.cpu().numpy(),
            'p_qk1_p': p_qk1_p.cpu().numpy(),
        }
