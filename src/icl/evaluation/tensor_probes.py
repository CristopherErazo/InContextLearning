import torch
import numpy as np


def get_activations(model, sub_batch, path, device):
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
        logits = output.get('logits', None) # shape (n_test, seq_len, vocab_size)
    return {'attn1': attn1, 'attn2': attn2, 'logits': logits}
    

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
    

def get_logit_distributions(model, sub_batch, device, n_bins=30):

    input = sub_batch['sequence'][:, :-1].to(device) # shape (n_test, seq_len)
    mask = sub_batch['mask'].to(device) # shape (n_test, seq_len, seq_len)

    with torch.no_grad():
        logits = model(input,mask)  #shape (n_test, seq_len, vocab_size)

    # Get the masks
    ind_possible = sub_batch['ind_possible'].to(device) # shape (n_test, seq_len)
    ind_not_possible = sub_batch['ind_not_possible'].to(device) # shape (n_test, seq_len)

    # Get the target tokens where the induction is possible
    target_ind = sub_batch['target_ind_positions'].to(device) # shape (num_masked_positions,)
    # which means that target_ind[i] is the target token (=0,1,...,V-1) for the i-th element in the masked batch

    # Mask logits
    logits_masked = logits[ind_possible] # shape (num_masked_positions, vocab_size)
    # for debugg apply softmax to get probabilities
    # logits_masked = torch.softmax(logits_masked, dim=-1) # shape (num_masked_positions, vocab_size)
    vmin,vmax = logits_masked.min().item(), logits_masked.max().item()

    # On-target logits: (N_masked,)
    on_target_logits = logits_masked[
        torch.arange(logits_masked.size(0), device=device),
        target_ind,
    ]

    # Off-target logits: (N_masked, vocab_size - 1)
    vocab_size = logits_masked.size(-1) 
    vocab_indices = torch.arange(vocab_size, device=device)

    off_target_mask = vocab_indices.unsqueeze(0) != target_ind.unsqueeze(1)
    off_target_logits = logits_masked[off_target_mask].reshape(
        logits_masked.size(0), vocab_size - 1
    )

    # move to cpu for histogram computation
    on_target_logits = on_target_logits.detach().cpu()
    off_target_logits = off_target_logits.detach().cpu()

    # Compute on-off histograms
    on_hist , edges = torch.histogram(on_target_logits, bins=n_bins, range=(vmin, vmax),density=True)
    off_hist, _ = torch.histogram(off_target_logits.flatten(), bins=n_bins, range=(vmin, vmax),density=True)
    return {
        'on_hist': on_hist,
        'off_hist': off_hist,
        'edges': edges,
        'range': (vmin, vmax),
        'on_mean': on_target_logits.mean(),
        'off_mean': off_target_logits.mean(),
        'on_std': on_target_logits.std(),
        'off_std': off_target_logits.std(),
    }


def get_per_position_on_off_logits(model, sub_batch, device):

    input = sub_batch['sequence'][:, :-1].to(device) # shape (n_test, seq_len)
    mask = sub_batch['mask'].to(device) # shape (n_test, seq_len, seq_len)
    L = input.size(1)
    with torch.no_grad():
        logits = model(input,mask)  #shape (n_test, seq_len, vocab_size)

    # Get the masks
    ind_possible = sub_batch['ind_possible'].to(device) # shape (n_test, seq_len)

    b_idx, l_idx = torch.where(ind_possible) # shape (num_masked_positions,)

    # Masked logits and targets for positions where induction is possible
    N = b_idx.shape[0]  # number of positions where induction is possible = num_masked_positions
    masked_logits = logits[b_idx, l_idx]  # shape (num_masked_positions, V)
    masked_targets = sub_batch['target_ind_positions'].to(device) # shape (num_masked_positions,)

    # Get ontarget logits for positions where induction is possible
    # Target logits
    on_target_logits = masked_logits.gather( 1, masked_targets[:, None]).squeeze(1)    # (N,)

    off_target_logits = masked_logits.clone()
    off_target_logits[torch.arange(N, device=device), masked_targets] = float('nan')  # set on-target logits to NaN
    off_target_logits = off_target_logits[~torch.isnan(off_target_logits)]  # remove NaN values
    off_target_logits = off_target_logits.view(N, -1)  # reshape to (N, V-1)
    # Create a dictionary to store the results for each sequence length
    on_results = [on_target_logits[l_idx == l] for l in range(L)]
    off_results = [off_target_logits[l_idx == l] for l in range(L)]
    all_results = [masked_logits[l_idx == l] for l in range(L)]

    on_results = [r.cpu().numpy() for r in on_results]
    off_results = [r.cpu().numpy() for r in off_results]
    all_results = [r.cpu().numpy() for r in all_results]
    

    return {'on':on_results, 'off':off_results, 'all':all_results}

    
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
