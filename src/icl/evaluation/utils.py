import torch
import math
import numpy as np


def filter_batch(test_batch, device='cpu'):
    """
    Filter the batch to remove sequences that have non-trigger tokens with counts > 1 
    (i.e. keeping only sequences that have no trigger tokens with counts > 1 
    or sequences that have no trigger tokens at all).
    This is done to ensure that the evaluation is done on sequences where induction is possible. 
    The function returns the filtered batch and the fraction of sequences kept.
    """

    is_trigg = test_batch['is_trigg'].to(device) # shape (batch_size, seq_len)
    counts = test_batch['counts'].to(device) # shape (batch_size, seq_len)
    scores = (is_trigg * counts) # shape (batch_size, seq_len)
    max_score = scores.max(dim=-1).values # shape (batch_size,)

    # Keep sequences that have score > 1 
    # meaning: removing sequences that have non-trigger tokens at all
    # or sequences that have only trigger tokens with counts = 1
    keep_indices = (max_score > 1) # shape (batch_size,)
    filtered_batch = {k: v.to(device)[keep_indices] for k, v in test_batch.items()}
    sequences_kept = keep_indices.sum().item()
    
    return filtered_batch, sequences_kept/test_batch['sequence'].shape[0]


def preprocess_batch(test_batch, device='cpu'):

    test_batch , frac_kept = filter_batch(test_batch, device=device)

    sequence = test_batch['sequence'].to(device) # shape (B, L+1)
    target = sequence[:, 1:] # shape (B, L)
    is_trigg = test_batch['is_trigg'].to(device) # shape (B, L)
    counts = test_batch['counts'].to(device) # shape (B, L)        
    # First filter batch to remove sequences that have non-trigger tokens with counts > 1 (i.e. keeping only sequences that have no trigger tokens with counts > 1 or sequences that have no trigger tokens at all)


    # Masks for positions where induction is possible (is_trigg == 1 and counts > 1) and not possible
    ind_possible = (is_trigg == 1) & (counts > 1) # shape (B, L)
    ind_not_possible = ~ind_possible # shape (B, L)
    # Compute fraction of positions where induction is possible
    frac_ind_possible = ind_possible.float().mean().item()

    # Evaluate target at only trigger positions
    target_ind_positions = target[ind_possible]  # shape (num_masked_positions,)

    test_batch['ind_possible'] = ind_possible
    test_batch['ind_not_possible'] = ind_not_possible
    test_batch['target_ind_positions'] = target_ind_positions

    # Move all tensors to the specified device
    test_batch = {k: v.to(device) for k, v in test_batch.items()}
    return test_batch, frac_ind_possible, frac_kept


    

def compute_entropies_and_dkl(P_b:torch.Tensor,P_u:torch.Tensor):
    """ 
    Compute KL divergences between P_b and uniform distribution, P_u and uniform distribution, as well as the entropies of P_b and P_u.
    """
    vocab_size = P_u.shape[0]
    # Average dkl between bigram distribution and uniform 1/V
    kl_Pb_uniform = (P_b * (torch.log(P_b + 1e-10) - math.log(1.0/vocab_size + 1e-10))).sum(dim=-1).mean().item()
    
    # dkl between unigram and uniform 1/V
    kl_Pu_uniform = (P_u * (torch.log(P_u + 1e-10) - math.log(1.0/vocab_size + 1e-10))).sum().item()
   
    # Average entropy of bigram distribution
    entropy_Pb = -(P_b * torch.log(P_b + 1e-10)).sum(dim=-1).mean().item()
    entropy_Pu = -(P_u * torch.log(P_u + 1e-10)).sum().item()
    max_entropy = math.log(vocab_size)
    # return kl_Pb_uniform, kl_Pu_uniform, entropy_Pb, entropy_Pu, max_entropy
    # Return a dicitonary of the computed values for better readability
    return {
        "kl_Pb_uniform": kl_Pb_uniform,
        "kl_Pu_uniform": kl_Pu_uniform,
        "entropy_Pb": entropy_Pb,
        "entropy_Pu": entropy_Pu,
        "max_entropy": max_entropy
    }



def optimal_pop_losses(test_batch,P_b,p0=0.99):
    vocab_size = P_b.shape[-1]
    device = P_b.device
    input = test_batch['sequence'][:,1:-1].to(device) # shape (batch_size, seq_len-2)
    output = test_batch['sequence'][:,2:].to(device) # shape (batch_size, seq_len-2)
    is_trigg = test_batch['is_trigg'][:,1:].to(device) # shape (batch_size, seq_len-2)
    seq_len = input.shape[1]+2

    trigg_per_seq = is_trigg.sum(dim=-1).float().mean().item()/(seq_len-2)

    H_cond = -torch.sum(P_b * torch.log(P_b + 1e-10), dim=-1)  # shape (seq_len,)

    # Case 1: As if the model makes prediction with the dual 'teacher' model (up to p0 mass to avoid log(0) issues)
    input_eval = input[is_trigg==0] # shape (num_non_trigg_tokens,)
    loss = H_cond[input_eval]  # shape (num_non_trigg_tokens,)
    loss = loss.mean().item()
    pop_loss_1 = trigg_per_seq*(-math.log(p0)) + (1-trigg_per_seq)*loss

    # Case 2: As if the model makes prediction with induction head only, agnostic about frequencies
    pop_loss_2 = trigg_per_seq*(-math.log(p0)) + (1-trigg_per_seq)*math.log(vocab_size)

    # Case 3: As if the model makes predicitons with bigram statistics only
    loss_no_trig = H_cond[input]  # shape (batch_size, seq_len-2)

    # For loss2 evaluate the condition probability for each input,output pair 
    loss_trig = -torch.log(P_b[input,output]+1e-10)  # shape (batch_size, seq_len-2)

    # Where input is a trigger set loss = loss_trig, else loss = loss_no_trig
    loss = loss_no_trig.clone()
    loss[is_trigg==1] = loss_trig[is_trigg==1]
    pop_loss_3 = loss.mean().item()

    # return pop_loss_1, pop_loss_2, pop_loss_3, trigg_per_seq
    return {
        "pop_loss_dual": pop_loss_1,
        "pop_loss_induction": pop_loss_2,
        "pop_loss_bigram": pop_loss_3,
        "trigg_per_seq": trigg_per_seq
    }


def get_sub_batch(test_batch, device, n_test = 5):
    """ 
    Get a sub-batch of the given batch of size n_test. 
    """

    sub_batch = {k: v[:n_test].to(device) for k, v in test_batch.items()}
    return sub_batch



def get_best_sub_batch(test_batch, device, n_test = 5):
    """ 
    Get the 'best' sub-batch of the given batch of size n_test. 

    To get the 'best' sub-batch, asign a score to each element in the batch with is equal to the 
    sum of the counts of the trigger tokens in the sequence. Then select the n_test elements with the highest scores.
    """
    is_trigg = test_batch['is_trigg'].to(device) # shape (batch_size, seq_len)
    counts = test_batch['counts'].to(device) # shape (batch_size, seq_len)
    scores = (is_trigg * counts).sum(dim=-1) # shape (batch_size,)
    best_indices = torch.topk(scores, n_test).indices # shape (n_test,)

    sub_batch = {k: v[best_indices] for k, v in test_batch.items()}
    return best_indices, sub_batch




def get_indices(test_batch,vocab_size,device):
    """
    Get indices of the batch where is_trigg == 1 and counts > 1 (where induction can happen) 
    and the corresponding permutation of the vocabulary for each index such that the 
    last column is the evaluated output token and the rest are all other tokens in the vocabulary.
    """

    is_trigg = test_batch['is_trigg'].to(device) # shape (batch_size, seq_len)
    counts = test_batch['counts'].to(device) # shape (batch_size, seq_len)
    # Get indices (batch, seq_len) where is_trigg == 1 and counts > 1
    idx_ind = torch.nonzero(is_trigg & (counts > 1), as_tuple=False) # shape (num_indices, 2)
    n_ind = idx_ind.shape[0]

    # Extract the sequence and output from the test batch only for the indices where induction can happen
    sequence = test_batch['sequence'].to(device) # shape (B,L+1)
    output = sequence[:, 1:].to(device) # shape (B,L)
    output_ind = output[idx_ind[:,0], idx_ind[:,1]] # shape (num_indices,)

    # Construct permutation of the vocabulary for each index in idx_ind, where the last column is the evaluated output token and the rest are all other tokens in the vocabulary
    all_idx = torch.arange(vocab_size, device=device).expand(n_ind, vocab_size) # shape (num_indices, vocab_size)
    non_target_idx = all_idx[all_idx != output_ind[:, None]].view(n_ind, vocab_size - 1)
    perm = torch.cat([non_target_idx, output_ind[:, None]], dim=1) # shape (num_indices, vocab_size)

    return idx_ind, perm



def on_off_logit_masks(batch, vocab_size,device='cpu'):
    """ 
    generate a mask to apply to the logits (shape(B,L,V)) such that each element (b,l,v) is true
    it the input token at position l in batch b is a trigger and the counts for that trigger are larger than 1
    and the corresponding output token is v. 
    This mask will be used to compute the on-target and off-target logits for each trigger token in the batch.
    """ 


    is_trigg = batch['is_trigg'].to(device) # shape (batch_size, seq_len)
    counts = batch['counts'].to(device) # shape (batch_size, seq_len)
    target = batch['sequence'][:, 1:].to(device) # shape (batch_size, seq_len)

    batch_size, seq_len = is_trigg.shape
    logit_shape = (batch_size, seq_len, vocab_size)

    trigg_mask = is_trigg.bool().unsqueeze(-1) & (counts > 1).unsqueeze(-1) 
    on_target_mask = (torch.arange(vocab_size, device=device).view(1, 1, -1) == target.unsqueeze(-1))
    off_target_mask = ~on_target_mask
    on_target_mask = trigg_mask & on_target_mask
    off_target_mask = trigg_mask & off_target_mask


    # create a mask for the logits that mask out the first 2 sequence positions
    mask_first_two_positions = torch.ones(logit_shape, dtype=torch.bool, device=device)  # shape (batch_size, seq_len, vocab_size)
    mask_first_two_positions[:, :2, :] = False

    on_target_mask = on_target_mask & mask_first_two_positions
    off_target_mask = off_target_mask & mask_first_two_positions
    all_mask = mask_first_two_positions & trigg_mask
    return on_target_mask, off_target_mask, all_mask # sizes (batch_size, seq_len, vocab_size)


def on_off_attn1_masks(batch,device='cpu'):
    mask = batch['mask'].to(device) # shape (batch_size, seq_len, seq_len)
    attn_shape = mask.shape # shape (batch_size, seq_len, seq_len)
    all_mask = mask # shape (batch_size, seq_len, seq_len)

    # On mask is true only in sub-diagonal. must be same shape as all_mask
    # Make a matrix with 1 in sub-diagonal, zero everywere then replicate and convert to bool
    sub_diag = torch.eye(attn_shape[-1], dtype=torch.bool, device=device).roll(1, dims=0) # shape (seq_len, seq_len)
    on_mask = sub_diag.unsqueeze(0).expand(attn_shape[0], -1, -1) # shape (batch_size, seq_len, seq_len)
    off_mask = ~on_mask & all_mask # shape (batch_size, seq_len, seq_len)
    return on_mask, off_mask, all_mask

def on_off_attn2_masks(batch,device='cpu'):
    mask = batch['mask'].to(device) # shape (batch_size, seq_len, seq_len)
    sequence = batch['sequence'].to(device) # shape (batch_size, seq_len + 1)
    is_trigg = batch['is_trigg'].to(device) # shape (batch_size, seq_len)

    attn_shape = mask.shape # shape (batch_size, seq_len, seq_len)
    all_mask = mask # shape (batch_size, seq_len, seq_len)

    # On mask is true if the input token is a trigger and the output token is the corresponding output token. must be same shape as all_mask
    on_mask = torch.zeros(attn_shape, dtype=torch.bool, device=device) # shape (batch_size, seq_len, seq_len)
    for b in range(attn_shape[0]):
        for l in range(attn_shape[1]):
            if is_trigg[b, l]:
                target_token = sequence[b, l + 1] # the corresponding output token is at position l+1 in the sequence
                # on mask is true for all previous positions (0 to l) that have the target token
                for prev_l in range(l):
                    if sequence[b, prev_l] == target_token:
                        on_mask[b, l, prev_l] = True
                        # Pass if already found a match, to avoid overwriting with a later match
                        break
    
    off_mask = ~on_mask & all_mask # shape (batch_size, seq_len, seq_len)
    return on_mask, off_mask, all_mask

def get_on_off_masks(batch,vocab_size,device='cpu'):
    on_target_mask, off_target_mask, all_mask_logits = on_off_logit_masks(batch,vocab_size,device=device)
    on_attn1_mask, off_attn1_mask, all_mask_attn1 = on_off_attn1_masks(batch,device=device)
    on_attn2_mask, off_attn2_mask, all_mask_attn2 = on_off_attn2_masks(batch,device=device)
    return {
        "logits": {
            "on": on_target_mask,
            "off": off_target_mask,
            "all": all_mask_logits
        },
        "attn1": {
            "on": on_attn1_mask,
            "off": off_attn1_mask,
            "all": all_mask_attn1
        },
        "attn2": {
            "on": on_attn2_mask,
            "off": off_attn2_mask,
            "all": all_mask_attn2
        }
    }
    


def get_evaluation_times(args):
    """
    Get the evaluation times based on the configuration in args.
    """
    print_scale = args.print_scale
    total_steps = args.total_steps
    nprints = args.n_prints
    nprints_model = args.n_prints_model
    if print_scale == 'log':
        print_total_steps = np.unique(np.logspace(-0.01, np.log10(total_steps-1), num=nprints).astype(int))
        print_total_steps_model = np.unique(np.logspace(-0.01, np.log10(total_steps-1), num=nprints_model).astype(int))
    elif print_scale == 'linear':
        print_total_steps = np.linspace(0, total_steps-1, num=nprints).astype(int)
        print_total_steps_model = np.linspace(0, total_steps-1, num=nprints_model).astype(int)
    return print_total_steps, print_total_steps_model