import torch
import math
import numpy as np

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
    sequence = test_batch['sequence'][:n_test].to(device) # shape (n_test, seq_len + 1)
    trigger_set = test_batch['trigger_set'][:n_test].to(device) # shape (n_test, K)
    output_set = test_batch['output_set'][:n_test].to(device) # shape (n_test, K)
    counts = test_batch['counts'][:n_test].to(device) # shape (n_test, seq_len)
    is_trigg = test_batch['is_trigg'][:n_test].to(device) # shape (n_test, seq_len)
    mask = test_batch['mask'][:n_test].to(device) # shape (n_test, seq_len, seq_len)
    return {
        "sequence": sequence, # shape (n_test, seq_len + 1)
        "trigger_set": trigger_set, # shape (n_test, K)
        "output_set": output_set, # shape (n_test, K)
        "counts": counts,    # shape (n_test, seq_len)
        "is_trigg": is_trigg, # shape (n_test, seq_len)
        "mask": mask # shape (n_test, seq_len, seq_len)
    }



def on_off_masks(logit_shape, batch, device='cpu'):
    """ 
    generate a mask to apply to the logits (shape(B,L,V)) such that each element (b,l,v) is true
    it the input token at position l in batch b is a trigger and the counts for that trigger are larger than 1
    and the corresponding output token is v. 
    This mask will be used to compute the on-target and off-target logits for each trigger token in the batch.
    """ 
    batch_size, seq_len, vocab_size = logit_shape

    is_trigg = batch['is_trigg'].to(device) # shape (batch_size, seq_len)
    counts = batch['counts'].to(device) # shape (batch_size, seq_len)
    target = batch['sequence'][:, 1:].to(device) # shape (batch_size, seq_len)

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
    return on_target_mask, off_target_mask, all_mask


def get_evaluation_times(print_scale, total_steps, nprints, nprints_model):
    if print_scale == 'log':
        print_total_steps = np.unique(np.logspace(-0.01, np.log10(total_steps-1), num=nprints).astype(int))
        print_total_steps_model = np.unique(np.logspace(-0.01, np.log10(total_steps-1), num=nprints_model).astype(int))
    elif print_scale == 'linear':
        print_total_steps = np.linspace(0, total_steps-1, num=nprints).astype(int)
        print_total_steps_model = np.linspace(0, total_steps-1, num=nprints_model).astype(int)
    return print_total_steps, print_total_steps_model