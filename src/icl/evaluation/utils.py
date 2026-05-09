import torch
import math


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