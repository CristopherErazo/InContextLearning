import torch
import math
from .utils import on_off_masks

def compute_kl(P, Q):
    "P and Q are distributions over the same support, shape (..., vocab_size) = (B,L,V) usually"
    return (P * (torch.log(P + 1e-10) - torch.log(Q + 1e-10))).sum(dim=-1).mean()

class EvalContext:
    def __init__(self, model, batch, loss_fn, P_b, P_u):
        device = next(model.parameters()).device

        # Ground Variables
        self.sequence = batch['sequence'].to(device) # shape (B, L+1)
        self.input = self.sequence[:, :-1] # shape (B, L)
        self.target = self.sequence[:, 1:] # shape (B, L)
        self.is_trigger = batch['is_trigger'].to(device) # shape (B, L)
        self.counts = batch['counts'].to(device) # shape (B, L)
        self.mask = batch['mask'].to(device) # shape (B, L, L)
        self.trigger_set = batch['trigger_set'].to(device) # shape (B, K)
        self.trigger_set_unique = self.trigger_set[0] # shape (K,)
        self.only_triggers = (self.input.unsqueeze(-1) == self.trigger_set.unsqueeze(1)).any(-1) & (self.counts >= 2) # shape (B, L)
        self.only_non_triggers = ~( (self.input.unsqueeze(-1) == self.trigger_set.unsqueeze(1)).any(-1) ) # shape (B, L)
        self.all = torch.ones_like(self.input, dtype=torch.bool) # shape (B, L)
        self.seq_len = self.input.size(1)

        self.model = model
        self.rank = model.rank
        self.vocab_size = model.vocab_size
        self.beta = model.beta



        
        self.d_model = model.d_model
        self.loss_fn = loss_fn
        self.P_b = P_b.to(device) # shape (V, V)
        self.P_u = P_u.to(device) # shape (V,)

        with torch.no_grad():
            self.logits = model(self.input, self.mask, path='full') # shape (B, L, V)
            self.logits_bigram = model(self.input, self.mask, path='bigram') # shape (B, L, V)
            self.logits_induction = model(self.input, self.mask, path='induction') # shape (B, L, V)

            self.model_prob = torch.softmax(self.logits, dim=-1) # shape (B, L, V)
            self.model_prob_bigram = torch.softmax(self.logits_bigram, dim=-1) # shape (B, L, V)
            self.std_logits = self.logits.std().item()
        
        # Masks for on-target and off-target logits
        # generate a mask to apply to the logits (shape(B,L,V)) such that each element (b,l,v) is true
        # it the input token at position l in batch b is a trigger and the counts for that trigger are larger than 1
        # and the corresponding output token is v. This mask will be used to compute the on-target and off-target logits for each trigger token in the batch.

        on_target_mask = batch["is_trigg"].bool().unsqueeze(-1) & (batch["counts"] > 1).unsqueeze(-1) 
        on_target_mask = on_target_mask & (torch.arange(self.vocab_size, device=device).view(1, 1, -1) == batch["sequence"][:, 1:].unsqueeze(-1))


        # Matrices of the model
        
        self.pos = model.embed.P.weight.data # shape (L, d)
        self.E = model.embed.E.weight.data # shape (V, d)
        
        WQ = model.attn1.WQ.weight.data # shape (r, d)
        WK = model.attn1.WK.weight.data # shape (r, d)
        self.WQK1 = WQ.t() @ WK # shape (d, d)
        # self.WQK1 = model.attn1.WQK.weight.data.T # shape (d, d)
        self.P_WQK1_P = self.pos @ self.WQK1 @ self.pos.t() / math.sqrt(self.rank) # shape (L, L)
        self.mask_sub_diagonal = torch.diag(torch.ones(self.seq_len-1, dtype=torch.bool), diagonal=-1)
        self.mask_tril = torch.tril(torch.ones((self.seq_len, self.seq_len), dtype=torch.bool))
        self.mask_sub_diagonal_tril = self.mask_tril & ~self.mask_sub_diagonal

        WV = model.attn1.WV.weight.data # shape (r, d)
        WO = model.attn1.WO.weight.data # shape (d, r)
        self.WOV1 = WO @ WV # shape (d, d)
        # self.WOV1 = model.attn1.WOV.weight.data # shape (d, d)
        
        WQ = model.attn2.WQ.weight.data # shape (r, d)
        WK = model.attn2.WK.weight.data # shape (r, d)
        self.WQK2 = WQ.t() @ WK # shape (d, d)
        # self.WQK2 = model.attn2.WQK.weight.data # shape (d, d)

        self.M = self.WQK2 #@ self.WOV1 # shape (d, d)
        self.E_M_E = self.E @ self.M @ self.E.t() / math.sqrt(self.rank) # shape (V, V) 

        self.mask_trigg_trigg = torch.zeros((self.vocab_size, self.vocab_size), dtype=torch.bool, device=device)
        self.mask_trigg_trigg[self.trigger_set_unique, self.trigger_set_unique] = True
        self.mask_trigg_nontrigg = ~self.mask_trigg_trigg

        WV = model.attn2.WV.weight.data # shape (r, d)
        WO = model.attn2.WO.weight.data # shape (d, r)
        self.WOV2 = WO @ WV # shape (d, d)
        # self.WOV2 = model.attn2.WOV.weight.data # shape (d, d)
        self.U = model.unembed.U.weight.data # shape (V, d)
        self.U_WOV2_E = self.U @ self.WOV2 @ self.E.t()  # shape (V, V)

        self.mask_tok_tok = torch.diag(torch.ones(self.vocab_size, dtype=torch.bool), diagonal=0)
        self.mask_tok_nontok = ~self.mask_tok_tok

        


class EvalContextLogits:
    def __init__(self, model, batch, loss_fn, P_b=None, P_u=None):
        device = next(model.parameters()).device

        # Ground Variables
        self.sequence = batch['sequence'].to(device) # shape (B, L+1)
        self.input = self.sequence[:, :-1] # shape (B, L)
        self.target = self.sequence[:, 1:] # shape (B, L)
        self.is_trigg = batch['is_trigg'].to(device) # shape (B, L)
        self.counts = batch['counts'].to(device) # shape (B, L)
        self.mask = batch['mask'].to(device) # shape (B, L, L)
        self.trigger_set = batch['trigger_set'].to(device) # shape (B, K)
        self.trigger_set_unique = self.trigger_set[0] # shape (K,)
        self.only_triggers = (self.input.unsqueeze(-1) == self.trigger_set.unsqueeze(1)).any(-1) & (self.counts >= 2) # shape (B, L)
        self.only_non_triggers = ~( (self.input.unsqueeze(-1) == self.trigger_set.unsqueeze(1)).any(-1) ) # shape (B, L)
        self.all = torch.ones_like(self.input, dtype=torch.bool) # shape (B, L)
        
        self.rank = model.rank
        self.vocab_size = model.vocab_size
        self.seq_len = model.seq_len

        self.loss_fn = loss_fn  



        with torch.no_grad():
            self.logits = model(self.input, self.mask) # shape (B, L, V)

        # Masks for on-target and off-target logits
        # generate a mask to apply to the logits (shape(B,L,V)) such that each element (b,l,v) is true
        # it the input token at position l in batch b is a trigger and the counts for that trigger are larger than 1
        # and the corresponding output token is v. This mask will be used to compute the on-target and off-target logits for each trigger token in the batch.
        self.on_target_mask , self.off_target_mask, _ = on_off_masks(self.logits.shape, batch, device=device)

class IC_TopKAccuracy:
    def __init__(self, k):
        self.k = k
        self.name = f"top{self.k}_accuracy"

    def __call__(self, ctx):
        logits = ctx.logits[ctx.only_triggers]  # shape (num_masked_positions, vocab_size)
        targets = ctx.target[ctx.only_triggers] # shape (num_masked_positions,)

        topk = logits.topk(self.k, dim=-1).indices # shape (num_masked_positions, k)
        correct = (topk == targets.unsqueeze(-1)).any(dim=-1)

        return correct.float().mean().item()

class KLMetric:
    def __init__(self, name = 'kl_b_full', P_fn = lambda ctx: ctx.P_b[ctx.input] , Q_fn = lambda ctx: ctx.model_prob):
        """
        P_fn, Q_fn: functions that take ctx and return distributions
        """
        self.name = name
        self.P_fn = P_fn
        self.Q_fn = Q_fn

    def __call__(self, ctx):
        P = self.P_fn(ctx)
        Q = self.Q_fn(ctx)

        kl = compute_kl(P, Q)
        return kl.item()


class LossMetric:
    def __init__(self, name = 'loss', logits_fn = lambda ctx: ctx.logits[ctx.all], target_fn = lambda ctx: ctx.target[ctx.all], rescale=False):
        """
        logits_fn: function(ctx) -> logits tensor (num_masked_positions,V)
        target_fn: function(ctx) -> target tensor (num_masked_positions,)
        rescale: whether to normalize logits to match global std
        """
        self.name = name
        self.logits_fn = logits_fn
        self.target_fn = target_fn
        self.rescale = rescale

    def __call__(self, ctx):
        logits_masked = self.logits_fn(ctx)
        targets_masked = self.target_fn(ctx)

        # Optional rescaling
        if self.rescale:
            std_global = ctx.std_logits
            std_masked = logits_masked.std()
            logits_masked = logits_masked * (std_global / std_masked)
    
        # Compute loss
        return ctx.loss_fn(logits_masked, targets_masked).item()

class LogitMeanMetric:
    def __init__(self, name = 'logit', logits_fn = lambda ctx: ctx.logits[ctx.all]):
        self.name = name
        self.logits_fn = logits_fn

    def __call__(self, ctx):
        logits_masked = self.logits_fn(ctx)
        
        return logits_masked.mean().item()

class LogitStdMetric:
    def __init__(self, name = 'logit_std', logits_fn = lambda ctx: ctx.logits[ctx.all]):
        self.name = name
        self.logits_fn = logits_fn

    def __call__(self, ctx):
        logits_masked = self.logits_fn(ctx)
        
        return logits_masked.std().item()

# On-Off target logits metrics

class OnOffLogitsMetric:
    def __init__(self,
                 name = 'on_target_mean',
                 mask_fn = lambda ctx: ctx.on_target_mask,
                 type = 'mean'):
        self.name = name
        self.mask_fn = mask_fn
        self.type = type
    
    def __call__(self, ctx):
        mask = self.mask_fn(ctx)
        logits_masked = ctx.logits[mask] 

        if self.type == 'mean':
            return logits_masked.mean().item()
        elif self.type == 'std':
            return logits_masked.std().item()
        else:
            raise ValueError(f"Unknown type {self.type} for OnOffLogitsMetric")
        

# Order Parameters

class OrderParameterMetric:
    def __init__(self, 
                 name = 'q', 
                 values_fn = lambda ctx: ctx.P_WQK1_P, 
                 mask_fn = lambda ctx: ctx.mask_sub_diagonal,
                 type = 'mean',
                 constant_factor = 1.0
                 ):
        self.name = name
        self.values_fn = values_fn
        self.mask_fn = mask_fn
        self.type = type
        self.constant_factor = constant_factor
    def __call__(self, ctx):
        values = self.values_fn(ctx)
        mask = self.mask_fn(ctx)

        masked_values = values[mask]

        if self.type == 'mean':
            return masked_values.mean().item() * self.constant_factor
        elif self.type == 'std':
            return masked_values.std().item() * self.constant_factor
        else:
            raise ValueError(f"Unknown type {self.type} for OrderParameterMetric")



class M1:
    def __init__(self):
        self.name = 'm1'
    def __call__(self, ctx):
        # ── m1 : average p_s^T WQK1 p_{s-1} over adjacent pairs ─────────────────
        WQK1 = ctx.WQK1 # shape (d, d)
        P = ctx.pos # shape (L, d)
        idx = torch.arange(1, ctx.seq_len, device=P.device) # shape (L-1,)
        p_s   = P[idx]                     # (L-1, d)
        p_sm1 = P[idx - 1]                 # (L-1, d)
        # p_s^T A1 p_{s-1} for each pair, then average
        m_vals = (p_s @ WQK1 * p_sm1).sum(dim=-1) # shape (L-1,)
        m = m_vals.mean()
        # return m.item() / (math.sqrt(ctx.d_model))    # Divided by sqrt(rank) to make it comparable across ranks
        return m.item() / math.sqrt(ctx.rank)


class center1:
    def __init__(self):
        self.name = 'center1'
    def __call__(self, ctx):
        WQK1 = ctx.WQK1 # shape (d, d)
        trace = torch.trace(WQK1)
        const = (1/(ctx.seq_len+1))*(4 + (ctx.seq_len-1)/ctx.vocab_size)
        return trace.item() * const
        
class Eta1:
    def __init__(self,mode='norm'):
        self.name = 'eta1'
        self.mode = mode
    def __call__(self, ctx):
        if self.mode == 'norm':
            # ── sigma1 : ||WQK1||_F / sqrt(r) ───────────────────────────────────────
            WQK1 = ctx.WQK1 # shape (d, d)
            WWT = WQK1 @ WQK1.t()
            trace1 = torch.trace(WWT)
            # WW = WQK1 @ WQK1
            # trace2 = torch.trace(WW)
            # const = (1/(ctx.seq_len+1))*(6+ (ctx.seq_len-1)/ctx.vocab_size)
            # variance  = trace1 #+ const*trace2
            # sigma1 = math.sqrt(variance)
            # sigma1 = WQK1.norm(p='fro') 
            # return 4*sigma1.item() / math.sqrt(ctx.d_model)
            return math.sqrt(trace1) / math.sqrt(ctx.rank)
        elif self.mode == 'std':
            # ── sigma1 : std of p_s^T WQK1 p_{s-1} random pairs ───────────────────────────────────────
            WQK1 = ctx.WQK1 # shape (d, d)
            P = ctx.pos # shape (L, d)
            idx = torch.randperm(ctx.seq_len - 1, device=P.device) # shape (L-1,)
            p_s   = P[idx]                     # (L-1, d)
            idx = torch.randperm(ctx.seq_len - 1, device=P.device) # shape (L-1,)
            p_s_prime = P[idx]                 # (L-1, d)
            # p_s^T A1 p_{s'}m1 for each pair, then take std
            m_vals = (p_s @ WQK1 * p_s_prime).sum(dim=-1) # shape (L-1,)
            sigma1 = m_vals.std()
            return sigma1.item() / ctx.d_model
        else:
            raise ValueError(f"Unknown mode {self.mode} for Eta1")


class M2:
    def __init__(self):
        self.name = 'm2'
    def __call__(self, ctx):
        # ── q : average e_t^T M e_t over t in trigger_set_unique ─────────────────
        M = ctx.M                                # (d, d)
        # M = ctx.WQK2
        # WOV1 = ctx.WOV1                          # (d, d)
        # normWOV1 = WOV1.norm(p='fro') 
        idx = ctx.trigger_set_unique # shape (K,)
        e_t = ctx.E[idx]           # shape (K, d)
        q_vals = (e_t @ M * e_t).sum(dim=-1) # shape (K,)
        q = q_vals.mean()
        # q = q / (normWOV1 * M.size(0))
        return 0.5*q.item()/ math.sqrt(ctx.rank)
        # return q.item() / math.sqrt(ctx.d_model)


class Eta2:
    def __init__(self):
        self.name = 'eta2'
    def __call__(self, ctx):
        # ── eta = ||M||_F / sqrt(d) ───────────────────────────────────────────────
        WQK2 = ctx.WQK2 # shape (d, d)
        # WOV1 = ctx.WOV1                          # (d, d)
        # normWOV1 = WOV1.norm(p='fro') 
        eta = WQK2.norm(p='fro')
        return eta.item()/math.sqrt(4*ctx.rank)
        # return eta.item() * math.sqrt(ctx.d_model)/16
    
class Gamma:
    def __init__(self):
        self.name = 'gamma'
    def __call__(self, ctx):
        # ── gamma : average u_t^T WOV2 e_t over t in vocabulary ─────────────────
        WOV2 = ctx.WOV2 
        U = ctx.U # shape (V, d)
        E = ctx.E # shape (V, d)
        gamma_vals = (U @ WOV2 * E).sum(dim=-1) # shape (V,)
        gamma = gamma_vals.mean()
        return gamma.item()#/U.size(1) # normalize by d

class Eta_Gamma:
    def __init__(self):
        self.name = 'eta_gamma'
    def __call__(self, ctx):
        # ── eta = ||M||_F / sqrt(d) ───────────────────────────────────────────────
        WOV2 = ctx.WOV2 # shape (d, d)
        # WOV1 = ctx.WOV1                          # (d, d)
        # normWOV1 = WOV1.norm(p='fro') 
        eta = WOV2.norm(p='fro')
        return ctx.beta*eta.item()/math.sqrt(2*ctx.d_model)
        # return eta.item() * math.sqrt(ctx.d_model)/16       

class Evaluator:
    def __init__(self, metrics):
        self.metrics = metrics

    def evaluate(self, model, batch, loss_fn, P_b, P_u):
        model.eval()

        ctx = EvalContext(model, batch, loss_fn, P_b, P_u)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric(ctx)

        return results
    

class EvaluatorLogits:
    def __init__(self, metrics):
        self.metrics = metrics

    def evaluate(self, model, batch,loss_fn):
        model.eval()

        ctx = EvalContextLogits(model, batch,loss_fn)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric(ctx)

        return results
    

