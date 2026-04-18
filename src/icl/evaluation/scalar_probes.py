import torch


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
        self.counts = batch['counts'].to(device) # shape (B, L)
        self.mask = batch['mask'].to(device) # shape (B, L, L)
        self.trigger_set = batch['trigger_set'].to(device) # shape (B, K)
        self.only_triggers = (self.input.unsqueeze(-1) == self.trigger_set.unsqueeze(1)).any(-1) & (self.counts >= 2) # shape (B, L)
        self.only_non_triggers = ~( (self.input.unsqueeze(-1) == self.trigger_set.unsqueeze(1)).any(-1) ) # shape (B, L)
        self.all = torch.ones_like(self.input, dtype=torch.bool) # shape (B, L)
        
        self.model = model
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
    
