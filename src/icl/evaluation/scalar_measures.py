from functools import cached_property
import torch
import math
from .utils import on_off_logit_masks



class Evaluator:
    def __init__(self, metrics):
        self.metrics = metrics

    def evaluate(self, model, batch,loss_fn):
        model.eval()

        ctx = EvalContext(model, batch,loss_fn)

        results = {}
        for metric in self.metrics:
            out = metric(ctx)
            if isinstance(out, dict):
                results.update(out)
            else:
                results[metric.name] = metric(ctx)

        return results
    

class EvalContext:
    def __init__(self, model, batch, loss_fn):
        device = next(model.parameters()).device

        # DATA VARIABLES
        sequence = batch['sequence'].to(device) # shape (B, L+1)
        input = sequence[:, :-1] # shape (B, L)
        self.target = sequence[:, 1:] # shape (B, L)
        mask = batch['mask'].to(device) # shape (B, L, L)
        # masks for positions where induction is possible and not possible
        self.ind_possible = batch['ind_possible'].to(device) # shape (B, L)
        self.ind_not_possible = batch['ind_not_possible'].to(device) # shape (B, L)
        self.all = torch.ones_like(input, dtype=torch.bool) # shape (B, L)
        self.ind_target = batch['target_ind_positions'].to(device) # shape (num_masked_positions,)

        # MODEL VARIABLES
        
        self.pred_mode = model.pred_mode
        if self.pred_mode == "last":
            self.ind_possible = self.ind_possible[:, -1:]  # shape (B, 1)
            self.ind_not_possible = self.ind_not_possible[:, -1:]  # shape (B, 1)
            self.target = self.target[:, -1:]  # shape (B, 1)
            self.ind_target = self.ind_target[:, -1:]  # shape (B, 1)
        self.loss_fn = loss_fn  

        
        with torch.no_grad():
            self.logits = model(input, mask)  # shape (B, L, V) or (B, 1, V) if pred_mode == "last"
        self.logits_ind = self.logits[self.ind_possible]           # (N, V)

        self.on_target_logits = self.logits_ind.gather(1, self.ind_target[:, None]).squeeze(1)    #(N,)                                            # (N,)
        mask_off = torch.ones_like(self.logits_ind, dtype=torch.bool)
        mask_off.scatter_(1, self.ind_target[:, None], False)
        self.off_target_logits = self.logits_ind[mask_off].view(self.logits_ind.size(0), -1)     #(N,V-1)                                                      # (N, V-1)


class LossMetric:
    def __init__(self, 
                 name = 'loss', 
                 mask = 'all'
                ):
        """
        name: Name of the metric.
        mask: Which positions to consider for the loss. Options are 'all', 'ind', 'no_ind'.
        """
        self.name = name
        self.mask_name = mask
        if not mask in ['all', 'ind', 'no_ind']:
            raise ValueError(f"Invalid mask name: {mask}. Must be one of 'all', 'ind', 'no_ind'.")
    def __call__(self, ctx: EvalContext):
        if self.mask_name == "all":
            mask = ctx.all
        elif self.mask_name == "ind":
            mask = ctx.ind_possible
        elif self.mask_name == "no_ind":
            mask = ctx.ind_not_possible   

        logits_masked = ctx.logits[mask]
        targets_masked = ctx.target[mask]
    
        # Compute loss
        return ctx.loss_fn(logits_masked, targets_masked).item()

class IC_TopKAccuracy:
    def __init__(self, k):
        self.k = k
        self.name = f"top{self.k}_accuracy"

    def __call__(self, ctx: EvalContext):
        logits = ctx.logits_ind  # shape (num_masked_positions, vocab_size)
        targets = ctx.ind_target # shape (num_masked_positions,)

        topk = logits.topk(self.k, dim=-1).indices # shape (num_masked_positions, k)
        correct = (topk == targets.unsqueeze(-1)).any(dim=-1)

        return correct.float().mean().item()

class AvgAccuracy:
    def __init__(self):
        self.name = "avg_accuracy"

    def __call__(self, ctx: EvalContext):
        """
        Compute the average 'output probability mass' assigned to the correct target tokens across all positions in the batch
        """
        logits = ctx.logits_ind # shape (num_masked_positions, vocab_size)
        targets = ctx.ind_target # shape (num_masked_positions,)

        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1) # shape (num_masked_positions, vocab_size)
        # Gather the probabilities corresponding to the target tokens
        target_probs = probs.gather(1, targets.unsqueeze(-1)).squeeze(-1) # shape (num_masked_positions,)

        return target_probs.mean().item()

class LogitStatistics:
    def __init__(self):
        self.name = "logit_statistics_dummy_name"

    def __call__(self, ctx: EvalContext):
        """
        Compute  the mean and var of on-off target logits for positions where induction is possible
        and the average correlation on/off logits
        """
        on_target_logits = ctx.on_target_logits # shape (num_masked_positions,)
        off_target_logits = ctx.off_target_logits # shape (num_masked_positions, vocab_size-1)

        mean_on = on_target_logits.mean().item()
        var_on = on_target_logits.var().item() # the unbia

        mean_off = off_target_logits.mean().item()
        var_off = off_target_logits.var().item()

        # Compute correlation between on-target and off-target logits
        # For each position, compute the mean of the off-target logits
        logit_product = (on_target_logits[:, None]-mean_on)*(off_target_logits-mean_off) # shape (num_masked_positions, vocab_size-1)
        correlation = logit_product.mean().item()
        return {
            "on_logit_mean": mean_on,
            "on_logit_var": var_on,
            "off_logit_mean": mean_off,
            "off_logit_var": var_off,
            "on_off_correlation": correlation
        }