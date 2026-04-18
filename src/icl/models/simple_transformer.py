from torch import nn
import torch
import math


class EmbeddingModule(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model):
        super().__init__()
        self.E = nn.Embedding(vocab_size, d_model)
        self.P = nn.Embedding(seq_len, d_model)
    
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return self.E(x) + self.P(positions)

class AttentionLayer(nn.Module):
    def __init__(self, d_model, dropout=0.0, lin_attn=False):
        super().__init__()
        self.WQK = nn.Linear(d_model, d_model, bias=False)
        self.WOV = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.lin_attn = lin_attn

    def forward(self, X, mask):
        S = self.WQK(X) @ X.transpose(-2, -1)
        if self.lin_attn:
            A = S.masked_fill(~mask, 0.0)/math.sqrt(X.size(-1))
        else:
            S = S*math.sqrt(X.size(-1))
            S = S.masked_fill(~mask, float('-inf'))
            A = S.softmax(dim=-1)

        A = self.dropout(A)
        Y = A @ self.WOV(X)
        return X + Y, A
    

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.WF = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, X):
        Y = self.WF(X)
        return X + Y, Y


class UnembeddingModule(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.U = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, X):
        return self.U(X)
    

class DualModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model
        self.drop = args.dropout
        self.lin_attn = args.lin_attn

        self.embed = EmbeddingModule(self.vocab_size, self.seq_len, self.d_model)
        self.attn1 = AttentionLayer(self.d_model, self.drop, self.lin_attn)
        self.attn2 = AttentionLayer(self.d_model, self.drop, self.lin_attn)
        self.ff = FeedForwardLayer(self.d_model)
        self.unembed = UnembeddingModule(self.d_model, self.vocab_size)

    def forward(self, x, mask, path="full"):
        """ path can be "full", "induction" or "bigram" """
        X0 = self.embed(x)

        if path in ["induction", "full"]:
            X1, _ = self.attn1(X0, mask)
            X2, _ = self.attn2(X1, mask)
        else:
            X2 = X0

        if path in ["bigram", "full"]:
            X3, _ = self.ff(X2)
        else:
            X3 = X2

        logits = self.unembed(X3)
        return logits
    
    def full_output(self, x, mask, path="full"):
        out = {}
        X0 = self.embed(x)
        out['X0'] = X0
        if path in ["induction", "full"]:
            X1, A1 = self.attn1(X0, mask)
            out['X1'], out['A1'] = X1, A1
            X2, A2 = self.attn2(X1, mask)
            out['X2'], out['A2'] = X2, A2
        else:
            X2 = X0
            out['X2'] = X2
        if path in ["bigram", "full"]:
            X3, Y3 = self.ff(X2)
            out['X3'], out['Y3'] = X3, Y3
        else:
            X3 = X2
            out['X3'] = X3
        out['logits'] = self.unembed(X3)
        return out
    
   
def initialize_model(model,path="full"):
    """ 
    Initialize model as ~ N(0, 1/sqrt(d_model)) and freeze: 
    [ embedding, positional embedding, unembedding, VO projection of first attention]
    """

    # Initialize and freeze them all at first
    for param in model.parameters():
        param.data.copy_(torch.randn_like(param) / math.sqrt(model.d_model))
        param.requires_grad = False

    # Unfreeze depending on the path
    if path == "full":
        model.attn1.WQK.weight.requires_grad = True
        model.attn2.WQK.weight.requires_grad = True
        model.attn2.WOV.weight.requires_grad = True
        model.ff.WF.weight.requires_grad = True
    elif path == "induction":
        model.attn1.WQK.weight.requires_grad = True
        model.attn2.WQK.weight.requires_grad = True
        model.attn2.WOV.weight.requires_grad = True
    elif path == "bigram":
        model.ff.WF.weight.requires_grad = True
    elif path == "full_trigg":
        model.attn1.WQK.weight.requires_grad = True
        model.attn2.WQK.weight.requires_grad = True
        model.attn2.WOV.weight.requires_grad = True
        model.ff.WF.weight.requires_grad = True
    else:
        raise ValueError("Invalid path. Options are 'full', 'induction', and 'bigram'.")
    
    return model
