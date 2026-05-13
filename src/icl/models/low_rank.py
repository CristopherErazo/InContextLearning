from torch import nn
import torch
import math


class EmbeddingModule(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model):
        super().__init__()
        self.E = nn.Embedding(vocab_size, d_model)
        self.P = nn.Embedding(seq_len, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return (self.E(x) + self.P(positions))

class AttentionLayer(nn.Module):
    def __init__(self, d_model, rank, dropout=0.0, lin_attn=False):
        super().__init__()
        self.WQ = nn.Linear(d_model, rank, bias=False)
        self.WK = nn.Linear(d_model, rank, bias=False)
        self.WV = nn.Linear(d_model, rank, bias=False)
        self.WO = nn.Linear(rank, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.lin_attn = lin_attn
        self.rank = rank

    def forward(self, X, mask):
        """X.shape = (B,L,d)"""
        q = self.WQ(X) #(B,L,r)
        k = self.WK(X) #(B,L,r)
        v = self.WV(X) #(B,L,r)
        # (B,L,r) @ (B,r,L) --> (B,L,L)
        S = q @ k.transpose(-2, -1)
        if self.lin_attn:
            A = S.masked_fill(~mask, 0.0)/math.sqrt(X.size(-1))
        else:
            S = S/math.sqrt(self.rank)
            S = S.masked_fill(~mask, float('-inf'))
            A = S.softmax(dim=-1)

        A = self.dropout(A)
        Y = A @ v #(B,L,r) 
        Y = self.WO(Y) #(B,L,d)
        return X + Y, A
    

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, rank):
        super().__init__()
        self.WF_down = nn.Linear(d_model, rank, bias=False)
        self.WF_up = nn.Linear(rank, d_model, bias=False)
    
    def forward(self, X):
        Y = self.WF_down(X)
        Y = self.WF_up(Y)
        return X + Y, Y


class UnembeddingModule(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.U = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, X):
        return self.U(X)
    

class LowRankTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model
        self.drop = args.dropout
        self.lin_attn = args.lin_attn
        self.rank = args.rank

        self.embed = EmbeddingModule(self.vocab_size, self.seq_len, self.d_model)
        self.attn1 = AttentionLayer(self.d_model, self.rank, self.drop, self.lin_attn)
        self.attn2 = AttentionLayer(self.d_model, self.rank, self.drop, self.lin_attn)
        self.ff = FeedForwardLayer(self.d_model, self.rank)
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
    # Initialize E,P,U ~ N(0,1), WO ~N(0,1/sqrt(r) and the rest ~ N(0,1/sqrt(d_model))
    for name, param in model.named_parameters():
        if "embed" in name or "unembed" in name:
            param.data.copy_(torch.randn_like(param))
        elif "WO" in name:
            param.data.copy_(torch.randn_like(param) / math.sqrt(model.rank))
        else:
            param.data.copy_(torch.randn_like(param) / math.sqrt(model.d_model))
        param.requires_grad = False

    # Unfreeze depending on the path
    if path == "full":
        model.attn1.WQ.weight.requires_grad = True
        model.attn1.WK.weight.requires_grad = True
        model.attn2.WQ.weight.requires_grad = True
        model.attn2.WK.weight.requires_grad = True
        model.attn2.WV.weight.requires_grad = True
        model.attn2.WO.weight.requires_grad = True
        model.ff.WF_down.weight.requires_grad = True
        model.ff.WF_up.weight.requires_grad = True
    elif path == "induction":
        model.attn1.WQ.weight.requires_grad = True
        model.attn1.WK.weight.requires_grad = True
        model.attn2.WQ.weight.requires_grad = True
        model.attn2.WK.weight.requires_grad = True
        model.attn2.WV.weight.requires_grad = True
        model.attn2.WO.weight.requires_grad = True
    elif path == "bigram":
        model.ff.WF_down.weight.requires_grad = True
        model.ff.WF_up.weight.requires_grad = True
    elif path == "full_trigg":
        model.attn1.WQ.weight.requires_grad = True
        model.attn1.WK.weight.requires_grad = True
        model.attn2.WQ.weight.requires_grad = True
        model.attn2.WK.weight.requires_grad = True
        model.attn2.WV.weight.requires_grad = True
        model.attn2.WO.weight.requires_grad = True
        model.ff.WF_down.weight.requires_grad = True
        model.ff.WF_up.weight.requires_grad = True
    else:
        raise ValueError("Invalid path. Options are 'full', 'induction', and 'bigram'.")
    
    return model
    
   
# def initialize_model(model,path="full"):
#     """ 
#     Initialize model as ~ N(0, 1/sqrt(d_model)) and freeze: 
#     [ embedding, positional embedding, unembedding, VO projection of first attention]
#     """

#     # Initialize and freeze them all at first
#     for param in model.parameters():
#         param.data.copy_(torch.randn_like(param) / math.sqrt(model.d_model))
#         param.requires_grad = False

#     # Unfreeze depending on the path
#     if path == "full":
#         model.attn1.WQ.weight.requires_grad = True
#         model.attn1.WK.weight.requires_grad = True
#         model.attn2.WQ.weight.requires_grad = True
#         model.attn2.WK.weight.requires_grad = True
#         model.attn2.WV.weight.requires_grad = True
#         model.attn2.WO.weight.requires_grad = True
#         model.ff.WF_down.weight.requires_grad = True
#         model.ff.WF_up.weight.requires_grad = True
#     elif path == "induction":
#         model.attn1.WQ.weight.requires_grad = True
#         model.attn1.WK.weight.requires_grad = True
#         model.attn2.WQ.weight.requires_grad = True
#         model.attn2.WK.weight.requires_grad = True
#         model.attn2.WV.weight.requires_grad = True
#         model.attn2.WO.weight.requires_grad = True
#     elif path == "bigram":
#         model.ff.WF_down.weight.requires_grad = True
#         model.ff.WF_up.weight.requires_grad = True
#     elif path == "full_trigg":
#         model.attn1.WQ.weight.requires_grad = True
#         model.attn1.WK.weight.requires_grad = True
#         model.attn2.WQ.weight.requires_grad = True
#         model.attn2.WK.weight.requires_grad = True
#         model.attn2.WV.weight.requires_grad = True
#         model.attn2.WO.weight.requires_grad = True
#         model.ff.WF_down.weight.requires_grad = True
#         model.ff.WF_up.weight.requires_grad = True
#     else:
#         raise ValueError("Invalid path. Options are 'full', 'induction', and 'bigram'.")
    
#     return model
