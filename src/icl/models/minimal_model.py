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

        e = self.E(x)
        p = self.P(positions).unsqueeze(0).expand(x.size(0), -1, -1)
        return e, p # shapes = (B,L,d)

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

    def forward(self, Q,K,V, mask):
        """Q,K,V.shape = (B,L,d)"""
        q = self.WQ(Q) #(B,L,r)
        k = self.WK(K) #(B,L,r)
        v = self.WV(V) #(B,L,r)
        # (B,L,r) @ (B,r,L) --> (B,L,L)
        S = q @ k.transpose(-2, -1)/math.sqrt(self.rank)
        if self.lin_attn:
            A = S.masked_fill(~mask, 0.0)/math.sqrt(self.rank) #(B,L,L)
        else:
            A = (S.masked_fill(~mask, float('-inf'))).softmax(dim=-1)

        A = self.dropout(A)
        Y = A @ v #(B,L,r) 
        Y = self.WO(Y)/math.sqrt(2) #(B,L,d)
        return Y, A, S


class UnembeddingModule(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.U = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, X):
        return self.U(X)
    

class MinimalTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model
        self.drop = args.dropout
        self.lin_attn = args.lin_attn
        self.rank = args.rank
        self.beta = args.beta

        self.embed = EmbeddingModule(self.vocab_size, self.seq_len, self.d_model)
        self.attn1 = AttentionLayer(self.d_model, self.rank, self.drop, self.lin_attn)
        self.attn2 = AttentionLayer(self.d_model, self.rank, self.drop, self.lin_attn)
        self.unembed = UnembeddingModule(self.d_model, self.vocab_size)

    def forward(self, x, mask, path="full"):
        
        e , p = self.embed(x)
        X1, _ , _ = self.attn1(p, p, e, mask)
        X2, _ , _ = self.attn2(e, X1, e, mask)
        logits = self.unembed(X2)*self.beta
        return logits
    
    def full_output(self, x, mask, path="full"):
        out = {}
        e , p = self.embed(x)
        X1, A1 , S1 = self.attn1(p, p, e, mask)
        X2, A2 , S2 = self.attn2(e, X1, e, mask)        
        out['X1'], out['A1'], out['S1'] = X1, A1, S1
        out['X2'], out['A2'], out['S2'] = X2, A2, S2

        out['logits'] = self.unembed(X2)*self.beta
        return out


def initialize_model(model,path="full",sigma_0=1.0):
    # Initialize E,P,U ~ N(0,1), WO ~N(0,1/sqrt(r) and the rest ~ N(0,1/sqrt(d_model))
    for name, param in model.named_parameters():
        if "embed.E" in name or "embed.P" in name:
            param.data.copy_(torch.randn_like(param))
        elif "WO" in name:
            param.data.copy_(sigma_0*torch.randn_like(param) / math.sqrt(model.rank))
        else:
            param.data.copy_(sigma_0*torch.randn_like(param) / math.sqrt(model.d_model))
        param.requires_grad = False

    model.attn1.WQ.weight.requires_grad = True
    model.attn1.WK.weight.requires_grad = True
    model.attn2.WQ.weight.requires_grad = True
    model.attn2.WK.weight.requires_grad = True
    model.attn2.WV.weight.requires_grad = True
    model.attn2.WO.weight.requires_grad = True    
    return model
    
   