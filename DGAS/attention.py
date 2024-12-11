import torch.nn as nn
import torch
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Self-attention.
    """

    def __init__(self, factor, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask):
        attn_weight = torch.matmul(q, k.transpose(2, 3)) / self.factor

        if attn_mask is not None:
            attn_weight = attn_weight.masked_fill(attn_mask == 0, -1e9)

        attn_weight = self.dropout(F.softmax(attn_weight, dim=-1))

        context = torch.matmul(attn_weight, v)

        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_heads * d_k)
        self.w_k = nn.Linear(d_model, n_heads * d_k)
        self.w_v = nn.Linear(d_model, n_heads * d_v)
        self.fc = nn.Linear(n_heads * d_v, d_model)

        self.attention = ScaledDotProductAttention(d_k ** -0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, padding_mask=None):

        n_heads, dk, dv = self.n_heads, self.d_k, self.d_v
        batch_size, seq_q, seq_k, seq_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_q(q).view(batch_size, seq_q, n_heads, dk)
        k = self.w_k(k).view(batch_size, seq_k, n_heads, dk)
        v = self.w_v(v).view(batch_size, seq_v, n_heads, dv)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1)

        context = self.attention(q, k, v, padding_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_q, -1)
        context = self.fc(context)
        context = self.layer_norm(context)
        context = self.dropout(context)

        output = context + residual

        return output

