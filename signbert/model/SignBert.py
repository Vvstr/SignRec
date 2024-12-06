import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __int__(self, embed_size, heads, dropout=0, forward_expansion):
        super(EncoderBlock).__init__()
        self.attention = SelfAttention(embed_size, heads)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, *args, **kwargs):
        super(SelfAttention).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
