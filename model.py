import math
import torch
from FlashMHA import FlashMHA
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import numpy as np

from dejavu import DejaVuConfig

INIT_STD = 0.02
device = 'cuda'
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub


class AWGNChannel(nn.Module):
    def __init__(self, snr: float = 12):
        super(AWGNChannel, self).__init__()
        self.snr = snr
        self.snr_factor = 10 ** (self.snr / 10.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the power of the input signal
        x_power = torch.mean(x ** 2)

        # Calculate the noise power based on SNR
        n_power = x_power / self.snr_factor

        # Generate Gaussian noise with the calculated noise power
        noise = torch.randn_like(x) * torch.sqrt(n_power)

        return x + noise


class RoPE(nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        theta = 1. / (base ** (torch.arange(0, head_dim, 2) / head_dim))
        self.register_buffer('theta', theta)

    def forward(self, qk):
        # qk: batch, num_head*2, sequence, head_dim

        s = torch.arange(qk.size(2), device=qk.device)

        freqs = torch.outer(s, self.theta)  # seq_len, dim // 2
        freqs = torch.cat((freqs, freqs), dim=-1)

        qk1, qk2 = qk.chunk(2, dim=-1)
        qk2 = torch.cat((-qk2, qk1), dim=-1)

        return qk * freqs.cos() + qk2 * freqs.sin()


class RMSNorm(nn.Module):

    def __init__(self, dim_size, eps=1e-6):
        super().__init__()

        self.root_dim = math.sqrt(dim_size)
        self.weight = nn.Parameter(torch.ones(dim_size))
        self.eps = eps

    def forward(self, x):
        x = F.normalize(x, dim=-1, eps=self.eps) * self.root_dim * self.weight

        return x


class MLP(nn.Module):
    def __init__(self, input_dim, drop_rate):
        super().__init__()

        self.hidden = input_dim * 8 // 3
        self.in_proj = nn.Linear(input_dim, self.hidden * 2)
        self.out_proj = nn.Linear(self.hidden, input_dim)

        self.drop = nn.Dropout(drop_rate, True)

        torch.nn.init.normal_(self.in_proj.weight, std=INIT_STD)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        x = self.in_proj(x)
        x1, x2 = torch.chunk(x, 2, -1)
        x = F.silu(x1) * x2
        x = self.drop(x)
        x = self.out_proj(x)

        return x


class LearnedSparseAttention(nn.Module):
    def __init__(self, input_dim, num_heads, num_kv_heads=None, sparsity_factor=4, temperature=1.0):
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = input_dim // num_heads
        self.sparsity_factor = sparsity_factor
        self.temperature = temperature

        self.q_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.rope = RoPE(self.head_dim)

        # Learnable parameters for sparsity
        self.sparsity_proj = nn.Linear(self.head_dim, self.head_dim // self.sparsity_factor)

        torch.nn.init.normal_(self.q_proj.weight, std=INIT_STD)
        torch.nn.init.normal_(self.k_proj.weight, std=INIT_STD)
        torch.nn.init.normal_(self.v_proj.weight, std=INIT_STD)
        torch.nn.init.normal_(self.sparsity_proj.weight, std=INIT_STD)
        nn.init.zeros_(self.out_proj.bias)

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_kv_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_kv_heads)

        q = self.rope(q)
        k = self.rope(k)

        if self.num_kv_heads != self.num_heads:
            k = self.repeat_kv(k, self.num_heads // self.num_kv_heads)
            v = self.repeat_kv(v, self.num_heads // self.num_kv_heads)

        # Learn sparsity patterns
        q_sparse = self.sparsity_proj(q)
        k_sparse = self.sparsity_proj(k)

        # Compute sparsity scores
        sparsity_scores = torch.einsum('bhid,bhjd->bhij', q_sparse, k_sparse) / math.sqrt(
            self.head_dim // self.sparsity_factor)
        sparsity_mask = torch.sigmoid(sparsity_scores / self.temperature)

        # Compute attention scores
        attn_scores = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(self.head_dim)

        # Apply learned sparsity mask
        attn_scores = attn_scores * sparsity_mask

        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores.masked_fill_(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights
        output = torch.einsum('bhij,bhjd->bhid', attn_weights, v)

        output = rearrange(output, 'b h s d -> b s (h d)')
        output = self.out_proj(output)

        return output


class SparseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_factor=2, sparsity_factor=4, temperature=1.0, drop_rate=0.1):
        super().__init__()

        self.hidden_dim = input_dim * hidden_dim_factor
        self.sparsity_factor = sparsity_factor
        self.temperature = temperature

        self.in_proj = nn.Linear(input_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, input_dim)

        # Learnable parameters for sparsity
        self.sparsity_in = nn.Linear(input_dim, input_dim // self.sparsity_factor)
        self.sparsity_hidden = nn.Linear(self.hidden_dim, self.hidden_dim // self.sparsity_factor)

        self.drop = nn.Dropout(drop_rate)

        # Initialize weights
        torch.nn.init.normal_(self.in_proj.weight, std=INIT_STD)
        torch.nn.init.normal_(self.out_proj.weight, std=INIT_STD)
        torch.nn.init.normal_(self.sparsity_in.weight, std=INIT_STD)
        torch.nn.init.normal_(self.sparsity_hidden.weight, std=INIT_STD)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        # Compute sparsity scores for input
        sparsity_scores_in = self.sparsity_in(x)  # (batch_size, seq_len, input_dim // sparsity_factor)

        # Project input to hidden dimension
        hidden = self.in_proj(x)
        hidden = F.gelu(hidden)  # or any other activation function

        # Compute sparsity scores for hidden
        sparsity_scores_hidden = self.sparsity_hidden(hidden)  # (batch_size, seq_len, hidden_dim // sparsity_factor)

        # Compute sparsity mask
        # Reshape for batch matrix multiplication
        sparsity_scores_in = sparsity_scores_in.view(batch_size * seq_len, -1)
        sparsity_scores_hidden = sparsity_scores_hidden.view(batch_size * seq_len, -1)

        sparsity_mask = torch.bmm(sparsity_scores_in.unsqueeze(1), sparsity_scores_hidden.unsqueeze(2)).squeeze(1)
        sparsity_mask = sparsity_mask.view(batch_size, seq_len, 1)
        sparsity_mask = torch.sigmoid(sparsity_mask / self.temperature)

        # Apply sparsity mask
        hidden = hidden * sparsity_mask

        hidden = self.drop(hidden)
        output = self.out_proj(hidden)

        return output


# class Block(nn.Module):
#     def __init__(self, input_dim, num_heads, res_drop_rate, h_drop_rate,
#                  attn_sparsity_factor=4, attn_temperature=1.0,
#                  mlp_hidden_dim_factor=2, mlp_sparsity_factor=4, mlp_temperature=1.0):
#         super().__init__()
#
#         self.attn = LearnedSparseAttention(input_dim, num_heads,
#                                            sparsity_factor=attn_sparsity_factor,
#                                            temperature=attn_temperature)
#         self.mlp = SparseMLP(input_dim, hidden_dim_factor=mlp_hidden_dim_factor,
#                              sparsity_factor=mlp_sparsity_factor,
#                              temperature=mlp_temperature,
#                              drop_rate=h_drop_rate)
#
#         self.norm1 = RMSNorm(input_dim)
#         self.norm2 = RMSNorm(input_dim)
#
#         self.drop = nn.Dropout(res_drop_rate)
#
#     def forward(self, x):
#         x_out = self.attn(self.norm1(x))
#         x = self.drop(x_out) + x
#
#         x_out = self.mlp(self.norm2(x))
#         x = self.drop(x_out) + x
#
#         return x



class Attention(nn.Module):
    def __init__(self, input_dim, num_heads, num_kv_heads=None):
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = input_dim // num_heads

        self.q_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.rope = RoPE(self.head_dim)

        torch.nn.init.normal_(self.q_proj.weight, std=INIT_STD)
        torch.nn.init.normal_(self.k_proj.weight, std=INIT_STD)
        torch.nn.init.normal_(self.v_proj.weight, std=INIT_STD)
        nn.init.zeros_(self.out_proj.bias)

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_kv_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_kv_heads)

        q = self.rope(q)
        k = self.rope(k)

        if self.num_kv_heads != self.num_heads:
            k = self.repeat_kv(k, self.num_heads // self.num_kv_heads)
            v = self.repeat_kv(v, self.num_heads // self.num_kv_heads)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        x = rearrange(x, 'b h s d -> b s (h d)')
        x = self.out_proj(x)

        return x


class Block(nn.Module):

    def __init__(self, input_dim, num_heads, res_drop_rate, h_drop_rate):
        super().__init__()

        self.attn = Attention(input_dim, num_heads)
        self.mlp = MLP(input_dim, h_drop_rate)

        self.norm1 = RMSNorm(input_dim)
        self.norm2 = RMSNorm(input_dim)

        self.drop = nn.Dropout(res_drop_rate, True)

    def forward(self, x):
        x_out = self.attn(self.norm1(x))
        x = self.drop(x_out) + x

        x_out = self.mlp(self.norm2(x))
        x = self.drop(x_out) + x

        return x


# class SemanticCommunicationSystem(nn.Module):  # pure DeepSC
#     def __init__(self, vocab_size, embed_dim=128, snr=12, K=8):
#         super(SemanticCommunicationSystem, self).__init__()
#         self.snr = snr
#         self.K = K
#         self.config = DejaVuConfig(
#             vocab_size=vocab_size,
#             n_embd=embed_dim,
#             n_layer=3,
#             n_head=4,
#             residual_in_fp32=True,
#             fused_dropout_add_ln=True,
#             use_flash_attn=True,
#             fused_bias_fc=True,
#             mlp_sparse=True,
#             att_sparse=True
#         )
#
#         self.embedding = nn.Embedding(vocab_size, embed_dim)  # which means the corpus has input_size kinds of words and
#         self.encoder = nn.ModuleList([Block(embed_dim, 4, 0.1, 0.1).cuda() for _ in range(3)])
#         self.denseEncoder1 = nn.Linear(embed_dim, 256)
#         self.denseEncoder2 = nn.Linear(256, 2 * self.K)
#         self.noise_channel = AWGNChannel(snr).cuda()
#         self.denseDecoder1 = nn.Linear(2 * self.K, 256)
#         self.denseDecoder2 = nn.Linear(256, embed_dim)
#         self.decoder = nn.ModuleList([Block(embed_dim, 4, 0.1, 0.1).cuda() for _ in range(3)])
#
#         self.prediction = nn.Linear(embed_dim, vocab_size)
#         self.softmax = nn.Softmax(dim=2)  # dim=2 means that it calculates softmax in the feature dimension
#
#     def forward(self, inputs):
#         x = self.embedding(inputs)
#         for i in range(len(self.encoder)):
#             x = self.encoder[i](x)
#
#         x = self.denseEncoder1(x)
#         x = self.denseEncoder2(x)
#
#         x = self.noise_channel(x)  # assuming snr = 12db
#
#         x = self.denseDecoder1(x)
#         x = self.denseDecoder2(x)
#         for i in range(len(self.decoder)):
#             x = self.decoder[i](x)
#
#         codeSemantic = F.linear(x, self.embedding.weight)
#         info = self.softmax(codeSemantic)
#         return info
#
#     def set_snr(self, snr):
#         self.noise_channel = AWGNChannel(snr).cuda()


class OptimizedBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout_rate):
        super().__init__()
        self.attention = FlashMHA(input_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class OptimizedSemanticCommunicationSystem(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, snr=12, K=4):
        super(OptimizedSemanticCommunicationSystem, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.encoder = nn.ModuleList([OptimizedBlock(embed_dim, 2, 0.1) for _ in range(2)])
        self.channel_encoder = nn.Linear(embed_dim, 2 * K)
        self.channel_decoder = nn.Linear(2 * K, embed_dim)
        self.decoder = nn.ModuleList([OptimizedBlock(embed_dim, 2, 0.1) for _ in range(2)])
        self.prediction = nn.Linear(embed_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)  # dim=2 means that it calculates softmax in the feature dimension
        self.noise_channel = AWGNChannel(snr)

    def forward(self, inputs):
        x = self.embedding(inputs)
        for block in self.encoder:
            x = block(x)
        x = self.channel_encoder(x)
        # Simulate AWGN channel (simplified for demonstration)
        x = self.noise_channel(x)
        x = self.channel_decoder(x)
        for block in self.decoder:
            x = block(x)
        x = F.linear(x, self.embedding.weight)
        x = self.softmax(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == OptimizedBlock:
                torch.quantization.fuse_modules(m.ffn, ['0', '1', '2'], inplace=True)

    def set_snr(self, snr):
        self.noise_channel = AWGNChannel(snr)


class LossFn(nn.Module):  # Loss function
    def __init__(self):
        super(LossFn, self).__init__()

    def forward(self, output, label, length_sen):
        delta = 1e-7  # used to avoid vanishing gradient
        device = output.device
        # Create a mask for valid lengths
        max_length = output.size(1)
        mask = torch.arange(max_length).expand(len(length_sen), max_length).to(device) < length_sen.unsqueeze(1)

        # Mask the output and label
        output_masked = output[mask]
        label_masked = label[mask]

        # Compute the loss using masked values
        loss = -torch.sum(
            label_masked * torch.log(output_masked + delta)) / length_sen.float().sum()
        return loss
