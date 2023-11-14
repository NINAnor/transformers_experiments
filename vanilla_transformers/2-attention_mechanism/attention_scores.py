import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Example sentence
sentence = "The quick brown fox"
tokens = sentence.lower().split()

# Assuming a simple vocabulary where each word is assigned a unique number
vocab = {word: idx for idx, word in enumerate(set(tokens))}
embed_dim = 512  # Embedding dimension

# Tokenize and encode the sentence
token_ids = torch.tensor([vocab[word] for word in tokens], dtype=torch.long)

# Embedding layer
embedding_layer = nn.Embedding(len(vocab), embed_dim)
embeddings = embedding_layer(token_ids)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

pos_encoder = PositionalEncoding(embed_dim)
position_encoded = pos_encoder(embeddings.unsqueeze(1))

# Scaled Dot-Product Attention
def scaled_dot_product_attention(query, key, value):
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    d_k = query.size(-1)
    scaled_attention_logits = matmul_qk / math.sqrt(d_k)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

# Assume Q, K, V are the same for simplicity
Q = K = V = position_encoded.squeeze(1)

# Computing Attention
attention_output, attention_weights = scaled_dot_product_attention(Q, K, V)

print("Attention Output:", attention_output)
print("Attention Weights:", attention_weights)
