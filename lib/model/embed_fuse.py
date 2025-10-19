import torch
import torch.nn as nn
from functools import partial, wraps

class EmbeddingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Process the first embedding [batch_size, 257, 768]
        self.linear1 = nn.Linear(768, 1024)
        
        # Process the second embedding [batch_size, 256, 1280]
        self.linear2 = nn.Linear(1280, 1024)
        
        # Transformer encoder to handle sequence relationships
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output projection to get the desired dimension
        self.output_proj = nn.Linear(1024, 2048)
        
        # Learnable position embeddings for the output sequence
        self.pos_embedding = nn.Parameter(torch.randn(1, 77, 1024))

    def forward(self, emb1, emb2):
        # Process embeddings
        x1 = self.linear1(emb1)  # [batch_size, 257, 1024]
        x2 = self.linear2(emb2)  # [batch_size, 256, 1024]
        
        # Concatenate and take the first 77+256=333 tokens
        # (or use any merging strategy that works for your case)
        x = torch.cat([x1, x2], dim=1)[:, :333, :]  # [batch_size, 333, 1024]
        
        # Use transformer to capture relationships
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)  # [batch_size, 333, 1024]
        
        # Extract the first 77 tokens and add position embeddings
        x = x[:, :77, :] + self.pos_embedding
        
        # Project to the final dimension
        output = self.output_proj(x)  # [batch_size, 77, 2048]
        
        return output
    