import torch
import torch.nn as nn

class FeatureTransformer(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, hidden_size, d_model=2048):
        super(FeatureTransformer, self).__init__()
        # Flatten the image and project to d_model
        self.flatten = nn.Flatten()  # Flatten the spatial dimensions
        self.input_projection = nn.Linear(input_size, d_model)  # Project to d_model
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, hidden_size)

    def forward(self, x):
        # Flatten the input (batch_size, height, width) -> (batch_size, height * width)
        x = self.flatten(x)
        
        # Project to d_model
        x = self.input_projection(x)
        
        # Add sequence length dimension (batch_size, sequence_length=1, d_model)
        x = x.unsqueeze(1)
        
        # Forward pass through Transformer
        x = self.transformer_encoder(x)
        
        # Use the output of the last timestep
        x = self.fc(x[:, -1, :])
        return x