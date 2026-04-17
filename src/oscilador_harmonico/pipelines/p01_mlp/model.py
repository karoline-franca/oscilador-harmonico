"""
Definição da arquitetura do MLP.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron para prever posição, velocidade e tempo.
    
    Entrada: [x0, v0, ω] - condições iniciais e frequência angular
    Saída: [x, v, t] - posição, velocidade e tempo
    """
    
    def __init__(self, input_dim=3, hidden_dims=[64, 128, 64], output_dim=3, dropout=0.1):
        """
        Args:
            input_dim: Dimensão da entrada (x0, v0, ω) = 3
            hidden_dims: Lista com dimensões das camadas ocultas
            output_dim: Dimensão da saída (x, v, t) = 3
            dropout: Taxa de dropout para regularização
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: tensor de entrada (batch_size, input_dim) - [x0, v0, ω]
            
        Returns:
            tensor de saída (batch_size, output_dim) - [x, v, t]
        """
        return self.network(x)