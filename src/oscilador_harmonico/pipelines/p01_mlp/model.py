"""
Definição da arquitetura do MLP.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron para prever posição e velocidade.
    
    Entrada: [x0, v0, ω, t] - condições iniciais, frequência angular e tempo
    Saída: [x, v] - posição e velocidade
    """
    
    def __init__(self, input_dim=4, hidden_dims=[64, 128, 64], output_dim=2):
        """
        Args:
            input_dim: Dimensão da entrada (x0, v0, ω, t) = 4
            hidden_dims: Lista com dimensões das camadas ocultas
            output_dim: Dimensão da saída (x, v) = 2
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Sigmoid())
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
            x: tensor de entrada (batch_size, input_dim) - [x0, v0, ω, t]
            
        Returns:
            tensor de saída (batch_size, output_dim) - [x, v]
        """
        return self.network(x)