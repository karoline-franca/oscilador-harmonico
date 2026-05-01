"""
Definição da arquitetura do MLP.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron para prever posição.
    
    Entrada: [x0, v0, ω, t] - condições iniciais, frequência angular e tempo
    Saída: [x] - posição
    """

    def __init__(self, input_dim=4, hidden_dims=[64, 128, 64], output_dim=1, activation='sigmoid'):
        super(MLP, self).__init__()

        activation_functions = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
        }
        
        self.activation = activation_functions.get(activation.lower(), nn.Sigmoid())
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
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
            tensor de saída (batch_size, output_dim) - [x]
        """
        return self.network(x)