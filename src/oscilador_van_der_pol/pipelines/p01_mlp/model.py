"""
Definição da arquitetura do MLP para previsão de trajetórias completas do oscilador de Van der Pol.
"""

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron para prever trajetórias completas do oscilador de Van der Pol.
    
    Entrada: [x0, y0] - condições iniciais (posição, velocidade)
    Saída: [x_0, y_0, x_1, y_1, ..., x_N, y_N] - trajetória completa
    """

    def __init__(self, input_dim=2, hidden_dims=[64, 128, 256], output_dim=None, 
                 num_timesteps=127, activation='relu', seed=None):
        """
        Args:
            input_dim: Dimensão da entrada (2: x0, y0) - [posição inicial, velocidade inicial]
            hidden_dims: Dimensões das camadas ocultas
            output_dim: Dimensão da saída (2 * num_timesteps)
            num_timesteps: Número de instantes de tempo na trajetória
            activation: Função de ativação
            seed: Semente para reprodutibilidade
        """
        super(MLP, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        # output_dim a partir do número de timesteps
        if output_dim is None:
            output_dim = 2 * num_timesteps
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps

        activation_functions = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
        }
        
        self.activation = activation_functions.get(activation.lower(), nn.ReLU())
        
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
            x: tensor de entrada (batch_size, input_dim) - [x0, y0] (posição inicial, velocidade inicial)
            
        Returns:
            tensor de saída (batch_size, output_dim) - [x_0, y_0, x_1, y_1, ..., x_N, y_N]
        """
        return self.network(x)
    
    def get_trajectory(self, x0, y0, tempos=None):
        """
        Obtém a trajetória completa para uma condição inicial.
        
        Args:
            x0: Posição inicial (float)
            y0: Velocidade inicial (float)
            tempos: Array com os tempos (opcional, apenas para referência)
            
        Returns:
            dict com 'posicao', 'velocidade' e 'tempos'
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor([[x0, y0]], dtype=torch.float32)
            output = self.forward(x).numpy().flatten()
        
        # separa posição e velocidade (intercalados)
        posicao = output[0::2]
        velocidade = output[1::2]
        
        # se tempos não foi fornecido, cria um array baseado no número de pontos
        if tempos is None:
            tempos = np.arange(self.num_timesteps)
        elif len(tempos) != self.num_timesteps:
            tempos = np.linspace(tempos[0], tempos[-1], self.num_timesteps)
        
        return {
            'posicao': posicao,
            'velocidade': velocidade,
            'tempos': tempos
        }
    
    def get_trajectories_batch(self, x0_batch, y0_batch, tempos=None):
        """
        Obtém trajetórias completas para múltiplas condições iniciais.
        
        Args:
            x0_batch: Array de posições iniciais
            y0_batch: Array de velocidades iniciais
            tempos: Array com os tempos (opcional)
            
        Returns:
            dict com 'posicao', 'velocidade' e 'tempos'
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(np.column_stack([x0_batch, y0_batch]), dtype=torch.float32)
            outputs = self.forward(x).numpy()
        
        # separa posição e velocidade para cada trajetória
        posicao = outputs[:, 0::2]
        velocidade = outputs[:, 1::2]
        
        if tempos is None:
            tempos = np.arange(self.num_timesteps)
        elif len(tempos) != self.num_timesteps:
            tempos = np.linspace(tempos[0], tempos[-1], self.num_timesteps)
        
        return {
            'posicao': posicao,
            'velocidade': velocidade,
            'tempos': tempos
        }