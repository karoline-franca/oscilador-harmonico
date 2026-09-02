"""
Definição da arquitetura do MLP para previsão de trajetórias completas do oscilador de FitzHugh-Nagumo.
"""

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron para prever trajetórias completas do oscilador de FitzHugh-Nagumo.
    
    Entrada: [v0, w0] - condições iniciais (potencial, recuperação)
    Saída: [v_0, w_0, v_1, w_1, ..., v_N, w_N] - trajetória completa
    """

    def __init__(self, input_dim=2, hidden_dims=[64, 128, 256], output_dim=None, 
                 num_timesteps=127, activation='relu', seed=None):
        """
        Args:
            input_dim: Dimensão da entrada (2: v0, w0) - [potencial inicial, recuperação inicial]
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
            x: tensor de entrada (batch_size, input_dim) - [v0, w0] (potencial inicial, recuperação inicial)
            
        Returns:
            tensor de saída (batch_size, output_dim) - [v_0, w_0, v_1, w_1, ..., v_N, w_N]
        """
        return self.network(x)
    
    def get_trajectory(self, v0, w0, tempos=None):
        """
        Obtém a trajetória completa para uma condição inicial.
        
        Args:
            v0: Potencial inicial (float)
            w0: Recuperação inicial (float)
            tempos: Array com os tempos (opcional, apenas para referência)
            
        Returns:
            dict com 'potencial', 'recuperacao' e 'tempos'
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor([[v0, w0]], dtype=torch.float32)
            output = self.forward(x).numpy().flatten()
        
        # separa potencial e recuperação (intercalados)
        potencial = output[0::2]
        recuperacao = output[1::2]
        
        # se tempos não foi fornecido, cria um array baseado no número de pontos
        if tempos is None:
            tempos = np.arange(self.num_timesteps)
        elif len(tempos) != self.num_timesteps:
            tempos = np.linspace(tempos[0], tempos[-1], self.num_timesteps)
        
        return {
            'potencial': potencial,
            'recuperacao': recuperacao,
            'tempos': tempos
        }
    
    def get_trajectories_batch(self, v0_batch, w0_batch, tempos=None):
        """
        Obtém trajetórias completas para múltiplas condições iniciais.
        
        Args:
            v0_batch: Array de potenciais iniciais
            w0_batch: Array de recuperações iniciais
            tempos: Array com os tempos (opcional)
            
        Returns:
            dict com 'potencial', 'recuperacao' e 'tempos'
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(np.column_stack([v0_batch, w0_batch]), dtype=torch.float32)
            outputs = self.forward(x).numpy()
        
        # separa potencial e recuperação para cada trajetória
        potencial = outputs[:, 0::2]
        recuperacao = outputs[:, 1::2]
        
        if tempos is None:
            tempos = np.arange(self.num_timesteps)
        elif len(tempos) != self.num_timesteps:
            tempos = np.linspace(tempos[0], tempos[-1], self.num_timesteps)
        
        return {
            'potencial': potencial,
            'recuperacao': recuperacao,
            'tempos': tempos
        }