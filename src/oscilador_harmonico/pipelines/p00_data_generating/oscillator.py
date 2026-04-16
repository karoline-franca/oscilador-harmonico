"""
Classe para resolver a equação do oscilador harmônico simples usando PyTorch e Runge-Kutta.
"""

import numpy as np
import torch
from .rk4 import RungeKutta4


class OsciladorHarmonicoPyTorch:
    """
    Classe para resolver a equação do oscilador harmônico simples usando PyTorch e Runge-Kutta
    """
    
    def __init__(self, frequencias_angulares, amortecimento=0.0, device='cpu'):
        """
        Parâmetros do oscilador para múltiplos sistemas simultâneos
        
        Args:
            frequencias_angulares (list ou tensor): frequências angulares de cada sistema (rad/s)
            amortecimento (float): coeficiente de amortecimento (kg/s)
            device (str): dispositivo para computação ('cpu' ou 'cuda')
        """
        # converte para tensor (n_sistemas,)
        if isinstance(frequencias_angulares, list):
            self.frequencias_angulares = torch.tensor(frequencias_angulares, dtype=torch.float32, device=device)
        else:
            self.frequencias_angulares = frequencias_angulares.clone().detach().to(device)
        
        self.device = device
        self.amortecimento = amortecimento
        self.n_sistemas = len(self.frequencias_angulares)
        self.b = torch.tensor(amortecimento, dtype=torch.float32, device=device).expand(self.n_sistemas)
        self.frequencias_lineares = self.frequencias_angulares / (2 * np.pi)
        self.periodos = 1.0 / self.frequencias_lineares
        self.rk4 = RungeKutta4(device=device)
        
    def equacoes_movimento(self, t, estados):
        """
        Define as equações do movimento para múltiplas condições iniciais e múltiplos sistemas
        
        estados: tensor da forma (n_condicoes, n_sistemas, 2)
        retorna: tensor da forma (n_condicoes, n_sistemas, 2)
        """
        x = estados[:, :, 0]
        v = estados[:, :, 1]
        
        dxdt = v
        
        # rearanja omega2 para que tenha as dimensões condizentes com o número de sistemas (omega2, n_sistemas)
        omega2 = self.frequencias_angulares ** 2
        omega2_expanded = omega2.unsqueeze(0).expand(x.shape[0], -1)
        b_expanded = self.b.unsqueeze(0).expand(x.shape[0], -1)
        
        dvdt = -omega2_expanded * x - b_expanded * v
        
        return torch.stack([dxdt, dvdt], dim=2)
        
    def resolve_multi_condicoes_sistemas(self, condicoes_iniciais, t_final, dt):
        """
        Resolve a EDO para múltiplas condições iniciais e múltiplos sistemas simultaneamente
        
        condicoes_iniciais: tensor de forma (n_condicoes, 2) com [x0, v0]
        t_final: tempo final
        dt: passo temporal
        
        Returns:
            dict: Dicionário com os resultados da simulação (solução)
        """
        n_condicoes = condicoes_iniciais.shape[0]
        
        # rearanja condicoes_iniciais para que tenha as dimensões condizentes com o número de sistemas (n_condicoes, n_sistemas, 2)
        cond_iniciais_expand = condicoes_iniciais.unsqueeze(1).expand(-1, self.n_sistemas, -1)
        
        solucao_rk4 = self.rk4.solve(
            func=self.equacoes_movimento,
            condicoes_iniciais=cond_iniciais_expand,
            t_range=(0, t_final),
            dt=dt
        )
        
        tempos = solucao_rk4['tempo']
        estados = solucao_rk4['estados']  # (n_passos, n_condicoes, n_sistemas, 2)
        
        posicoes = estados[:, :, :, 0]
        velocidades = estados[:, :, :, 1]
        
        # calcula energias (normalizadas por massa) para cada sistema
        omega2 = self.frequencias_angulares.cpu().numpy() ** 2
        omega2_expanded = omega2.reshape(1, 1, -1)  # (1, 1, n_sistemas)
        
        energia_cinetica = 0.5 * velocidades**2
        energia_potencial = 0.5 * omega2_expanded * posicoes**2
        energia_mecanica = energia_cinetica + energia_potencial
        
        # calcula amplitudes máximas para cada condição inicial e sistema
        amplitudes_max = np.max(np.abs(posicoes), axis=0)
        
        return {
            'tempo': tempos,
            'posicao': posicoes,
            'velocidade': velocidades,
            'energia_cinetica': energia_cinetica,
            'energia_potencial': energia_potencial,
            'energia_mecanica': energia_mecanica,
            'amplitudes': amplitudes_max,
            'condicoes_iniciais': condicoes_iniciais.cpu().numpy(),
            'frequencias_angulares': self.frequencias_angulares.cpu().numpy(),
            'frequencias_lineares': self.frequencias_lineares.cpu().numpy(),
            'periodos': self.periodos.cpu().numpy(),
            'n_condicoes': n_condicoes,
            'n_sistemas': self.n_sistemas
        }