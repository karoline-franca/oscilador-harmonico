"""
Classe para resolver o oscilador de Lotka-Volterra usando PyTorch e Runge-Kutta.
O oscilador de Lotka-Volterra descreve a dinâmica entre populações de presas (x) e predadores (y).
Sistema: dx/dt = a*x - b*x*y
         dy/dt = -c*y + d*x*y
"""

import torch
import numpy as np
from .rk4 import RungeKutta4


class OsciladorLotkaVolterra:
    """
    Classe para resolver a equação do oscilador de Lotka-Volterra usando PyTorch e Runge-Kutta.
    """
    
    def __init__(self, taxas_crescimento, taxas_mortalidade, 
                 taxas_predacao, taxas_eficiencia, device='cpu'):
        """
        Parâmetros do oscilador para múltiplos sistemas simultâneos.
        
        Args:
            taxas_crescimento (list ou tensor): taxas de crescimento das presas (a) em 1/s
            taxas_mortalidade (list ou tensor): taxas de mortalidade dos predadores (c) em 1/s
            taxas_predacao (list ou tensor): taxas de predação (b) em 1/(indivíduo·s)
            taxas_eficiencia (list ou tensor): eficiência de conversão (d) em 1/(indivíduo·s)
            device (str): dispositivo para computação ('cpu' ou 'cuda')
        """
        if isinstance(taxas_crescimento, list):
            self.taxas_crescimento = torch.tensor(taxas_crescimento, dtype=torch.float32, device=device)
        else:
            self.taxas_crescimento = taxas_crescimento.clone().detach().to(device)
            
        if isinstance(taxas_mortalidade, list):
            self.taxas_mortalidade = torch.tensor(taxas_mortalidade, dtype=torch.float32, device=device)
        else:
            self.taxas_mortalidade = taxas_mortalidade.clone().detach().to(device)
            
        if isinstance(taxas_predacao, list):
            self.taxas_predacao = torch.tensor(taxas_predacao, dtype=torch.float32, device=device)
        else:
            self.taxas_predacao = taxas_predacao.clone().detach().to(device)
            
        if isinstance(taxas_eficiencia, list):
            self.taxas_eficiencia = torch.tensor(taxas_eficiencia, dtype=torch.float32, device=device)
        else:
            self.taxas_eficiencia = taxas_eficiencia.clone().detach().to(device)
        
        self.device = device
        self.n_sistemas = len(self.taxas_crescimento)
        
        # ponto de equilíbrio não-trivial (x*, y*)
        self.presas_eq = self.taxas_mortalidade / self.taxas_eficiencia
        self.predadores_eq = self.taxas_crescimento / self.taxas_predacao
        
        # frequência angular das oscilações (aproximação linear em torno do equilíbrio)
        self.frequencias_angulares = torch.sqrt(self.taxas_crescimento * self.taxas_mortalidade)
        self.periodos = 2.0 * np.pi / self.frequencias_angulares
        
        self.rk4 = RungeKutta4(device=device)
        
    def equacoes_movimento(self, t, estados):
        """
        Define as equações do movimento para múltiplas condições iniciais e múltiplos sistemas.
        
        estados: tensor da forma (n_condicoes, n_sistemas, 2)
                 onde o último eixo é [presas (x), predadores (y)]
        retorna: tensor da forma (n_condicoes, n_sistemas, 2)
        """
        x = estados[:, :, 0]  # presas
        y = estados[:, :, 1]  # predadores
        
        # dimensões para transmissão correta das taxas para cada sistema
        a = self.taxas_crescimento.unsqueeze(0).expand(x.shape[0], -1)
        b = self.taxas_predacao.unsqueeze(0).expand(x.shape[0], -1)
        c = self.taxas_mortalidade.unsqueeze(0).expand(x.shape[0], -1)
        d = self.taxas_eficiencia.unsqueeze(0).expand(x.shape[0], -1)
        
        dxdt = a * x - b * x * y
        
        dydt = -c * y + d * x * y
        
        return torch.stack([dxdt, dydt], dim=2)
    
    def resolve_multi_condicoes_sistemas(self, condicoes_iniciais, t_final, dt, save_every=1):
        """
        Resolve a EDO para múltiplas condições iniciais e múltiplos sistemas simultaneamente.
        
        condicoes_iniciais: tensor de forma (n_condicoes, 2) com [x0, y0]
        t_final: tempo final
        dt: passo temporal
        save_every: salvar a cada 'save_every' passos
        
        Returns:
            dict: Dicionário com os resultados da simulação
        """
        n_condicoes = condicoes_iniciais.shape[0]
        
        # expande condicoes_iniciais para (n_condicoes, n_sistemas, 2)
        cond_iniciais_expand = condicoes_iniciais.unsqueeze(1).expand(-1, self.n_sistemas, -1)
        
        solucao_rk4 = self.rk4.solve(
            func=self.equacoes_movimento,
            condicoes_iniciais=cond_iniciais_expand,
            t_range=(0, t_final),
            dt=dt,
            save_every=save_every
        )
        
        tempos = solucao_rk4['tempo']
        estados = solucao_rk4['estados']  # (n_passos, n_condicoes, n_sistemas, 2)
        
        # presas e predadores
        presas = estados[:, :, :, 0]
        predadores = estados[:, :, :, 1]
        
        # constante de movimento (H) para cada sistema
        # H = d*x + b*y - c*ln(x) - a*ln(y)
        epsilon = 1e-10  # valor para evitar log(0)
        a = self.taxas_crescimento.cpu().numpy().reshape(1, 1, -1)
        b = self.taxas_predacao.cpu().numpy().reshape(1, 1, -1)
        c = self.taxas_mortalidade.cpu().numpy().reshape(1, 1, -1)
        d = self.taxas_eficiencia.cpu().numpy().reshape(1, 1, -1)
        
        constante_movimento = (d * presas + b * predadores - 
                              c * np.log(presas + epsilon) - 
                              a * np.log(predadores + epsilon))
        
        # amplitudes máximas para cada condição inicial e sistema
        amplitudes_presas = np.max(presas, axis=0)
        amplitudes_predadores = np.max(predadores, axis=0)
        
        return {
            'tempo': tempos,
            'presas': presas,
            'predadores': predadores,
            'estados': estados,
            'constante_movimento': constante_movimento,
            'amplitudes_presas': amplitudes_presas,
            'amplitudes_predadores': amplitudes_predadores,
            'presas_eq': self.presas_eq.cpu().numpy(),
            'predadores_eq': self.predadores_eq.cpu().numpy(),
            'frequencias_angulares': self.frequencias_angulares.cpu().numpy(),
            'periodos': self.periodos.cpu().numpy(),
            'condicoes_iniciais': condicoes_iniciais.cpu().numpy(),
            'taxas_crescimento': self.taxas_crescimento.cpu().numpy(),
            'taxas_mortalidade': self.taxas_mortalidade.cpu().numpy(),
            'taxas_predacao': self.taxas_predacao.cpu().numpy(),
            'taxas_eficiencia': self.taxas_eficiencia.cpu().numpy(),
            'n_condicoes': n_condicoes,
            'n_sistemas': self.n_sistemas
        }