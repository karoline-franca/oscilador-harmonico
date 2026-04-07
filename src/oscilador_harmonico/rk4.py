"""
Classe para resolver sistemas de EDOs usando o método Runge-Kutta de 4ª ordem.
"""

import torch
import numpy as np


class RungeKutta4:
    """
    Implementação do método Runge-Kutta de 4ª ordem para resolver sistemas de EDOs.
    
    O método RK4 é aplicável a qualquer sistema de equações diferenciais ordinárias
    da forma: dy/dt = f(t, y), onde y é um vetor de estados.
    """
    
    def __init__(self, device='cpu'):
        """
        Inicializa o solver RK4.
        
        Args:
            device (str): dispositivo para computação ('cpu' ou 'cuda')
        """
        self.device = device
        
    def step(self, func, estados, t, dt):
        """
        Executa um passo do método RK4.
        
        Args:
            func (callable): Função que define o sistema de EDOs.
                            Deve ter assinatura: func(t, estados) -> derivadas
            estados (torch.Tensor): Estados atuais (n_condicoes, n_sistemas, n_variaveis)
            t (float): Tempo atual
            dt (float): Passo de tempo
            
        Returns:
            torch.Tensor: Novos estados após o passo dt
        """
        k1 = func(t, estados)
        k2 = func(t + 0.5 * dt, estados + 0.5 * dt * k1)
        k3 = func(t + 0.5 * dt, estados + 0.5 * dt * k2)
        k4 = func(t + dt, estados + dt * k3)
        
        estados_novos = estados + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return estados_novos
    
    def solve(self, func, condicoes_iniciais, t_span, dt, save_every=1):
        """
        Resolve o sistema de EDOs usando RK4.
        
        Args:
            func (callable): Função que define o sistema de EDOs.
                            Deve ter assinatura: func(t, estados) -> derivadas
            condicoes_iniciais (torch.Tensor): Condições iniciais (n_condicoes, n_sistemas, n_variaveis)
            t_span (tuple): Intervalo de tempo (t_inicial, t_final)
            dt (float): Passo de tempo
            save_every (int): Salvar a cada 'save_every' passos (para reduzir memória)
            
        Returns:
            dict: Dicionário com os resultados da simulação
        """
        t0, tf = t_span
        n_passos = int((tf - t0) / dt) + 1
        
        indices_salvos = list(range(0, n_passos, save_every))
        if indices_salvos[-1] != n_passos - 1:
            indices_salvos.append(n_passos - 1)
        
        n_passos_salvos = len(indices_salvos)
        n_condicoes = condicoes_iniciais.shape[0]
        n_sistemas = condicoes_iniciais.shape[1]
        n_variaveis = condicoes_iniciais.shape[2]
        
        tempos_salvos = np.zeros(n_passos_salvos)
        estados_salvos = np.zeros((n_passos_salvos, n_condicoes, n_sistemas, n_variaveis))
        
        estados = condicoes_iniciais.clone()
        t = t0
        
        tempos_salvos[0] = t
        estados_salvos[0] = estados.cpu().numpy()
        
        idx_salvo = 1
        for i in range(1, n_passos):
            estados = self.step(func, estados, t, dt)
            t = t0 + i * dt
            
            if i in indices_salvos:
                tempos_salvos[idx_salvo] = t
                estados_salvos[idx_salvo] = estados.cpu().numpy()
                idx_salvo += 1
        
        return {
            'tempo': tempos_salvos,
            'estados': estados_salvos,
            'n_condicoes': n_condicoes,
            'n_sistemas': n_sistemas,
            'n_variaveis': n_variaveis
        }