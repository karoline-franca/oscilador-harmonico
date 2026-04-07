"""
Classe para resolver a equação do oscilador harmônico simples usando PyTorch e Runge-Kutta.
"""

import numpy as np
import torch


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
        
    def equacoes_movimento(self, estados):
        """
        Define as equações do movimento para múltiplas condições iniciais e múltiplos sistemas
        
        estados: tensor de forma (n_condicoes, n_sistemas, 2)
        retorna: tensor de forma (n_condicoes, n_sistemas, 2)
        """
        x = estados[:, :, 0]
        v = estados[:, :, 1]
        
        dxdt = v
        
        # rearanja omega^2 para que tenha as dimensões condizentes com o número de linhas e sistemas (N, n_sistemas)
        omega2 = self.frequencias_angulares ** 2
        omega2_expanded = omega2.unsqueeze(0).expand(x.shape[0], -1)
        b_expanded = self.b.unsqueeze(0).expand(x.shape[0], -1)
        
        dvdt = -omega2_expanded * x - b_expanded * v
        
        return torch.stack([dxdt, dvdt], dim=2)
    
    def runge_kutta_4(self, estados, dt):
        """
        Método Runge-Kutta de 4ª ordem para múltiplas condições iniciais e múltiplos sistemas
        """
        k1 = self.equacoes_movimento(estados)
        k2 = self.equacoes_movimento(estados + 0.5 * dt * k1)
        k3 = self.equacoes_movimento(estados + 0.5 * dt * k2)
        k4 = self.equacoes_movimento(estados + dt * k3)
        
        estados_novos = estados + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return estados_novos
        
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
        n_passos = int(t_final / dt) + 1
        
        tempos = torch.linspace(0, t_final, n_passos, device=self.device)
        
        # inicializa tensores 3D: (n_passos, n_condicoes, n_sistemas)
        posicoes = torch.zeros((n_passos, n_condicoes, self.n_sistemas), device=self.device)
        velocidades = torch.zeros((n_passos, n_condicoes, self.n_sistemas), device=self.device)
        
        # rearanja condições_iniciais para que tenha as dimensões condizentes com o número de linhas e sistemas (N, n_sistemas)
        cond_iniciais_expand = condicoes_iniciais.unsqueeze(1).expand(-1, self.n_sistemas, -1)
        
        estados = cond_iniciais_expand.clone()
        posicoes[0] = estados[:, :, 0]
        velocidades[0] = estados[:, :, 1]
        
        for i in range(1, n_passos):
            estados = self.runge_kutta_4(estados, dt)
            posicoes[i] = estados[:, :, 0]
            velocidades[i] = estados[:, :, 1]
        
        # calcula energias (normalizadas por massa) para cada sistema
        omega2 = self.frequencias_angulares ** 2

        # rearanja omega2 para que tenha as dimensões condizentes com o número de linhas, passo temporal e condições iniciais (N, n_passos, n_condicoes)
        omega2_expanded = omega2.unsqueeze(0).unsqueeze(0).expand(n_passos, n_condicoes, -1)
        
        energia_cinetica = 0.5 * velocidades**2
        energia_potencial = 0.5 * omega2_expanded * posicoes**2
        energia_mecanica = energia_cinetica + energia_potencial
        
        # calcula amplitudes máximas para cada condição inicial e sistema
        amplitudes_max = torch.max(torch.abs(posicoes), dim=0)[0].cpu().numpy()
        
        return {
            'tempo': tempos.cpu().numpy(),
            'posicao': posicoes.cpu().numpy(),
            'velocidade': velocidades.cpu().numpy(),
            'energia_cinetica': energia_cinetica.cpu().numpy(),
            'energia_potencial': energia_potencial.cpu().numpy(),
            'energia_mecanica': energia_mecanica.cpu().numpy(),
            'amplitudes': amplitudes_max,
            'condicoes_iniciais': condicoes_iniciais.cpu().numpy(),
            'frequencias_angulares': self.frequencias_angulares.cpu().numpy(),
            'frequencias_lineares': self.frequencias_lineares.cpu().numpy(),
            'periodos': self.periodos.cpu().numpy(),
            'n_condicoes': n_condicoes,
            'n_sistemas': self.n_sistemas
        }