"""
Classe para resolver o oscilador de Van der Pol usando PyTorch e Runge-Kutta.
O oscilador de Van der Pol descreve um sistema não linear com amortecimento dependente da amplitude.
Sistema: dx/dt = y
         dy/dt = mu*(1 - x^2)*y - x
"""

import torch
import numpy as np
from .rk4 import RungeKutta4


class OsciladorVanDerPol:
    """
    Classe para resolver a equação do oscilador de Van der Pol usando PyTorch e Runge-Kutta.
    """
    
    def __init__(self, parametros_mu, device='cpu'):
        """
        Parâmetros do oscilador para múltiplos sistemas simultâneos.
        
        Args:
            parametros_mu (list ou tensor): parâmetros de não linearidade (mu) que controlam 
                                           a força do amortecimento não linear
            device (str): dispositivo para computação ('cpu' ou 'cuda')
        """
        if isinstance(parametros_mu, list):
            self.parametros_mu = torch.tensor(parametros_mu, dtype=torch.float32, device=device)
        else:
            self.parametros_mu = parametros_mu.clone().detach().to(device)
        
        self.device = device
        self.n_sistemas = len(self.parametros_mu)
        
        # ponto de equilíbrio (x*, y*) = (0, 0)
        self.posicao_eq = torch.zeros_like(self.parametros_mu)
        self.velocidade_eq = torch.zeros_like(self.parametros_mu)
        
        # mu pequeno, a frequência é aproximadamente 1
        # mu grande, a frequência escala como 1/mu
        # estimativa inicial
        self.frequencias_angulares = torch.ones_like(self.parametros_mu)
        self.periodos = 2.0 * np.pi * torch.ones_like(self.parametros_mu)
        
        # amplitude do ciclo limite (aproximadamente 2 para mu > 0)
        self.amplitude_ciclo_limite = 2.0 * torch.ones_like(self.parametros_mu)
        
        # estimativa inicial do período para mu > 0
        # mu grande, T ≈ (3 - 2*ln(2))/mu ≈ 1.6137/mu
        self.periodos_estimados = torch.where(
            self.parametros_mu > 1.0,
            (3.0 - 2.0 * np.log(2.0)) / self.parametros_mu,
            2.0 * np.pi * torch.ones_like(self.parametros_mu)
        )
        
        self.rk4 = RungeKutta4(device=device)
        
    def equacoes_movimento(self, t, estados):
        """
        Define as equações do movimento para múltiplas condições iniciais e múltiplos sistemas.
        
        estados: tensor da forma (n_condicoes, n_sistemas, 2)
                 onde o último eixo é [posição (x), velocidade (y)]
        retorna: tensor da forma (n_condicoes, n_sistemas, 2)
        """
        x = estados[:, :, 0]  # posição
        y = estados[:, :, 1]  # velocidade
        
        # dimensões para transmissão correta dos parâmetros mu para cada sistema
        mu = self.parametros_mu.unsqueeze(0).expand(x.shape[0], -1)
        
        dxdt = y
        
        # dy/dt = mu*(1 - x^2)*y - x
        dydt = mu * (1.0 - x**2) * y - x
        
        return torch.stack([dxdt, dydt], dim=2)
    
    def resolve_multi_condicoes_sistemas(self, condicoes_iniciais, t_final, dt, save_every=1):
        """
        Resolve a EDO para múltiplas condições iniciais e múltiplos sistemas simultaneamente.
        
        condicoes_iniciais: tensor de forma (n_condicoes, 2) com [x0, y0] (posição inicial, velocidade inicial)
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
        
        # posição e velocidade
        posicao = estados[:, :, :, 0]
        velocidade = estados[:, :, :, 1]
        
        # ========== VERIFICAÇÕES PARA O SISTEMA NÃO CONSERVATIVO ==========
        
        # 1. estabilidade do ciclo limite (para mu > 0)
        # para o oscilador de Van der Pol, a amplitude deve convergir para o ciclo limite
        # detecta se a amplitude da posição converge para ≈ 2
        amplitude_posicao = np.max(np.abs(posicao), axis=0)
        
        # calcula a convergência da amplitude (variação relativa nas últimas oscilações)
        n_passos = posicao.shape[0]
        # usa os últimos 20% dos pontos para verificar convergência
        n_finais = max(1, int(n_passos * 0.2))
        
        # divide a trajetória em blocos para verificar convergência
        n_blocos = 4
        tamanho_bloco = max(1, n_passos // n_blocos)
        amplitudes_blocos = []
        
        for i in range(n_blocos):
            inicio = i * tamanho_bloco
            fim = min((i + 1) * tamanho_bloco, n_passos)
            amp_bloco = np.max(np.abs(posicao[inicio:fim, :, :]), axis=0)
            amplitudes_blocos.append(amp_bloco)
        
        # calcula variação relativa entre blocos consecutivos
        variacao_amplitudes = []
        for i in range(len(amplitudes_blocos) - 1):
            variacao_relativa = np.abs(amplitudes_blocos[i+1] - amplitudes_blocos[i]) / (amplitudes_blocos[i] + 1e-10)
            variacao_amplitudes.append(variacao_relativa)
        
        # se a variação relativa é pequena, o sistema convergiu para o ciclo limite
        convergiu_ciclo = np.array(variacao_amplitudes[-1] < 0.01)  # menos de 1% de variação
        
        # 2. taxa de variação da energia (amortecimento)
        # mu > 0, energia deve ser adicionada para pequenas amplitudes (|x| < 1)
        # e removida para grandes amplitudes (|x| > 1)
        # dE/dt = mu*(1 - x^2)*y^2
        
        # calcula a energia nos primeiros e últimos momentos
        energia_media = 0.5 * velocidade**2 + 0.5 * posicao**2
        
        # energia média nas primeiras oscilações (início)
        n_inicial = max(1, int(n_passos * 0.1))
        energia_inicial_media = energia_media[:n_inicial, :, :].mean(axis=0)
        
        # energia média nas últimas oscilações (final - estado estacionário)
        energia_final_media = energia_media[-n_finais:, :, :].mean(axis=0)
        
        # variação da energia (deve tender a zero no ciclo limite)
        variacao_energia = energia_final_media - energia_inicial_media
        
        # 3. convergência para o ciclo limite independente das condições iniciais
        # mu > 0, diferentes condições iniciais devem convergir para o mesmo ciclo limite
        convergencia_universal = np.ones(self.n_sistemas)
        
        if n_condicoes > 1:
            # calcula a amplitude final para diferentes condições iniciais
            amp_final = amplitude_posicao
            # verifica se as amplitudes finais são similares (desvio padrão pequeno)
            if amp_final.shape[0] > 1:  # múltiplas condições iniciais
                std_amp = np.std(amp_final, axis=0)
                mean_amp = np.mean(amp_final, axis=0)
                # se o desvio padrão relativo é pequeno, convergiu para o mesmo ciclo limite
                convergencia_universal = (std_amp / (mean_amp + 1e-10)) < 0.05  # menos de 5% de variação
        
        return {
            'tempo': tempos,
            'posicao': posicao,
            'velocidade': velocidade,
            'estados': estados,
            
            'convergiu_ciclo_limite': convergiu_ciclo,
            'energia_inicial_media': energia_inicial_media.cpu().numpy() if isinstance(energia_inicial_media, torch.Tensor) else energia_inicial_media,
            'energia_final_media': energia_final_media.cpu().numpy() if isinstance(energia_final_media, torch.Tensor) else energia_final_media,
            'variacao_energia': variacao_energia.cpu().numpy() if isinstance(variacao_energia, torch.Tensor) else variacao_energia,
            'convergencia_universal': convergencia_universal,
            
            'amplitude_posicao': amplitude_posicao,
            'amplitude_velocidade': np.max(np.abs(velocidade), axis=0),
            'amplitude_ciclo_limite_teorica': self.amplitude_ciclo_limite.cpu().numpy(),
            'periodo_estimado': self.periodos_estimados.cpu().numpy(),
            
            'posicao_eq': self.posicao_eq.cpu().numpy(),
            'velocidade_eq': self.velocidade_eq.cpu().numpy(),
            
            'parametros_mu': self.parametros_mu.cpu().numpy(),
            'condicoes_iniciais': condicoes_iniciais.cpu().numpy(),
            
            'n_condicoes': n_condicoes,
            'n_sistemas': self.n_sistemas
        }
    
    def verificar_convergencia_ciclo_limite(self, resultado):
        """
        Verifica se o sistema convergiu para o ciclo limite.
        
        Args:
            resultado: dicionário retornado por resolve_multi_condicoes_sistemas
        
        Returns:
            dict: Métricas de convergência
        """
        convergiu = resultado['convergiu_ciclo_limite']
        variacao_energia = resultado['variacao_energia']
        periodicidade = resultado['periodicidade']
        convergencia_universal = resultado['convergencia_universal']
        amp_posicao = resultado['amplitude_posicao']
        amp_teorica = resultado['amplitude_ciclo_limite_teorica']
        
        # verifica se a amplitude está próxima do valor teórico
        erro_amplitude = np.abs(amp_posicao - amp_teorica) / amp_teorica
        
        # critérios de convergência
        criterio_periodicidade = periodicidade > 0.5
        criterio_convergencia = convergiu
        criterio_energia = np.abs(variacao_energia) < 0.1 * np.mean(resultado['energia_final_media'])
        criterio_universal = convergencia_universal > 0.5 if isinstance(convergencia_universal, np.ndarray) else convergencia_universal
        criterio_amplitude = erro_amplitude < 0.1  # erro menor que 10%
        
        convergencia_final = (
            criterio_periodicidade & 
            criterio_convergencia & 
            criterio_universal & 
            criterio_amplitude
        )
        
        return {
            'convergiu_para_ciclo_limite': convergencia_final,
            'periodicidade': periodicidade,
            'variacao_energia_estabilizada': criterio_energia,
            'convergencia_universal': criterio_universal,
            'erro_amplitude': erro_amplitude,
            'amplitude_observada': amp_posicao,
            'amplitude_teorica': amp_teorica,
            'convergencia_detalhada': {
                'periodicidade': criterio_periodicidade,
                'convergencia_amplitude': criterio_convergencia,
                'estabilizacao_energia': criterio_energia,
                'universalidade': criterio_universal,
                'erro_amplitude_aceitavel': criterio_amplitude
            }
        }