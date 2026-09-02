"""
Classe para resolver o oscilador de FitzHugh-Nagumo usando PyTorch e Runge-Kutta.
O oscilador de FitzHugh-Nagumo é um modelo simplificado bidimensional da excitação neuronal.
Sistema: dv/dt = v - v^3/3 - w + I
         dw/dt = epsilon*(v + a - b*w)
"""

import torch
import numpy as np
from .rk4 import RungeKutta4


class OsciladorFitzHughNagumo:
    """
    Classe para resolver as equações do oscilador de FitzHugh-Nagumo usando PyTorch e Runge-Kutta.
    """
    
    def __init__(self, parametros_epsilon, a=0.7, b=0.8, I=0.5, device='cpu'):
        """
        Parâmetros do oscilador para múltiplos sistemas simultâneos.
        
        Args:
            parametros_epsilon (list ou tensor): parâmetros de separação de escalas (epsilon) 
                                                que controlam a separação entre dinâmica rápida e lenta
            a (float): parâmetro de recuperação (offset da nuliclina de w)
            b (float): parâmetro de recuperação (inclinação da nuliclina de w)
            I (float): corrente aplicada (entrada externa)
            device (str): dispositivo para computação ('cpu' ou 'cuda')
        """
        if isinstance(parametros_epsilon, list):
            self.parametros_epsilon = torch.tensor(parametros_epsilon, dtype=torch.float32, device=device)
        else:
            self.parametros_epsilon = parametros_epsilon.clone().detach().to(device)
        
        self.a = a
        self.b = b
        self.I = I
        self.device = device
        self.n_sistemas = len(self.parametros_epsilon)
        
        # ponto de equilíbrio (v*, w*) obtido resolvendo:
        # v - v^3/3 - w + I = 0
        # w = (v + a)/b
        # resulta em uma equação cúbica que pode ter 1 ou 3 soluções
        # inicializamos com uma aproximação
        self.potencial_eq = torch.zeros_like(self.parametros_epsilon)
        self.recuperacao_eq = torch.zeros_like(self.parametros_epsilon)
        
        # epsilon pequeno, a frequência escala com epsilon
        # estimativa inicial(parâmetros típicos)
        self.frequencias_angulares = self.parametros_epsilon * 2.0 * np.pi
        
        # período estimado (escala com 1/epsilon para epsilon pequeno)
        self.periodos = 1.0 / self.parametros_epsilon * 2.0 * np.pi
        
        # amplitude do ciclo limite (aproximadamente 2 para o potencial v)
        # e (2+a)/b para a recuperação w
        self.amplitude_ciclo_limite_v = 2.0 * torch.ones_like(self.parametros_epsilon)
        self.amplitude_ciclo_limite_w = (2.0 + self.a) / self.b * torch.ones_like(self.parametros_epsilon)
        
        # estimativa inicial do período para epsilon pequeno
        # epsilon pequeno, T ≈ (3 - 2*ln(2))/epsilon
        self.periodos_estimados = torch.where(
            self.parametros_epsilon < 0.1,
            (3.0 - 2.0 * np.log(2.0)) / self.parametros_epsilon,
            2.0 * np.pi * torch.ones_like(self.parametros_epsilon)
        )
        
        # I grande, o sistema pode não oscilar (ponto fixo estável)
        # critério aproximado: oscila se I está entre dois valores críticos
        # parâmetros típicos (a=0.7, b=0.8), I_critical ≈ 0.33 e 0.67
        self.possui_oscilacao = torch.ones_like(self.parametros_epsilon, dtype=torch.bool)
        
        self.rk4 = RungeKutta4(device=device)
        
    def equacoes_movimento(self, t, estados):
        """
        Define as equações do movimento para múltiplas condições iniciais e múltiplos sistemas.
        
        estados: tensor da forma (n_condicoes, n_sistemas, 2)
                 onde o último eixo é [potencial (v), recuperação (w)]
        retorna: tensor da forma (n_condicoes, n_sistemas, 2)
        """
        v = estados[:, :, 0]  # potencial de membrana
        w = estados[:, :, 1]  # variável de recuperação
        
        # dimensões para transmissão correta dos parâmetros epsilon para cada sistema
        epsilon = self.parametros_epsilon.unsqueeze(0).expand(v.shape[0], -1)
        
        # dv/dt = v - v^3/3 - w + I
        dvdt = v - (v**3) / 3.0 - w + self.I
        
        # dw/dt = epsilon*(v + a - b*w)
        dwdt = epsilon * (v + self.a - self.b * w)
        
        return torch.stack([dvdt, dwdt], dim=2)
    
    def resolve_multi_condicoes_sistemas(self, condicoes_iniciais, t_final, dt, save_every=1):
        """
        Resolve a EDO para múltiplas condições iniciais e múltiplos sistemas simultaneamente.
        
        condicoes_iniciais: tensor de forma (n_condicoes, 2) com [v0, w0] (potencial inicial, recuperação inicial)
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
        
        # potencial e recuperação
        potencial = estados[:, :, :, 0]
        recuperacao = estados[:, :, :, 1]
        
        # ========== VERIFICAÇÕES PARA O SISTEMA NÃO CONSERVATIVO ==========
        
        # 1. estabilidade do ciclo limite (para epsilon > 0 e I na faixa correta)
        # para o oscilador de FitzHugh-Nagumo, a amplitude deve convergir para o ciclo limite
        # detecta se a amplitude do potencial converge para ≈ 2
        amplitude_potencial = np.max(np.abs(potencial), axis=0)
        amplitude_recuperacao = np.max(np.abs(recuperacao), axis=0)
        
        # calcula a convergência da amplitude (variação relativa nas últimas oscilações)
        n_passos = potencial.shape[0]
        # usa os últimos 20% dos pontos para verificar convergência
        n_finais = max(1, int(n_passos * 0.2))
        
        # divide a trajetória em blocos para verificar convergência
        n_blocos = 4
        tamanho_bloco = max(1, n_passos // n_blocos)
        amplitudes_blocos_v = []
        amplitudes_blocos_w = []
        
        for i in range(n_blocos):
            inicio = i * tamanho_bloco
            fim = min((i + 1) * tamanho_bloco, n_passos)
            amp_bloco_v = np.max(np.abs(potencial[inicio:fim, :, :]), axis=0)
            amp_bloco_w = np.max(np.abs(recuperacao[inicio:fim, :, :]), axis=0)
            amplitudes_blocos_v.append(amp_bloco_v)
            amplitudes_blocos_w.append(amp_bloco_w)
        
        # calcula variação relativa entre blocos consecutivos
        variacao_amplitudes_v = []
        variacao_amplitudes_w = []
        for i in range(len(amplitudes_blocos_v) - 1):
            variacao_relativa_v = np.abs(amplitudes_blocos_v[i+1] - amplitudes_blocos_v[i]) / (amplitudes_blocos_v[i] + 1e-10)
            variacao_relativa_w = np.abs(amplitudes_blocos_w[i+1] - amplitudes_blocos_w[i]) / (amplitudes_blocos_w[i] + 1e-10)
            variacao_amplitudes_v.append(variacao_relativa_v)
            variacao_amplitudes_w.append(variacao_relativa_w)
        
        # se a variação relativa é pequena, o sistema convergiu para o ciclo limite
        convergiu_ciclo_v = np.array(variacao_amplitudes_v[-1] < 0.01)  # menos de 1% de variação
        convergiu_ciclo_w = np.array(variacao_amplitudes_w[-1] < 0.01)
        convergiu_ciclo = convergiu_ciclo_v & convergiu_ciclo_w
        
        # 2. taxa de variação da energia (pseudo-energia)
        # dE/dt = v^2 - v^4/3 - v*w + I*v + epsilon*w*v + epsilon*a*w - epsilon*b*w^2
        
        # calcula a pseudo-energia nos primeiros e últimos momentos
        pseudo_energia = 0.5 * recuperacao**2 + 0.5 * potencial**2
        
        # energia média nas primeiras oscilações (início)
        n_inicial = max(1, int(n_passos * 0.1))
        energia_inicial_media = pseudo_energia[:n_inicial, :, :].mean(axis=0)
        
        # energia média nas últimas oscilações (final - estado estacionário)
        energia_final_media = pseudo_energia[-n_finais:, :, :].mean(axis=0)
        
        # variação da energia (deve tender a zero no ciclo limite)
        variacao_energia = energia_final_media - energia_inicial_media
        
        # 3. convergência para o ciclo limite independente das condições iniciais
        # diferentes condições iniciais devem convergir para o mesmo ciclo limite
        convergencia_universal = np.ones(self.n_sistemas)
        
        if n_condicoes > 1:
            # calcula a amplitude final para diferentes condições iniciais
            amp_final_v = amplitude_potencial
            amp_final_w = amplitude_recuperacao
            # verifica se as amplitudes finais são similares (desvio padrão pequeno)
            if amp_final_v.shape[0] > 1:  # múltiplas condições iniciais
                std_amp_v = np.std(amp_final_v, axis=0)
                mean_amp_v = np.mean(amp_final_v, axis=0)
                std_amp_w = np.std(amp_final_w, axis=0)
                mean_amp_w = np.mean(amp_final_w, axis=0)
                # se o desvio padrão relativo é pequeno, convergiu para o mesmo ciclo limite
                convergencia_universal = ((std_amp_v / (mean_amp_v + 1e-10)) < 0.05) & \
                                        ((std_amp_w / (mean_amp_w + 1e-10)) < 0.05)
        
        # 4. periodicidade - verifica se há oscilações regulares
        # detecta picos no potencial
        periodicidade = np.zeros((n_condicoes, self.n_sistemas), dtype=bool)
        for i in range(n_condicoes):
            for j in range(self.n_sistemas):
                v_traj = potencial[:, i, j].cpu().numpy() if isinstance(potencial, torch.Tensor) else potencial[:, i, j]
                # encontra picos (máximos locais)
                picos = 0
                for k in range(1, len(v_traj) - 1):
                    if v_traj[k] > v_traj[k-1] and v_traj[k] > v_traj[k+1] and v_traj[k] > 0.5:
                        picos += 1
                # se tem pelo menos 3 picos, consideramos periódico
                periodicidade[i, j] = picos >= 3
        
        return {
            'tempo': tempos,
            'potencial': potencial,
            'recuperacao': recuperacao,
            'estados': estados,
            'periodicidade': periodicidade,
            
            'convergiu_ciclo_limite': convergiu_ciclo,
            'energia_inicial_media': energia_inicial_media.cpu().numpy() if isinstance(energia_inicial_media, torch.Tensor) else energia_inicial_media,
            'energia_final_media': energia_final_media.cpu().numpy() if isinstance(energia_final_media, torch.Tensor) else energia_final_media,
            'variacao_energia': variacao_energia.cpu().numpy() if isinstance(variacao_energia, torch.Tensor) else variacao_energia,
            'convergencia_universal': convergencia_universal,
            
            'amplitude_potencial': amplitude_potencial,
            'amplitude_recuperacao': amplitude_recuperacao,
            'amplitude_ciclo_limite_teorica_v': self.amplitude_ciclo_limite_v.cpu().numpy(),
            'amplitude_ciclo_limite_teorica_w': self.amplitude_ciclo_limite_w.cpu().numpy(),
            'periodo_estimado': self.periodos_estimados.cpu().numpy(),
            
            'potencial_eq': self.potencial_eq.cpu().numpy(),
            'recuperacao_eq': self.recuperacao_eq.cpu().numpy(),
            
            'parametros_epsilon': self.parametros_epsilon.cpu().numpy(),
            'a': self.a,
            'b': self.b,
            'I': self.I,
            'possui_oscilacao': self.possui_oscilacao.cpu().numpy(),
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
        amp_v = resultado['amplitude_potencial']
        amp_w = resultado['amplitude_recuperacao']
        amp_teorica_v = resultado['amplitude_ciclo_limite_teorica_v']
        amp_teorica_w = resultado['amplitude_ciclo_limite_teorica_w']
        
        # verifica se a amplitude está próxima do valor teórico
        erro_amplitude_v = np.abs(amp_v - amp_teorica_v) / amp_teorica_v
        erro_amplitude_w = np.abs(amp_w - amp_teorica_w) / amp_teorica_w
        
        # média dos erros
        erro_amplitude = (erro_amplitude_v + erro_amplitude_w) / 2.0
        
        # critérios de convergência (adaptados para múltiplas dimensões)
        # periodicidade: verifica se a maioria das condições iniciais é periódica
        if len(periodicidade.shape) > 1:
            criterio_periodicidade = np.mean(periodicidade, axis=0) > 0.5
        else:
            criterio_periodicidade = periodicidade > 0.5
            
        # convergência: verifica se cada sistema convergiu
        criterio_convergencia = convergiu
        
        # energia: verifica se a variação da energia é pequena
        energia_final = np.mean(resultado['energia_final_media'], axis=0) if len(resultado['energia_final_media'].shape) > 1 else resultado['energia_final_media']
        criterio_energia = np.abs(variacao_energia) < 0.1 * (energia_final + 1e-10)
        
        # universalidade: verifica se diferentes condições iniciais convergem para o mesmo ciclo
        criterio_universal = convergencia_universal > 0.5 if isinstance(convergencia_universal, np.ndarray) else convergencia_universal
        
        # amplitude: erro menor que 10%
        criterio_amplitude = erro_amplitude < 0.1
        
        # convergência final para cada sistema
        convergencia_final = (
            criterio_periodicidade & 
            criterio_convergencia & 
            criterio_universal & 
            criterio_amplitude
        )
        
        return {
            'convergiu_para_ciclo_limite': convergencia_final,
            'periodicidade': criterio_periodicidade,
            'variacao_energia_estabilizada': criterio_energia,
            'convergencia_universal': criterio_universal,
            'erro_amplitude': erro_amplitude,
            'amplitude_potencial_observada': amp_v,
            'amplitude_recuperacao_observada': amp_w,
            'amplitude_potencial_teorica': amp_teorica_v,
            'amplitude_recuperacao_teorica': amp_teorica_w,
            'convergencia_detalhada': {
                'periodicidade': criterio_periodicidade,
                'convergencia_amplitude': criterio_convergencia,
                'estabilizacao_energia': criterio_energia,
                'universalidade': criterio_universal,
                'erro_amplitude_aceitavel': criterio_amplitude
            }
        }