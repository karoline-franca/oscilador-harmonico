"""
Nodes do pipeline Kedro para o oscilador de FitzHugh-Nagumo.
"""

import numpy as np
import pandas as pd
import torch
from oscilador_fitz_hugh_nagumo.utils import cria_grafico_2d
import os
from datetime import datetime
from typing import Dict, Any, Tuple

from .fhn import OsciladorFitzHughNagumo


def gera_condicoes_iniciais_node(parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Gera condições iniciais aleatórias para potencial (v0) e recuperação (w0).
    """
    intervals = parameters['intervals']
    n_condicoes = parameters['simulation']['n_condicoes_iniciais']
    seed = parameters['seed']
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    v0 = np.random.uniform(intervals['v0_min'], intervals['v0_max'], n_condicoes)
    w0 = np.random.uniform(intervals['w0_min'], intervals['w0_max'], n_condicoes)
    
    return pd.DataFrame({'v0': v0, 'w0': w0})


def gera_parametros_oscilador_node(parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Gera parâmetros do sistema FitzHugh-Nagumo (epsilon, a, b, I, R).
    """
    intervals = parameters['intervals']
    
    epsilon = intervals['parametro_epsilon']
    a = parameters.get('a', 0.7)  # parâmetro de recuperação (offset)
    b = parameters.get('b', 0.8)  # parâmetro de recuperação (inclinação)
    I = parameters.get('I', 0.5)  # corrente aplicada
    R = parameters.get('R', 0.1)  # resistência de acoplamento

    # classificação qualitativa do comportamento baseado em epsilon
    if epsilon < 0.01:
        desc_epsilon = "Muito lento (epsilon muito pequeno)"
    elif epsilon < 0.05:
        desc_epsilon = "Dinâmica de duas escalas"
    elif epsilon < 0.1:
        desc_epsilon = "Moderadamente separado"
    elif epsilon < 0.5:
        desc_epsilon = "Dinâmica acoplada"
    else:
        desc_epsilon = "Dinâmica rápida (epsilon grande)"
    
    # estimativa do período para epsilon pequeno
    # Para epsilon pequeno, T ≈ (3 - 2*ln(2))/epsilon
    if epsilon < 0.1:
        periodo_estimado = (3.0 - 2.0 * np.log(2.0)) / max(epsilon, 1e-10)
    else:
        # epsilon moderado, estimativa baseada em simulações típicas
        periodo_estimado = 2.0 * np.pi / max(epsilon, 1e-10)

    # estimativa grosseira da faixa de oscilação
    try:
        I_critical_min = (a - 1.0/3.0) / b
        I_critical_max = (a + 1.0/3.0) / b
        possui_oscilacao = (I_critical_min < I < I_critical_max)
    except ZeroDivisionError:
        # quando b = 0, não há divisão por zero
        I_critical_min = -np.inf
        I_critical_max = np.inf
        possui_oscilacao = True  # assume que oscila para b=0
    
    descricao = f"epsilon={epsilon:.4f}, a={a:.2f}, b={b:.2f}, I={I:.2f}, R={R:.2f}, {desc_epsilon}"
    
    return pd.DataFrame({
        'parametro_epsilon': [epsilon],
        'parametro_a': [a],
        'parametro_b': [b],
        'parametro_I': [I],
        'parametro_R': [R],
        'descricao_sistema': [descricao],
        'periodo_estimado': [periodo_estimado],
        'classificacao_epsilon': [desc_epsilon],
        'possui_oscilacao': [possui_oscilacao],
        'I_critical_min': [I_critical_min],
        'I_critical_max': [I_critical_max]
    })


def executa_simulacao_rk4_node(
    condicoes_iniciais: pd.DataFrame,
    parametros_oscilador: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executa a simulação RK4 para todos os sistemas.
    """
    sim_params = parameters['simulation']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # condições iniciais
    v0_vals = condicoes_iniciais['v0'].values
    w0_vals = condicoes_iniciais['w0'].values
    cond_iniciais_tensor = torch.tensor(
        np.column_stack([v0_vals, w0_vals]), 
        dtype=torch.float32, 
        device=device
    )
    
    # parâmetros
    epsilon = parametros_oscilador['parametro_epsilon'].tolist()
    a = parametros_oscilador['parametro_a'].values[0]
    b = parametros_oscilador['parametro_b'].values[0]
    I = parametros_oscilador['parametro_I'].values[0]
    R = parametros_oscilador['parametro_R'].values[0]
    
    oscilador = OsciladorFitzHughNagumo(
        parametros_epsilon=epsilon,
        a=a,
        b=b,
        I=I,
        R=R,
        device=device
    )
    
    # tempo de simulação - garantir tempo suficiente para convergência
    periodo_estimado = parametros_oscilador['periodo_estimado'].values[0]
    # sistemas que não oscilam, usamos um tempo fixo
    if parametros_oscilador['possui_oscilacao'].values[0]:
        t_final = sim_params['num_periodos'] * max(periodo_estimado, 10.0)
    else:
        # sistemas estáveis, um tempo fixo é suficiente
        t_final = sim_params.get('t_final_padrao', 100.0)
    
    # simulação
    solucao = oscilador.resolve_multi_condicoes_sistemas(
        condicoes_iniciais=cond_iniciais_tensor,
        t_final=t_final,
        dt=sim_params['dt']
    )
    
    metadados = {
        'dispositivo': device,
        't_final': float(t_final),
        'n_passos': int(t_final / sim_params['dt']),
        'n_sistemas': int(oscilador.n_sistemas),
        'n_condicoes': int(len(condicoes_iniciais)),
        'total_trajetorias': int(oscilador.n_sistemas * len(condicoes_iniciais)),
        'periodo_estimado': float(periodo_estimado),
        'data_execucao': datetime.now().isoformat(),
        'parametros': {
            'epsilon': float(epsilon[0]) if epsilon else 0.0,
            'a': float(a),
            'b': float(b),
            'I': float(I),
            'R': float(R)
        },
        'possui_oscilacao': bool(parametros_oscilador['possui_oscilacao'].values[0])
    }
    
    return solucao, metadados


def gera_base_consolidada_node(
    solucao: Dict[str, Any],
    parametros_oscilador: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constrói a base de dados consolidada com verificações do sistema não conservativo.
    """
    dados = []
    n_condicoes = solucao['n_condicoes']
    n_sistemas = solucao['n_sistemas']
    n_passos = len(solucao['tempo'])
    
    convergiu_ciclo = solucao.get('convergiu_ciclo_limite', np.zeros((n_condicoes, n_sistemas)))
    periodicidade = solucao.get('periodicidade', np.ones((n_condicoes, n_sistemas)))
    convergencia_universal = solucao.get('convergencia_universal', np.ones((n_sistemas,)))
    
    # amplitude do potencial para cada condição inicial e sistema
    amplitude_potencial = solucao.get('amplitude_potencial', np.zeros((n_condicoes, n_sistemas)))
    amplitude_recuperacao = solucao.get('amplitude_recuperacao', np.zeros((n_condicoes, n_sistemas)))
    
    # energia média inicial e final
    energia_inicial_media = solucao.get('energia_inicial_media', np.zeros((n_condicoes, n_sistemas)))
    energia_final_media = solucao.get('energia_final_media', np.zeros((n_condicoes, n_sistemas)))
    variacao_energia = solucao.get('variacao_energia', np.zeros((n_condicoes, n_sistemas)))
    
    # amplitude teórica do ciclo limite
    amp_ciclo_teorica_v = solucao.get('amplitude_ciclo_limite_teorica_v', 2.0 * np.ones((n_sistemas,)))
    amp_ciclo_teorica_w = solucao.get('amplitude_ciclo_limite_teorica_w', (2.0 + parametros_oscilador['parametro_a'].values[0]) / parametros_oscilador['parametro_b'].values[0] * np.ones((n_sistemas,)))
    
    # parâmetros do sistema
    a = parametros_oscilador['parametro_a'].values[0] if 'parametro_a' in parametros_oscilador else 0.7
    b = parametros_oscilador['parametro_b'].values[0] if 'parametro_b' in parametros_oscilador else 0.8
    I = parametros_oscilador['parametro_I'].values[0] if 'parametro_I' in parametros_oscilador else 0.5
    R = parametros_oscilador['parametro_R'].values[0] if 'parametro_R' in parametros_oscilador else 0.1
    
    for i_sistema in range(n_sistemas):
        for i_cond in range(n_condicoes):
            id_traj = f"sistema_{i_sistema}_condicao_{i_cond}"
            
            for j in range(n_passos):
                convergiu = convergiu_ciclo[i_cond, i_sistema] if isinstance(convergiu_ciclo, np.ndarray) else convergiu_ciclo
                
                periodicidade_val = periodicidade[i_cond, i_sistema] if isinstance(periodicidade, np.ndarray) else periodicidade
                
                universalidade_val = convergencia_universal[i_sistema] if isinstance(convergencia_universal, np.ndarray) else convergencia_universal
                
                dados.append({
                    'sistema_id': i_sistema,
                    'simulacao_id': i_cond,
                    'id_trajetoria': id_traj,
                    'tempo': float(solucao['tempo'][j]),
                    'potencial': float(solucao['potencial'][j, i_cond, i_sistema]),
                    'recuperacao': float(solucao['recuperacao'][j, i_cond, i_sistema]),
                    'descricao_sistema': parametros_oscilador.iloc[i_sistema]['descricao_sistema'],
                    'parametro_epsilon': float(solucao['parametros_epsilon'][i_sistema]),
                    'parametro_a': float(a),
                    'parametro_b': float(b),
                    'parametro_I': float(I),
                    'parametro_R': float(R),
                    'periodo_estimado': float(parametros_oscilador.iloc[i_sistema]['periodo_estimado']),
                    'classificacao_epsilon': parametros_oscilador.iloc[i_sistema]['classificacao_epsilon'],
                    'possui_oscilacao': int(parametros_oscilador.iloc[i_sistema]['possui_oscilacao']),
                    'I_critical_min': float(parametros_oscilador.iloc[i_sistema]['I_critical_min']),
                    'I_critical_max': float(parametros_oscilador.iloc[i_sistema]['I_critical_max']),
                    'v0': float(solucao['condicoes_iniciais'][i_cond, 0]),
                    'w0': float(solucao['condicoes_iniciais'][i_cond, 1]),
                    'potencial_eq': float(solucao['potencial_eq'][i_sistema]),
                    'recuperacao_eq': float(solucao['recuperacao_eq'][i_sistema]),
                    
                    # sistema não conservativo
                    'convergiu_ciclo_limite': int(convergiu) if isinstance(convergiu, (bool, np.bool_)) else int(convergiu),
                    'periodicidade': float(periodicidade_val),
                    'convergencia_universal': float(universalidade_val),
                    
                    # métricas do ciclo limite - potencial
                    'amplitude_potencial': float(amplitude_potencial[i_cond, i_sistema]) if isinstance(amplitude_potencial, np.ndarray) else float(amplitude_potencial),
                    'amplitude_ciclo_limite_teorica_v': float(amp_ciclo_teorica_v[i_sistema]),
                    'erro_amplitude_v_relativo': abs(float(amplitude_potencial[i_cond, i_sistema]) - float(amp_ciclo_teorica_v[i_sistema])) / float(amp_ciclo_teorica_v[i_sistema]) if amp_ciclo_teorica_v[i_sistema] > 0 else 0.0,
                    
                    # métricas do ciclo limite - recuperação
                    'amplitude_recuperacao': float(amplitude_recuperacao[i_cond, i_sistema]) if isinstance(amplitude_recuperacao, np.ndarray) else float(amplitude_recuperacao),
                    'amplitude_ciclo_limite_teorica_w': float(amp_ciclo_teorica_w[i_sistema]),
                    'erro_amplitude_w_relativo': abs(float(amplitude_recuperacao[i_cond, i_sistema]) - float(amp_ciclo_teorica_w[i_sistema])) / float(amp_ciclo_teorica_w[i_sistema]) if amp_ciclo_teorica_w[i_sistema] > 0 else 0.0,
                    
                    # erro médio das amplitudes
                    'erro_amplitude_medio': (abs(float(amplitude_potencial[i_cond, i_sistema]) - float(amp_ciclo_teorica_v[i_sistema])) / float(amp_ciclo_teorica_v[i_sistema]) + 
                                            abs(float(amplitude_recuperacao[i_cond, i_sistema]) - float(amp_ciclo_teorica_w[i_sistema])) / float(amp_ciclo_teorica_w[i_sistema])) / 2.0 if amp_ciclo_teorica_v[i_sistema] > 0 and amp_ciclo_teorica_w[i_sistema] > 0 else 0.0,
                    
                    # energia média (pseudo-energia)
                    'energia_inicial_media': float(energia_inicial_media[i_cond, i_sistema]) if isinstance(energia_inicial_media, np.ndarray) else float(energia_inicial_media),
                    'energia_final_media': float(energia_final_media[i_cond, i_sistema]) if isinstance(energia_final_media, np.ndarray) else float(energia_final_media),
                    'variacao_energia': float(variacao_energia[i_cond, i_sistema]) if isinstance(variacao_energia, np.ndarray) else float(variacao_energia),
                    'energia_estabilizada': int(abs(float(variacao_energia[i_cond, i_sistema])) < 0.1 * float(energia_final_media[i_cond, i_sistema])) if isinstance(variacao_energia, np.ndarray) and energia_final_media[i_cond, i_sistema] > 0 else 0,
                })
    
    df = pd.DataFrame(dados)
    df['id_trajetoria'] = df['id_trajetoria'].astype(str)
    return df


def cria_visualizacoes_node(
    solucao: Dict[str, Any],
    parametros_oscilador: pd.DataFrame
) -> None:
    """
    Cria visualizações 2D e 3D do espaço de fases.
    """
    data_version = os.environ.get('DATA_VERSION', 'base_01')
    output_dir = f"data/05_model_input/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    descricoes = parametros_oscilador['descricao_sistema'].tolist()
    
    fig2d = cria_grafico_2d(solucao, descricoes)
    
    fig2d.write_html(f"{output_dir}/espaco_fases_2d.html")
    
    fig2d.show()