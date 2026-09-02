"""
Nodes do pipeline Kedro para o oscilador de Van der Pol.
"""

import numpy as np
import pandas as pd
import torch
from oscilador_van_der_pol.utils import cria_grafico_2d
import os
from datetime import datetime
from typing import Dict, Any, Tuple

from .vdp import OsciladorVanDerPol


def gera_condicoes_iniciais_node(parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Gera condições iniciais aleatórias para posição (x0) e velocidade (y0).
    """
    intervals = parameters['intervals']
    n_condicoes = parameters['simulation']['n_condicoes_iniciais']
    seed = parameters['seed']
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    x0 = np.random.uniform(intervals['x0_min'], intervals['x0_max'], n_condicoes)
    y0 = np.random.uniform(intervals['y0_min'], intervals['y0_max'], n_condicoes)
    
    return pd.DataFrame({'x0': x0, 'y0': y0})


def gera_parametros_oscilador_node(parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Gera parâmetros do sistema Van der Pol (mu).
    """
    intervals = parameters['intervals']
    
    mu = intervals['parametro_mu']
    
    # classificação qualitativa do comportamento
    if mu < 0.1:
        desc_mu = "Linear (mu muito pequeno)"
    elif mu < 0.5:
        desc_mu = "Fracamente não linear"
    elif mu < 2.0:
        desc_mu = "Moderadamente não linear"
    elif mu < 5.0:
        desc_mu = "Fortemente não linear"
    else:
        desc_mu = "Oscilação de relaxação (mu grande)"
    
    # estimativa do período
    if mu < 0.1:
        periodo_estimado = 2.0 * np.pi
    else:
        periodo_estimado = (3.0 - 2.0 * np.log(2.0)) / mu
    
    descricao = f"mu={mu:.3f}, {desc_mu}"
    
    return pd.DataFrame({
        'parametro_mu': [mu],
        'descricao_sistema': [descricao],
        'periodo_estimado': [periodo_estimado],
        'classificacao_mu': [desc_mu]
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
    x0_vals = condicoes_iniciais['x0'].values
    y0_vals = condicoes_iniciais['y0'].values
    cond_iniciais_tensor = torch.tensor(
        np.column_stack([x0_vals, y0_vals]), 
        dtype=torch.float32, 
        device=device
    )
    
    # parâmetros
    mu = parametros_oscilador['parametro_mu'].tolist()
    
    oscilador = OsciladorVanDerPol(
        parametros_mu=mu,
        device=device
    )
    
    # tempo de simulação - garantir tempo suficiente para convergência
    periodo_estimado = parametros_oscilador['periodo_estimado'].values[0]
    t_final = sim_params['num_periodos'] * max(periodo_estimado, 2.0 * np.pi)
    
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
        'parametros': {'mu': float(mu[0]) if mu else 0.0}
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
    periodicidade = solucao.get('periodicidade', np.ones((n_sistemas,)))
    convergencia_universal = solucao.get('convergencia_universal', np.ones((n_sistemas,)))
    
    # amplitude da posição para cada condição inicial e sistema
    amplitude_posicao = solucao.get('amplitude_posicao', np.zeros((n_condicoes, n_sistemas)))
    
    # energia média inicial e final
    energia_inicial_media = solucao.get('energia_inicial_media', np.zeros((n_condicoes, n_sistemas)))
    energia_final_media = solucao.get('energia_final_media', np.zeros((n_condicoes, n_sistemas)))
    variacao_energia = solucao.get('variacao_energia', np.zeros((n_condicoes, n_sistemas)))
    
    # amplitude teórica do ciclo limite
    amp_ciclo_teorica = solucao.get('amplitude_ciclo_limite_teorica', 2.0 * np.ones((n_sistemas,)))
    
    for i_sistema in range(n_sistemas):
        for i_cond in range(n_condicoes):
            id_traj = f"sistema_{i_sistema}_condicao_{i_cond}"
            
            for j in range(n_passos):
                convergiu = convergiu_ciclo[i_cond, i_sistema] if isinstance(convergiu_ciclo, np.ndarray) else convergiu_ciclo
                
                periodicidade_val = periodicidade[i_sistema] if isinstance(periodicidade, np.ndarray) else periodicidade
                
                universalidade_val = convergencia_universal[i_sistema] if isinstance(convergencia_universal, np.ndarray) else convergencia_universal
                
                dados.append({
                    'sistema_id': i_sistema,
                    'simulacao_id': i_cond,
                    'id_trajetoria': id_traj,
                    'tempo': float(solucao['tempo'][j]),
                    'posicao': float(solucao['posicao'][j, i_cond, i_sistema]),
                    'velocidade': float(solucao['velocidade'][j, i_cond, i_sistema]),
                    'descricao_sistema': parametros_oscilador.iloc[i_sistema]['descricao_sistema'],
                    'parametro_mu': float(solucao['parametros_mu'][i_sistema]),
                    'periodo_estimado': float(parametros_oscilador.iloc[i_sistema]['periodo_estimado']),
                    'classificacao_mu': parametros_oscilador.iloc[i_sistema]['classificacao_mu'],
                    'x0': float(solucao['condicoes_iniciais'][i_cond, 0]),
                    'y0': float(solucao['condicoes_iniciais'][i_cond, 1]),
                    'posicao_eq': float(solucao['posicao_eq'][i_sistema]),
                    'velocidade_eq': float(solucao['velocidade_eq'][i_sistema]),
                    
                    # sistema não conservativo
                    'convergiu_ciclo_limite': int(convergiu) if isinstance(convergiu, (bool, np.bool_)) else int(convergiu),
                    'periodicidade': float(periodicidade_val),
                    'convergencia_universal': float(universalidade_val),
                    
                    # métricas do ciclo limite
                    'amplitude_posicao': float(amplitude_posicao[i_cond, i_sistema]) if isinstance(amplitude_posicao, np.ndarray) else float(amplitude_posicao),
                    'amplitude_ciclo_limite_teorica': float(amp_ciclo_teorica[i_sistema]),
                    'erro_amplitude_relativo': abs(float(amplitude_posicao[i_cond, i_sistema]) - float(amp_ciclo_teorica[i_sistema])) / float(amp_ciclo_teorica[i_sistema]) if amp_ciclo_teorica[i_sistema] > 0 else 0.0,
                    
                    # energia média
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