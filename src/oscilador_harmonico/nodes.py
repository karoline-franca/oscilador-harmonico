"""
Nodes do pipeline Kedro para o Oscilador Harmônico Simples.
"""

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, Tuple

from oscilador_harmonico.utils import CORES_PALETA, formatar_numero_pt_br
from oscilador_harmonico.oscillator import OsciladorHarmonicoPyTorch


def gera_condicoes_iniciais_node(parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Node: Gera condições iniciais aleatórias.
    
    Args:
        parameters: Parâmetros do pipeline.
        
    Returns:
        DataFrame com condições iniciais (x0, v0).
    """
    intervals = parameters['intervals']
    n_condicoes = parameters['simulation']['n_condicoes_por_sistema']
    seed = parameters['seed']
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    x0_estratos = np.linspace(intervals['x0_min'], intervals['x0_max'], n_condicoes + 1)
    v0_estratos = np.linspace(intervals['v0_min'], intervals['v0_max'], n_condicoes + 1)
    
    x0 = []
    v0 = []
    
    for i in range(n_condicoes):
        x0_val = np.random.uniform(x0_estratos[i], x0_estratos[i+1])
        v0_val = np.random.uniform(v0_estratos[i], v0_estratos[i+1])
        x0.append(x0_val)
        v0.append(v0_val)
    
    # embaralha
    indices = np.random.permutation(n_condicoes)
    x0 = np.array(x0)[indices]
    v0 = np.array(v0)[indices]
    
    df = pd.DataFrame({
        'x0': x0,
        'v0': v0
    })
    
    return df


def gera_frequencias_angulares_node(parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Node: Gera frequências angulares aleatórias.
    
    Args:
        parameters: Parâmetros do pipeline.
        
    Returns:
        DataFrame com frequências angulares por sistema.
    """
    intervals = parameters['intervals']
    n_sistemas = parameters['simulation']['n_sistemas']
    seed = parameters['seed']
    
    if seed is not None:
        np.random.seed(seed)
    
    estratos = np.linspace(intervals['omega_min'], intervals['omega_max'], n_sistemas + 1)
    
    omegas = []
    for i in range(n_sistemas):
        omega = np.random.uniform(estratos[i], estratos[i+1])
        omegas.append(omega)
    
    np.random.shuffle(omegas)
    
    # cria descrições
    descricoes = []
    for omega in omegas:
        if omega < 1.0:
            descricoes.append("Lento")
        elif omega < 3.0:
            descricoes.append("Médio")
        elif omega < 8.0:
            descricoes.append("Rápido")
        else:
            descricoes.append("Muito Rápido")
    
    df = pd.DataFrame({
        'frequencia_angular_rads': omegas,
        'descricao_sistema': descricoes
    })
    
    return df

def executa_simulacao_rk4_node(
    condicoes_iniciais: pd.DataFrame,
    frequencias_angulares: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Node: Executa a simulação RK4 para todos os sistemas.
    """
    sim_params = parameters['simulation']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # prepara tensores
    x0_vals = condicoes_iniciais['x0'].values
    v0_vals = condicoes_iniciais['v0'].values
    cond_iniciais_tensor = torch.tensor(
        np.column_stack([x0_vals, v0_vals]), 
        dtype=torch.float32, 
        device=device
    )
    
    frequencias = frequencias_angulares['frequencia_angular_rads'].tolist()
    
    # cria oscilador
    oscilador = OsciladorHarmonicoPyTorch(
        frequencias_angulares=frequencias,
        device=device
    )
    
    # encontra sistema mais lento
    idx_lento = np.argmin(frequencias)
    periodo_lento = oscilador.periodos.cpu().numpy()[idx_lento]
    
    # calcula t_final
    t_final_calculado = sim_params['num_periodos'] * periodo_lento
    n_passos = int(np.ceil(t_final_calculado / sim_params['dt']))
    t_final = n_passos * sim_params['dt']
    
    # executa simulação - retorna apenas o dicionário solucao
    solucao = oscilador.resolve_multi_condicoes_sistemas(
        condicoes_iniciais=cond_iniciais_tensor,
        t_final=t_final,
        dt=sim_params['dt']
    )
    
    metadados = {
        'dispositivo': device,
        't_final': t_final,
        'n_passos': n_passos,
        'n_sistemas': oscilador.n_sistemas,
        'n_condicoes': len(condicoes_iniciais),
        'total_trajetorias': oscilador.n_sistemas * len(condicoes_iniciais),
        'periodo_lento': float(periodo_lento),
        'idx_sistema_lento': int(idx_lento),
        'data_execucao': datetime.now().isoformat()
    }
    
    return solucao, metadados


def gera_base_consolidada_node(
    solucao: Dict[str, Any],
    condicoes_iniciais: pd.DataFrame,
    frequencias_angulares: pd.DataFrame,
    metadados: pd.DataFrame
) -> pd.DataFrame:
    """
    Node: Constrói a base de dados consolidada.
    
    Args:
        solucao: Dicionário com resultados da simulação.
        condicoes_iniciais: DataFrame original.
        frequencias_angulares: DataFrame original.
        metadados: DataFrame com metadados.
        
    Returns:
        DataFrame consolidado.
    """
    dados = []
    
    n_condicoes = solucao['n_condicoes']
    n_sistemas = solucao['n_sistemas']
    n_passos = len(solucao['tempo'])
    
    for i_sistema in range(n_sistemas):
        for i_cond in range(n_condicoes):
            for j in range(n_passos):
                dados.append({
                    'sistema_id': i_sistema,
                    'simulacao_id': i_cond,
                    'tempo': formatar_numero_pt_br(solucao['tempo'][j]),
                    'posicao': formatar_numero_pt_br(solucao['posicao'][j, i_cond, i_sistema]),
                    'velocidade': formatar_numero_pt_br(solucao['velocidade'][j, i_cond, i_sistema]),
                    'descricao_sistema': frequencias_angulares.iloc[i_sistema]['descricao_sistema'],
                    'frequencia_angular': formatar_numero_pt_br(solucao['frequencias_angulares'][i_sistema]),
                    'frequencia_linear': formatar_numero_pt_br(solucao['frequencias_lineares'][i_sistema]),
                    'periodo_s': formatar_numero_pt_br(solucao['periodos'][i_sistema]),
                    'x0': formatar_numero_pt_br(solucao['condicoes_iniciais'][i_cond, 0]),
                    'v0': formatar_numero_pt_br(solucao['condicoes_iniciais'][i_cond, 1]),
                    'amplitude_max': formatar_numero_pt_br(solucao['amplitudes'][i_cond, i_sistema]),
                    'energia_cinetica': formatar_numero_pt_br(solucao['energia_cinetica'][j, i_cond, i_sistema]),
                    'energia_potencial': formatar_numero_pt_br(solucao['energia_potencial'][j, i_cond, i_sistema]),
                    'energia_mecanica': formatar_numero_pt_br(solucao['energia_mecanica'][j, i_cond, i_sistema]),
                })
    
    return pd.DataFrame(dados)

def cria_visualizacoes_node(
    solucao: Dict[str, Any],
    frequencias_angulares: pd.DataFrame
) -> None:
    """
    Node: Cria visualizações 2D e 3D e salva como HTML.
    
    Args:
        solucao: Dicionário com resultados.
        frequencias_angulares: DataFrame com frequências.
        
    Returns:
        Tupla com (figura_3d, figura_2d).
    """
    from oscilador_harmonico.utils import cria_grafico_3d, cria_grafico_2d
    
    descricoes = frequencias_angulares['descricao_sistema'].tolist()
    
    fig3d = cria_grafico_3d(solucao, descricoes)
    fig2d = cria_grafico_2d(solucao, descricoes)
    
    if fig3d is not None:
        fig3d.write_html("data/08_reporting/espaco_fases_3d.html")
        print("Gráfico 3D salvo em data/08_reporting/espaco_fases_3d.html")
    else:
        print("ERRO: fig3d é None")
    
    if fig2d is not None:
        fig2d.write_html("data/08_reporting/espaco_fases_2d.html")
        print("Gráfico 2D salvo em data/08_reporting/espaco_fases_2d.html")
    else:
        print("ERRO: fig2d é None")
    
    return None