"""
Nodes do pipeline Kedro para o Oscilador Harmônico Simples.
"""

import numpy as np
import pandas as pd
import torch
from oscilador_harmonico.utils import cria_grafico_3d, cria_grafico_2d
import os
from datetime import datetime
from typing import Dict, Any, Tuple

from .ohs import OsciladorHarmonicoPyTorch 

import plotly.graph_objects as go

def gera_condicoes_iniciais_node(parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Node: Gera condições iniciais aleatórias.
    
    Args:
        parameters: Parâmetros do pipeline.
        
    Returns:
        DataFrame com condições iniciais (x0, v0).
    """
    intervals = parameters['intervals']
    n_condicoes = parameters['simulation']['n_condicoes_iniciais']
    seed = parameters['seed']
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    x0 = np.random.uniform(intervals['x0_min'], intervals['x0_max'], n_condicoes)
    v0 = np.random.uniform(intervals['v0_min'], intervals['v0_max'], n_condicoes)
    
    df = pd.DataFrame({
        'x0': x0,
        'v0': v0
    })
    
    return df


def gera_frequencias_angulares_node(parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Node: Gera frequência angular única (estratificada para compatibilidade).
    
    Args:
        parameters: Parâmetros do pipeline.
        
    Returns:
        DataFrame com frequência angular para o sistema único.
    """
    intervals = parameters['intervals']
    seed = parameters['seed']
    
    if seed is not None:
        np.random.seed(seed)
    
    # frequência única definida no arquivo parameters.yml
    omega = intervals['omega']
    
    # descrição baseada no valor da frequência
    if omega < 1.0:
        descricao = "Muito Lento"
    elif omega < 3.0:
        descricao = "Lento"
    elif omega < 6.0:
        descricao = "Médio"
    elif omega < 9.0:
        descricao = "Rápido"
    else:
        descricao = "Muito Rápido"
    
    df = pd.DataFrame({
        'frequencia_angular_rads': [omega],
        'descricao_sistema': [descricao]
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
    
    # encontra sistema mais lento (único sistema)
    idx_lento = np.argmin(frequencias)
    periodo_lento = oscilador.periodos.cpu().numpy()[idx_lento]
    
    # calcula t_final
    t_final_calculado = sim_params['num_periodos'] * periodo_lento
    n_passos = int(np.ceil(t_final_calculado / sim_params['dt']))
    t_final = n_passos * sim_params['dt']
    
    # executa simulação - retorna apenas o dicionário solução
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
    frequencias_angulares: pd.DataFrame,
) -> pd.DataFrame:
    """
    Node: Constrói a base de dados consolidada.
    """
    dados = []
    
    n_condicoes = solucao['n_condicoes']
    n_sistemas = solucao['n_sistemas']
    n_passos = len(solucao['tempo'])
    
    for i_sistema in range(n_sistemas):
        for i_cond in range(n_condicoes):
            # cria identificador único da trajetória
            id_trajetoria = f"sistema_{i_sistema}_condicao_{i_cond}"
            
            for j in range(n_passos):
                dados.append({
                    'sistema_id': i_sistema,
                    'simulacao_id': i_cond,
                    'id_trajetoria': id_trajetoria,
                    'tempo': float(solucao['tempo'][j]),
                    'posicao': float(solucao['posicao'][j, i_cond, i_sistema]),
                    'velocidade': float(solucao['velocidade'][j, i_cond, i_sistema]),
                    'descricao_sistema': frequencias_angulares.iloc[i_sistema]['descricao_sistema'],
                    'frequencia_angular': float(solucao['frequencias_angulares'][i_sistema]),
                    'frequencia_linear': float(solucao['frequencias_lineares'][i_sistema]),
                    'periodo_s': float(solucao['periodos'][i_sistema]),
                    'x0': float(solucao['condicoes_iniciais'][i_cond, 0]),
                    'v0': float(solucao['condicoes_iniciais'][i_cond, 1]),
                    'amplitude_max': float(solucao['amplitudes'][i_cond, i_sistema]),
                    'energia_cinetica': float(solucao['energia_cinetica'][j, i_cond, i_sistema]),
                    'energia_potencial': float(solucao['energia_potencial'][j, i_cond, i_sistema]),
                    'energia_mecanica': float(solucao['energia_mecanica'][j, i_cond, i_sistema]),
                })
    
    df = pd.DataFrame(dados)
    df['id_trajetoria'] = df['id_trajetoria'].astype(str)

    return df


def cria_visualizacoes_node(
    solucao: Dict[str, Any],
    frequencias_angulares: pd.DataFrame
) -> None:
    """
    Node: Cria visualizações 2D e 3D e salva como HTML.
    
    Args:
        solucao: Dicionário com resultados.
        frequencias_angulares: DataFrame com frequências.
    """
    
    data_version = os.environ.get('DATA_VERSION', 'base_01')
    
    output_dir = f"data/08_reporting/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_3d_path = f"{output_dir}/espaco_fases_3d.html"
    grafico_2d_path = f"{output_dir}/espaco_fases_2d.html"
    
    descricoes = frequencias_angulares['descricao_sistema'].tolist()
    
    fig3d = cria_grafico_3d(solucao, descricoes)
    fig2d = cria_grafico_2d(solucao, descricoes)
    
    if fig3d is not None:
        fig3d.write_html(grafico_3d_path)

    else:
        print("ERRO: fig3d é None")
        fig3d = go.Figure()
        fig3d.update_layout(title="Erro ao gerar gráfico 3D")
        fig3d.write_html(grafico_3d_path)
    
    if fig2d is not None:
        fig2d.write_html(grafico_2d_path)
    else:
        print("ERRO: fig2d é None")
        fig2d = go.Figure()
        fig2d.update_layout(title="Erro ao gerar gráfico 2D")
        fig2d.write_html(grafico_2d_path)
    
    fig2d.show()
    fig3d.show()

    return None