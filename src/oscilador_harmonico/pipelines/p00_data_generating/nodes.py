"""
Nodes do pipeline Kedro.
"""

import numpy as np
import pandas as pd
import torch
from oscilador_harmonico.utils import (
    cria_grafico_3d, 
    cria_grafico_2d
)
import os
from datetime import datetime
from typing import Dict, Any, Tuple

from .olv import OsciladorLotkaVolterra

import plotly.graph_objects as go


def gera_condicoes_iniciais_node(parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Node: Gera condições iniciais aleatórias para presas e predadores.
    
    Args:
        parameters: Parâmetros do pipeline.
        
    Returns:
        DataFrame com condições iniciais (x0, y0).
    """
    intervals = parameters['intervals']
    n_condicoes = parameters['simulation']['n_condicoes_iniciais']
    seed = parameters['seed']
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    x0 = np.random.uniform(intervals['x0_min'], intervals['x0_max'], n_condicoes)
    y0 = np.random.uniform(intervals['y0_min'], intervals['y0_max'], n_condicoes)
    
    df = pd.DataFrame({
        'x0': x0,  # presas iniciais
        'y0': y0   # predadores iniciais
    })
    
    return df


def gera_parametros_oscilador_node(parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Node: Gera parâmetros do sistema Lotka-Volterra (a, b, c, d).
    
    Args:
        parameters: Parâmetros do pipeline.
        
    Returns:
        DataFrame com parâmetros do sistema (a, b, c, d).
    """
    intervals = parameters['intervals']
    seed = parameters['seed']
    
    if seed is not None:
        np.random.seed(seed)
    
    a = intervals['taxa_crescimento']       # taxa de crescimento das presas
    b = intervals['taxa_predacao']          # taxa de predação
    c = intervals['taxa_mortalidade']       # taxa de mortalidade dos predadores
    d = intervals['taxa_eficiencia']        # eficiência de conversão
    
    # crescimento das presas
    if a < 1.0:
        desc_crescimento = "Baixa"
    elif a < 2.0:
        desc_crescimento = "Média"
    else:
        desc_crescimento = "Alta"
    
    # predação
    if b < 0.3:
        desc_predacao = "Baixa"
    elif b < 0.6:
        desc_predacao = "Média"
    else:
        desc_predacao = "Alta"
    
    # mortalidade dos predadores
    if c < 1.0:
        desc_mortalidade = "Baixa"
    elif c < 2.0:
        desc_mortalidade = "Média"
    else:
        desc_mortalidade = "Alta"
    
    # eficiência de conversão
    if d < 0.15:
        desc_eficiencia = "Baixa"
    elif d < 0.3:
        desc_eficiencia = "Média"
    else:
        desc_eficiencia = "Alta"
    
    descricao = (f"cresc:{desc_crescimento}, "
                 f"pred:{desc_predacao}, "
                 f"mort:{desc_mortalidade}, "
                 f"ef:{desc_eficiencia}")
    
    df = pd.DataFrame({
        'taxa_crescimento_a': [a],
        'taxa_predacao_b': [b],
        'taxa_mortalidade_c': [c],
        'taxa_eficiencia_d': [d],
        'descricao_sistema': [descricao]
    })
    
    return df


def executa_simulacao_rk4_node(
    condicoes_iniciais: pd.DataFrame,
    parametros_oscilador: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Node: Executa a simulação RK4 para todos os sistemas.
    
    Args:
        condicoes_iniciais: DataFrame com condições iniciais (x0, y0)
        parametros_oscilador: DataFrame com parâmetros (a, b, c, d)
        parameters: Parâmetros do pipeline
        
    Returns:
        Tuple com solução e metadados da simulação
    """
    sim_params = parameters['simulation']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    x0_vals = condicoes_iniciais['x0'].values
    y0_vals = condicoes_iniciais['y0'].values
    cond_iniciais_tensor = torch.tensor(
        np.column_stack([x0_vals, y0_vals]), 
        dtype=torch.float32, 
        device=device
    )
    
    a = parametros_oscilador['taxa_crescimento_a'].tolist()
    b = parametros_oscilador['taxa_predacao_b'].tolist()
    c = parametros_oscilador['taxa_mortalidade_c'].tolist()
    d = parametros_oscilador['taxa_eficiencia_d'].tolist()
    
    oscilador = OsciladorLotkaVolterra(
        taxas_crescimento=a,
        taxas_mortalidade=c,
        taxas_predacao=b,
        taxas_eficiencia=d,
        device=device
    )
    
    idx_lento = np.argmin(oscilador.frequencias_angulares.cpu().numpy())
    periodo_lento = oscilador.periodos.cpu().numpy()[idx_lento]
    
    t_final_calculado = sim_params['num_periodos'] * periodo_lento
    n_passos = int(np.ceil(t_final_calculado / sim_params['dt']))
    t_final = n_passos * sim_params['dt']
    
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
        'data_execucao': datetime.now().isoformat(),
        'parametros': {
            'a': a,
            'b': b,
            'c': c,
            'd': d
        }
    }
    
    return solucao, metadados


def gera_base_consolidada_node(
    solucao: Dict[str, Any],
    parametros_oscilador: pd.DataFrame,
) -> pd.DataFrame:
    """
    Node: Constrói a base de dados consolidada para o oscilador de Lotka-Volterra.
    
    Args:
        solucao: Dicionário com resultados da simulação
        parametros_oscilador: DataFrame com parâmetros do sistema
        
    Returns:
        DataFrame consolidado com todos os dados
    """
    dados = []
    
    n_condicoes = solucao['n_condicoes']
    n_sistemas = solucao['n_sistemas']
    n_passos = len(solucao['tempo'])
    
    for i_sistema in range(n_sistemas):
        for i_cond in range(n_condicoes):

            id_trajetoria = f"sistema_{i_sistema}_condicao_{i_cond}"
            
            for j in range(n_passos):
                dados.append({
                    'sistema_id': i_sistema,
                    'simulacao_id': i_cond,
                    'id_trajetoria': id_trajetoria,
                    'tempo': float(solucao['tempo'][j]),
                    'presas': float(solucao['presas'][j, i_cond, i_sistema]),
                    'predadores': float(solucao['predadores'][j, i_cond, i_sistema]),
                    'descricao_sistema': parametros_oscilador.iloc[i_sistema]['descricao_sistema'],
                    'taxa_crescimento_a': float(solucao['taxas_crescimento'][i_sistema]),
                    'taxa_predacao_b': float(solucao['taxas_predacao'][i_sistema]),
                    'taxa_mortalidade_c': float(solucao['taxas_mortalidade'][i_sistema]),
                    'taxa_eficiencia_d': float(solucao['taxas_eficiencia'][i_sistema]),
                    'frequencia_angular': float(solucao['frequencias_angulares'][i_sistema]),
                    'periodo_s': float(solucao['periodos'][i_sistema]),
                    'x0': float(solucao['condicoes_iniciais'][i_cond, 0]),
                    'y0': float(solucao['condicoes_iniciais'][i_cond, 1]),
                    'presas_eq': float(solucao['presas_eq'][i_sistema]),
                    'predadores_eq': float(solucao['predadores_eq'][i_sistema]),
                    'amplitude_presas': float(solucao['amplitudes_presas'][i_cond, i_sistema]),
                    'amplitude_predadores': float(solucao['amplitudes_predadores'][i_cond, i_sistema]),
                    'constante_movimento': float(solucao['constante_movimento'][j, i_cond, i_sistema]),
                })
    
    df = pd.DataFrame(dados)
    df['id_trajetoria'] = df['id_trajetoria'].astype(str)
    
    return df


def cria_visualizacoes_node(
    solucao: Dict[str, Any],
    parametros_oscilador: pd.DataFrame
) -> None:
    """
    Node: Cria visualizações 2D, 3D e específicas do Lotka-Volterra.
    
    Args:
        solucao: Dicionário com resultados
        parametros_oscilador: DataFrame com parâmetros do sistema
    """
    
    data_version = os.environ.get('DATA_VERSION', 'base_01')
    
    output_dir = f"data/08_reporting/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_3d_path = f"{output_dir}/espaco_fases_3d.html"
    grafico_2d_path = f"{output_dir}/espaco_fases_2d.html"
    
    descricoes = parametros_oscilador['descricao_sistema'].tolist()
    
    fig3d = cria_grafico_3d(solucao, descricoes)
    fig2d = cria_grafico_2d(solucao, descricoes)

    fig3d.write_html(grafico_3d_path)
    fig2d.write_html(grafico_2d_path)
    
    fig2d.show()
    fig3d.show()
    
    return None