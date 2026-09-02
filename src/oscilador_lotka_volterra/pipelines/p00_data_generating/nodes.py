"""
Nodes do pipeline Kedro para o oscilador de Lotka-Volterra.
"""

import numpy as np
import pandas as pd
import torch
from oscilador_lotka_volterra.utils import cria_grafico_2d
import os
from datetime import datetime
from typing import Dict, Any, Tuple

from .olv import OsciladorLotkaVolterra


def gera_condicoes_iniciais_node(parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Gera condições iniciais aleatórias para presas (x0) e predadores (y0).
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
    Gera parâmetros do sistema Lotka-Volterra (a, b, c, d).
    """
    intervals = parameters['intervals']
    
    a = intervals['taxa_crescimento']
    b = intervals['taxa_predacao']
    c = intervals['taxa_mortalidade']
    d = intervals['taxa_eficiencia']
    
    desc_a = "Baixa" if a < 1.0 else "Média" if a < 2.0 else "Alta"
    desc_b = "Baixa" if b < 0.3 else "Média" if b < 0.6 else "Alta"
    desc_c = "Baixa" if c < 1.0 else "Média" if c < 2.0 else "Alta"
    desc_d = "Baixa" if d < 0.15 else "Média" if d < 0.3 else "Alta"
    
    descricao = f"cresc:{desc_a}, pred:{desc_b}, mort:{desc_c}, ef:{desc_d}"
    
    return pd.DataFrame({
        'taxa_crescimento_a': [a],
        'taxa_predacao_b': [b],
        'taxa_mortalidade_c': [c],
        'taxa_eficiencia_d': [d],
        'descricao_sistema': [descricao]
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
    
    # tempo de simulação
    periodo_lento = oscilador.periodos.cpu().numpy()[0]
    t_final = sim_params['num_periodos'] * periodo_lento
    
    # simulação
    solucao = oscilador.resolve_multi_condicoes_sistemas(
        condicoes_iniciais=cond_iniciais_tensor,
        t_final=t_final,
        dt=sim_params['dt']
    )
    
    metadados = {
        'dispositivo': device,
        't_final': t_final,
        'n_passos': int(t_final / sim_params['dt']),
        'n_sistemas': oscilador.n_sistemas,
        'n_condicoes': len(condicoes_iniciais),
        'total_trajetorias': oscilador.n_sistemas * len(condicoes_iniciais),
        'periodo_lento': float(periodo_lento),
        'data_execucao': datetime.now().isoformat(),
        'parametros': {'a': a[0], 'b': b[0], 'c': c[0], 'd': d[0]}
    }
    
    return solucao, metadados


def gera_base_consolidada_node(
    solucao: Dict[str, Any],
    parametros_oscilador: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constrói a base de dados consolidada.
    """
    dados = []
    n_condicoes = solucao['n_condicoes']
    n_sistemas = solucao['n_sistemas']
    n_passos = len(solucao['tempo'])
    
    for i_sistema in range(n_sistemas):
        for i_cond in range(n_condicoes):
            id_traj = f"sistema_{i_sistema}_condicao_{i_cond}"
            
            for j in range(n_passos):
                dados.append({
                    'sistema_id': i_sistema,
                    'simulacao_id': i_cond,
                    'id_trajetoria': id_traj,
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
    Cria visualizações 2D do espaço de fases.
    """
    data_version = os.environ.get('DATA_VERSION', 'base_01')
    output_dir = f"data/05_model_input/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    descricoes = parametros_oscilador['descricao_sistema'].tolist()
    
    fig2d = cria_grafico_2d(solucao, descricoes)
    fig2d.write_html(f"{output_dir}/espaco_fases_2d.html")
    fig2d.show()