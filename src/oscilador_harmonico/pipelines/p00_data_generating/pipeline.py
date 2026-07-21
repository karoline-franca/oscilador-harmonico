"""
Definição do pipeline Kedro.
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    gera_condicoes_iniciais_node,
    gera_parametros_oscilador_node,
    executa_simulacao_rk4_node,
    gera_base_consolidada_node,
    cria_visualizacoes_node
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Cria o pipeline completo do oscilador de Lotka-Volterra.
    
    Pipeline:
        1. Gera condições iniciais (presas e predadores)
        2. Gera parâmetros do sistema (a, b, c, d)
        3. Executa simulação RK4
        4. Gera base consolidada
        5. Cria visualizações
    """
    
    return Pipeline([
        
        node(
            func=gera_condicoes_iniciais_node,
            inputs="parameters",
            outputs="condicoes_iniciais",
            name="node_gera_condicoes_iniciais",
            tags=["generation", "initial_conditions"]
        ),
        
        node(
            func=gera_parametros_oscilador_node,
            inputs="parameters",
            outputs="parametros_oscilador",
            name="node_gera_parametros_oscilador",
            tags=["generation", "parameters"]
        ),
        
        node(
            func=executa_simulacao_rk4_node,
            inputs=["condicoes_iniciais", "parametros_oscilador", "parameters"],
            outputs=["solucao_rk4", "metadata_simulacao"],
            name="node_executa_simulacao_rk4",
            tags=["simulation", "rk4"]
        ),
        
        node(
            func=gera_base_consolidada_node,
            inputs=["solucao_rk4", "parametros_oscilador"],
            outputs="base_oscilador",
            name="node_gera_base_consolidada",
            tags=["data", "database"]
        ),
        
        node(
            func=cria_visualizacoes_node,
            inputs=["solucao_rk4", "parametros_oscilador"],
            outputs=None,
            name="node_cria_visualizacoes",
            tags=["visualization", "plotly"]
        ),
        
    ])