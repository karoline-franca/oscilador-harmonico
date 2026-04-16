"""
Definição do pipeline MLP.
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    prepara_dados_mlp_node,
    cria_modelo_mlp_node,
    treina_mlp_node,
    avalia_mlp_node,
    visualiza_previsoes_mlp_node,
    salva_modelo_mlp_node
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Cria o pipeline de treinamento da MLP.
    """
    
    return Pipeline([
        
        node(
            func=prepara_dados_mlp_node,
            inputs=["base_oscilador", "parameters"],
            outputs=["X_train", "y_train", "X_test", "y_test", "X_val", "y_val", 
                     "input_dim", "output_dim", "scaler_X", "scaler_y"],
            name="node_prepara_dados_mlp",
            tags=["mlp", "preprocessing"]
        ),
        
        node(
            func=cria_modelo_mlp_node,
            inputs=["input_dim", "output_dim", "parameters"],
            outputs="modelo_mlp",
            name="node_cria_modelo_mlp",
            tags=["mlp", "model"]
        ),
        
        node(
            func=treina_mlp_node,
            inputs=["modelo_mlp", "X_train", "y_train", "X_test", "y_test", "parameters"],
            outputs=["modelo_mlp_treinado", "history_mlp"],
            name="node_treina_mlp",
            tags=["mlp", "training"]
        ),
        
        node(
            func=avalia_mlp_node,
            inputs=["modelo_mlp_treinado", "X_val", "y_val", "scaler_y"],
            outputs="metricas_mlp",
            name="node_avalia_mlp",
            tags=["mlp", "evaluation"]
        ),
        
        node(
            func=visualiza_previsoes_mlp_node,
            inputs=["modelo_mlp_treinado", "X_val", "y_val", "scaler_y", "parameters"],
            outputs=None,
            name="node_visualiza_previsoes_mlp",
            tags=["mlp", "visualization"]
        ),
        
        node(
            func=salva_modelo_mlp_node,
            inputs=["modelo_mlp_treinado", "scaler_X", "scaler_y", "parameters"],
            outputs=None,
            name="node_salva_modelo_mlp",
            tags=["mlp", "saving"]
        ),
        
    ])