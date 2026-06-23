"""
Definição do pipeline MLP.
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    interpola_trajetorias_mlp_node,
    prepara_dados_mlp_node,
    cria_modelo_mlp_node,
    treina_mlp_node,
    avalia_metricas_mlp_node,
    visualiza_previsoes_mlp_node,
    visualiza_distribuicao_dados_separado,
    visualiza_previsoes_espaco_fases_node,
    interpola_trajetorias_avulsas_node,
    interpolacoes_pontuais_mlp_node,
    interpola_entre_trajetorias_mlp_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Cria o pipeline de treinamento da MLP.
    """
    
    return Pipeline([
        
        node(
            func=prepara_dados_mlp_node,
            inputs=["base_oscilador", "parameters"],
            outputs=["X_train", "y_train", "X_val", "y_val", "X_test", "y_test", 
                    "input_dim", "output_dim", "scaler_X", "scaler_y",
                    "trajetorias_train", "trajetorias_val", "trajetorias_test"],
            name="node_prepara_dados_mlp",
        ),

        node(
            func=visualiza_distribuicao_dados_separado,
            inputs=["base_oscilador", "parameters"],
            outputs=None,
            name="node_visualiza_distribuicao_dados",
        ),
 
        node(
            func=cria_modelo_mlp_node,
            inputs=["input_dim", "output_dim", "parameters"],
            outputs="modelo_mlp",
            name="node_cria_modelo_mlp",
        ),
        
        node(
            func=treina_mlp_node,
            inputs=["modelo_mlp", "X_train", "y_train", "X_val", "y_val", "parameters"],
            outputs=["modelo_mlp_treinado", "history_mlp"],
            name="node_treina_mlp",
        ),
        
        node(
            func=avalia_metricas_mlp_node,
            inputs=["modelo_mlp_treinado", "X_val", "y_val", "X_test", "y_test", "scaler_y"],
            outputs="metricas_mlp",
            name="node_avalia_metricas_mlp",
        ),
        
        node(
            func=visualiza_previsoes_mlp_node,
            inputs=["modelo_mlp_treinado", "X_test", "y_test", "scaler_y", "parameters"],
            outputs=None,
            name="node_visualiza_previsoes_mlp",
        ),

        node(
            func=visualiza_previsoes_espaco_fases_node,
            inputs=["modelo_mlp_treinado", "X_test", "y_test", "scaler_y", "parameters"],
            outputs=None,
            name="node_visualiza_previsoes_espaco_fases",
        ),

        node(
            func=interpola_trajetorias_avulsas_node,
            inputs=["modelo_mlp_treinado", "scaler_X", "scaler_y", "parameters"],
            outputs=None,
            name="node_interpola_trajetorias_avulsas",
        ),

        node(
            func=interpolacoes_pontuais_mlp_node,
            inputs=["modelo_mlp_treinado", "scaler_X", "scaler_y", "parameters"],
            outputs="base_interpolada_pontual",
            name="node_interpolacoes_pontuais_mlp",
        ),

        node(
            func=interpola_entre_trajetorias_mlp_node,
            inputs=["modelo_mlp_treinado", "scaler_X", "scaler_y", "parameters"],
            outputs="base_interpolada_entre_trajetorias",
            name="node_interpola_entre_trajetorias_mlp",
        ),

        node(
            func=interpola_trajetorias_mlp_node,
            inputs=["modelo_mlp_treinado", "scaler_X", "scaler_y", "parameters"],
            outputs="base_interpolada_trajetorias",
            name="node_interpola_trajetorias_mlp",
        ),

    ])
