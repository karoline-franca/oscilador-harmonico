"""
Definição do pipeline MLP para o oscilador de Van der Pol.
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    prepara_dados_mlp_node,
    visualiza_distribuicao_dados_separado,
    cria_modelo_mlp_node,
    treina_mlp_node,
    avalia_metricas_mlp_node,
    visualiza_previsoes_mlp_node,
    visualiza_previsoes_espaco_fases_node,
    interpola_trajetorias_avulsas_node,
    interpolacoes_pontuais_mlp_node,
    interpola_entre_trajetorias_mlp_node,
    interpola_trajetorias_mlp_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Cria o pipeline de treinamento da MLP para o oscilador de Van der Pol.
    
    Pipeline:
        1. Prepara dados (treino, validação, teste)
        2. Visualiza distribuição dos dados
        3. Cria modelo MLP
        4. Treina modelo
        5. Avalia métricas
        6. Visualiza previsões
        7. Visualiza espaço de fases
        8. Interpola trajetórias avulsas
        9. Interpolações pontuais
        10. Interpola entre trajetórias
        11. Interpola trajetórias
    """
    
    return Pipeline([
        
        node(
            func=prepara_dados_mlp_node,
            inputs=["base_oscilador", "parameters"],
            outputs=["X_train", "y_train", "X_val", "y_val", "X_test", "y_test", 
                    "input_dim", "output_dim", "scaler_X", "scaler_y",
                    "trajetorias_train", "trajetorias_val", "trajetorias_test",
                    "num_timesteps", "tempos_referencia"],
            name="node_prepara_dados_mlp",
            tags=["data_preparation", "mlp"]
        ),

        node(
            func=visualiza_distribuicao_dados_separado,
            inputs=["base_oscilador", "parameters"],
            outputs=None,
            name="node_visualiza_distribuicao_dados",
            tags=["visualization", "eda"]
        ),
 
        node(
            func=cria_modelo_mlp_node,
            inputs=["input_dim", "output_dim", "parameters"],
            outputs="modelo_mlp",
            name="node_cria_modelo_mlp",
            tags=["model_creation", "mlp"]
        ),
        
        node(
            func=treina_mlp_node,
            inputs=["modelo_mlp", "X_train", "y_train", "X_val", "y_val", "parameters"],
            outputs=["modelo_mlp_treinado", "history_mlp"],
            name="node_treina_mlp",
            tags=["training", "mlp"]
        ),
        
        node(
            func=avalia_metricas_mlp_node,
            inputs=["modelo_mlp_treinado", "X_val", "y_val", "X_test", "y_test", "scaler_y"],
            outputs="metricas_mlp",
            name="node_avalia_metricas_mlp",
            tags=["evaluation", "mlp"]
        ),
        
        node(
            func=visualiza_previsoes_mlp_node,
            inputs=["modelo_mlp_treinado", "X_test", "y_test", "scaler_y", "parameters"],
            outputs=None,
            name="node_visualiza_previsoes_mlp",
            tags=["visualization", "predictions"]
        ),

        node(
            func=visualiza_previsoes_espaco_fases_node,
            inputs=["modelo_mlp_treinado", "X_test", "y_test", "scaler_y", "parameters", "tempos_referencia"],
            outputs=None,
            name="node_visualiza_previsoes_espaco_fases",
            tags=["visualization", "phase_space"]
        ),

        node(
            func=interpola_trajetorias_avulsas_node,
            inputs=["modelo_mlp_treinado", "scaler_X", "scaler_y", "parameters", "tempos_referencia"],
            outputs=None,
            name="node_interpola_trajetorias_avulsas",
            tags=["interpolation", "trajectories"]
        ),

        node(
            func=interpolacoes_pontuais_mlp_node,
            inputs=["modelo_mlp_treinado", "scaler_X", "scaler_y", "parameters", "tempos_referencia"],
            outputs="base_interpolada_pontual",
            name="node_interpolacoes_pontuais_mlp",
            tags=["interpolation", "pointwise", "database"]
        ),

        node(
            func=interpola_entre_trajetorias_mlp_node,
            inputs=["modelo_mlp_treinado", "scaler_X", "scaler_y", "parameters", "tempos_referencia"],
            outputs="base_interpolada_entre_trajetorias",
            name="node_interpola_entre_trajetorias_mlp",
            tags=["interpolation", "between_trajectories", "database"]
        ),

        node(
            func=interpola_trajetorias_mlp_node,
            inputs=["modelo_mlp_treinado", "scaler_X", "scaler_y", "parameters", "tempos_referencia"],
            outputs="base_interpolada_trajetorias",
            name="node_interpola_trajetorias_mlp",
            tags=["interpolation", "trajectories", "database"]
        ),

    ])