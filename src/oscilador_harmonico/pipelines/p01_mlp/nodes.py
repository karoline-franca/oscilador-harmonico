# nodes saída x,v; entrada [x0, v0, ω, t]

"""
Nodes do pipeline MLP.
"""

import numpy as np
import pandas as pd
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any, Tuple
from .model import MLP
from oscilador_harmonico.utils import (
    cria_grafico_previsoes_mlp,
    cria_grafico_distribuicao_dados,
    cria_grafico_previsoes_espaco_fases,
    cria_grafico_interpolacao_completo,
    cria_grafico_interpolacao_espaco_fases,
    CORES_PALETA
)


def fixar_sementes(seed: int = 42):
    """Fixa todas as sementes para reprodutibilidade."""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepara_dados_mlp_node(base_oscilador: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple:
    """
    Prepara os dados para treinamento do MLP.
    
    Entrada: [x0, v0, frequencia_angular, tempo]
    Saída: [posicao, velocidade]
    
    Nota: Filtra apenas os dados do primeiro sistema (sistema_id = 0)
    """
    # filtra apenas os dados do primeiro sistema
    # base_oscilador = base_oscilador[base_oscilador['sistema_id'] == 0].copy()
    
    for col in base_oscilador.columns:
        if base_oscilador[col].dtype == 'object':
            try:
                base_oscilador[col] = base_oscilador[col].astype(str).str.replace(',', '.').astype(float)
            except:
                pass
    
    features_entrada = ['x0', 'v0', 'frequencia_angular', 'tempo']
    features_saida = ['posicao', 'velocidade']
    
    X_raw = base_oscilador[features_entrada].values.astype(np.float32)
    y_raw = base_oscilador[features_saida].values.astype(np.float32)
    
    print("\n=== SEPARAÇÃO DAS FEATURES DE ENTRADA E SAÍDA ===")
    print(f"  Tamanho original de X (entrada): {X_raw.shape}")
    print(f"  Tamanho original de y (saída): {y_raw.shape}")
    
    # normalização padrão das variáveis de entrada e saída
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)
    
    # 70% treino, 20% teste, 10% validação
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_scaled, test_size=0.30, random_state=42  # 30% temporário
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.3333, random_state=42  # 10% do total (33.33% dos 30%)
    )
    
    print(f"  Treino: {X_train.shape}, Teste: {X_test.shape}, Validação: {X_val.shape}")
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    return X_train, y_train, X_test, y_test, X_val, y_val, input_dim, output_dim, scaler_X, scaler_y


def visualiza_distribuicao_dados_separado(
    base_oscilador: pd.DataFrame, 
    parameters: Dict[str, Any]
) -> None:
    """
    Node separado para visualizar a distribuição dos dados no espaço de fases.
    Carrega os dados novamente e faz a divisão apenas para visualização.
    Não interfere no pipeline principal de treinamento.
    
    Args:
        base_oscilador: DataFrame com a base consolidada
        parameters: Parâmetros do pipeline
    """
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'default_v1')
    
    output_dir = f"data/08_reporting/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_distribuicao_dados = f"{output_dir}/distribuicao_dados.html"
    
    base_oscilador = base_oscilador[base_oscilador['sistema_id'] == 0].copy()
    
    for col in base_oscilador.columns:
        if base_oscilador[col].dtype == 'object':
            try:
                base_oscilador[col] = base_oscilador[col].astype(str).str.replace(',', '.').astype(float)
            except:
                pass
    
    posicao = base_oscilador['posicao'].values.astype(np.float32)
    velocidade = base_oscilador['velocidade'].values.astype(np.float32)
    y_combined = np.column_stack([posicao, velocidade])
    
    # 70% treino, 20% teste, 10% validação
    _, y_temp, _, _ = train_test_split(
        y_combined, y_combined, test_size=0.30, random_state=42  # 30% temporário
    )
    y_test, y_val, _, _ = train_test_split(
        y_temp, y_temp, test_size=0.3333, random_state=42  # 10% do total (33.33% dos 30%)
    )
    y_train = y_combined[:len(y_combined) - len(y_temp)]
    
    y_pos_train = y_train[:, 0].reshape(-1, 1)
    y_vel_train = y_train[:, 1].reshape(-1, 1)
    y_pos_test = y_test[:, 0].reshape(-1, 1)
    y_vel_test = y_test[:, 1].reshape(-1, 1)
    y_pos_val = y_val[:, 0].reshape(-1, 1)
    y_vel_val = y_val[:, 1].reshape(-1, 1)
    
    fig = cria_grafico_distribuicao_dados(
        y_pos_train=y_pos_train,
        y_vel_train=y_vel_train,
        y_pos_test=y_pos_test,
        y_vel_test=y_vel_test,
        y_pos_val=y_pos_val,
        y_vel_val=y_vel_val,
        titulo="Distribuição dos Dados - Espaço de Fases"
    )
    
    fig.write_html(grafico_distribuicao_dados)
    print(f"Gráfico de distribuição salvo em {grafico_distribuicao_dados}")
    
    fig.show()
    
    return None


def cria_modelo_mlp_node(input_dim: int, output_dim: int, parameters: Dict[str, Any]) -> nn.Module:
    """Cria o modelo MLP."""

    mlp_config = parameters.get('mlp', {})
    seed = parameters.get('seed', 42)
    
    fixar_sementes(seed)
    
    hidden_dims = mlp_config.get('hidden_dims', [64, 128, 64])
    activation = mlp_config.get('activation', 'relu')
    
    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=activation,
        seed=seed
    )
    
    print("\n=== MODELO MLP CRIADO ===")
    print(f"  Input dim: {input_dim} (x0, v0, ω, t)")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Output dim: {output_dim} (x, v)")
    print(f"  Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"  Função de ativação: {activation.capitalize()}")
    
    return model


def treina_mlp_node(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    parameters: Dict[str, Any]
) -> Tuple[nn.Module, Dict]:
    """Treina o modelo MLP."""

    mlp_config = parameters.get('mlp', {})
    
    batch_size = mlp_config.get('batch_size', 512)
    epochs = mlp_config.get('epochs', 500)
    learning_rate = mlp_config.get('learning_rate', 0.005)
    # weight_decay = mlp_config.get('weight_decay', 0.00005)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    model = model.to(device)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    history = {
        'train_loss': [],
        'test_loss': []
    }
    
    print("\n=== INICIANDO TREINAMENTO DO MLP ===")
    print(f"  Entrada: (x0, v0, ω, t) -> Saída: (x, v)")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    
    for epoch in range(epochs):
        # treino
        model.train()
        epoch_train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= len(train_loader)
        
        # teste
        model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                epoch_test_loss += loss.item()
        
        epoch_test_loss /= len(test_loader)
        
        history['train_loss'].append(float(epoch_train_loss))
        history['test_loss'].append(float(epoch_test_loss))
        
        scheduler.step(epoch_test_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Train Loss: {epoch_train_loss:.6f} | Test Loss: {epoch_test_loss:.6f}")
        
    return model, history


def avalia_mlp_node(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler_y: StandardScaler
) -> Dict[str, float]:
    """Avalia o modelo MLP nos dados de teste."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions_scaled_test = model(X_test_tensor).cpu().numpy()
        predictions_scaled_val = model(X_val_tensor).cpu().numpy()
    
    # desnormaliza previsões
    predictions_test = scaler_y.inverse_transform(predictions_scaled_test)
    y_test_original = scaler_y.inverse_transform(y_test)
    predictions_val = scaler_y.inverse_transform(predictions_scaled_val)
    y_val_original = scaler_y.inverse_transform(y_val)

    rmse_pos_test = float(np.sqrt(mean_squared_error(y_test_original[:, 0], predictions_test[:, 0])))
    rmse_vel_test = float(np.sqrt(mean_squared_error(y_test_original[:, 1], predictions_test[:, 1])))
    r2_pos_test = float(r2_score(y_test_original[:, 0], predictions_test[:, 0]))
    r2_vel_test = float(r2_score(y_test_original[:, 1], predictions_test[:, 1]))
    
    rmse_pos_val = float(np.sqrt(mean_squared_error(y_val_original[:, 0], predictions_val[:, 0])))
    rmse_vel_val = float(np.sqrt(mean_squared_error(y_val_original[:, 1], predictions_val[:, 1])))
    r2_pos_val = float(r2_score(y_val_original[:, 0], predictions_val[:, 0]))
    r2_vel_val = float(r2_score(y_val_original[:, 1], predictions_val[:, 1]))

    metrics = {
        'rmse_posicao_test': rmse_pos_test,
        'rmse_velocidade_test': rmse_vel_test,
        'r2_posicao_test': r2_pos_test,
        'r2_velocidade_test': r2_vel_test,
        'rmse_posicao_val': rmse_pos_val,
        'rmse_velocidade_val': rmse_vel_val,
        'r2_posicao_val': r2_pos_val,
        'r2_velocidade_val': r2_vel_val,
    }
    
    print("\n=== AVALIAÇÃO DO MODELO MLP ===")
    print(f"  RMSE Posição Teste: {rmse_pos_test:.6f}")
    print(f"  RMSE Velocidade Teste: {rmse_vel_test:.6f}")
    print(f"  R² Posição Teste: {r2_pos_test:.4f}")
    print(f"  R² Velocidade Teste: {r2_vel_test:.4f}")
    print(f"  RMSE Posição Validação: {rmse_pos_val:.6f}")
    print(f"  RMSE Velocidade Validação: {rmse_vel_val:.6f}")
    print(f"  R² Posição Validação: {r2_pos_val:.4f}")
    print(f"  R² Velocidade Validação: {r2_vel_val:.4f}")

    return metrics


def visualiza_previsoes_mlp_node(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any]
) -> None:
    """
    Visualiza as previsões do modelo MLP nos dados de validação.
    
    Args:
        model: Modelo treinado
        X_val: Dados de validação
        y_val: Targets de validação
        scaler_y: Scaler dos targets
        parameters: Parâmetros do pipeline
    """
    data_version = parameters.get('data_version', 'base_01')
    
    output_dir = f"data/08_reporting/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_previsoes_mlp = f"{output_dir}/previsoes_mlp.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions_scaled = model(X_val_tensor).cpu().numpy()
    
    # desnormaliza previsões
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_val_original = scaler_y.inverse_transform(y_val)
    
    fig = cria_grafico_previsoes_mlp(
        predictions=predictions,
        y_true=y_val_original,
        titulo="Previsão de Posição e Velocidade - MLP nos Dados de Validação"
    )
    
    fig.write_html(grafico_previsoes_mlp)
    print(f"Gráfico de previsões salvo em {grafico_previsoes_mlp}")
    
    fig.show()
    
    return None


def visualiza_previsoes_espaco_fases_node(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any]
) -> None:
    """
    Node: Visualiza as previsões do modelo no espaço de fases (Posição vs Velocidade).
    
    Args:
        model: Modelo treinado
        X_val: Dados de validação
        y_val: Targets de validação (posição e velocidade normalizadas)
        scaler_y: Scaler dos targets
        parameters: Parâmetros do pipeline
    """
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'default_v1')
    
    output_dir = f"data/08_reporting/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_previsoes_espaco_fases = f"{output_dir}/previsoes_espaco_fases.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions_scaled = model(X_val_tensor).cpu().numpy()
    
    # desnormaliza previsões
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_val_original = scaler_y.inverse_transform(y_val)
    
    y_pos_true = y_val_original[:, 0].reshape(-1, 1)
    y_vel_true = y_val_original[:, 1].reshape(-1, 1)
    y_pos_pred = predictions[:, 0].reshape(-1, 1)
    y_vel_pred = predictions[:, 1].reshape(-1, 1)
    
    fig = cria_grafico_previsoes_espaco_fases(
        y_pos_true=y_pos_true,
        y_vel_true=y_vel_true,
        y_pos_pred=y_pos_pred,
        y_vel_pred=y_vel_pred,
        titulo="Previsões do Modelo no Espaço de Fases - Dados de Validação"
    )
    
    fig.write_html(grafico_previsoes_espaco_fases)
    print(f"Gráfico de previsões no espaço de fases salvo em {grafico_previsoes_espaco_fases}")
    
    fig.show()
    
    return None


def interpola_trajetorias_node(
    model: nn.Module,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any]
) -> None:
    """
    Node: Usa o modelo treinado para fazer interpolações e prever trajetórias completas
    para novas condições iniciais e frequências não vistas durante o treinamento.
    Este nó simula o modelo em produção.
    
    Args:
        model: Modelo MLP treinado
        scaler_X: Scaler das features de entrada
        scaler_y: Scaler dos targets
        parameters: Parâmetros do pipeline
    """

    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_interpolacao_completa = f"{output_dir}/interpolacao_completa.html"
    grafico_interpolacao_espaco_fases = f"{output_dir}/interpolacao_espaco_fases.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    sim_params = parameters.get('simulation', {})
    dt = sim_params.get('dt', 0.01)
    
    print("\n=== INICIANDO MODELO PARA INTERPOLAÇÃO ===")
    
    # casos de teste para interpolação
    casos_teste = [
        {
            "nome": "Caso 1: ω=1.0 rad/s",
            "x0": 0.3,
            "v0": 0.0,
            "omega": 1.0,
            "t_final": 2 * np.pi / 1.0,
            "cor": CORES_PALETA[0]
        },
        {
            "nome": "Caso 2: ω=2.0 rad/s",
            "x0": 0.5,
            "v0": 0.0,
            "omega": 2.0,
            "t_final": 2 * np.pi / 2.0,
            "cor": CORES_PALETA[2]
        },
        {
            "nome": "Caso 3: ω=3.0 rad/s",
            "x0": 0.0,
            "v0": 0.8,
            "omega": 3.0,
            "t_final": 2 * np.pi / 3.0,
            "cor": CORES_PALETA[1]
        },
        {
            "nome": "Caso 4: ω=4.0 rad/s",
            "x0": -0.2,
            "v0": -0.5,
            "omega": 4.0,
            "t_final": 2 * np.pi / 4.0,
            "cor": CORES_PALETA[4]
        },
        {
            "nome": "Caso 5: ω=5.0 rad/s",
            "x0": 0.1,
            "v0": 0.2,
            "omega": 5.0,
            "t_final": 2 * np.pi / 5.0,
            "cor": CORES_PALETA[5]
        },
        {
            "nome": "Caso 6: ω=6.0 rad/s (extrapolação)",
            "x0": -0.4,
            "v0": -0.8,
            "omega": 6.0,
            "t_final": 2 * np.pi / 6.0,
            "cor": CORES_PALETA[14]
        }
    ]
    
    # gera os nomes das legendas dinamicamente
    for caso in casos_teste:
        caso["nome_legenda"] = (
            f"{caso['nome']}: x0={caso['x0']:.1f} m, "
            f"v0={caso['v0']:.1f} m/s, "
            f"ω={caso['omega']:.1f} rad/s, "
            f"T={caso['t_final']:.2f} s"
        )
    
    tempos_lista = []
    posicoes_lista = []
    velocidades_lista = []
    
    print("  Processando casos de teste...")
    for caso in casos_teste:
        t_max = caso["t_final"]
        tempos = np.arange(0, t_max + dt, dt)
        
        # prepara as entradas
        X_caso = np.zeros((len(tempos), 4))
        X_caso[:, 0] = caso["x0"]
        X_caso[:, 1] = caso["v0"]
        X_caso[:, 2] = caso["omega"]
        X_caso[:, 3] = tempos
        
        # normaliza e faz previsão
        X_caso_scaled = scaler_X.transform(X_caso)
        X_tensor = torch.tensor(X_caso_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            predictions_scaled = model(X_tensor).cpu().numpy()
        
        predictions = scaler_y.inverse_transform(predictions_scaled)
        
        tempos_lista.append(tempos)
        posicoes_lista.append(predictions[:, 0])
        velocidades_lista.append(predictions[:, 1])
    
    print("  Gerando gráficos de interpolação...")
    
    fig_completo = cria_grafico_interpolacao_completo(
        tempos_lista=tempos_lista,
        posicoes_lista=posicoes_lista,
        velocidades_lista=velocidades_lista,
        casos_info=casos_teste,
        titulo="Modelo em Produção: Posição e Velocidade vs Tempo"
    )
    
    fig_fases = cria_grafico_interpolacao_espaco_fases(
        posicoes_lista=posicoes_lista,
        velocidades_lista=velocidades_lista,
        casos_info=casos_teste,
        titulo="Modelo em Produção: Espaço de Fases"
    )
    
    if fig_completo is not None:
        fig_completo.write_html(grafico_interpolacao_completa)
        print(f"Gráfico de interpolação (Posição e Velocidade) salvo em {grafico_interpolacao_completa}")
    else:
        print("ERRO: fig_completo é None")
    
    if fig_fases is not None:
        fig_fases.write_html(grafico_interpolacao_espaco_fases)
        print(f"Gráfico de interpolação (Espaço de Fases) salvo em {grafico_interpolacao_espaco_fases}")
    else:
        print("ERRO: fig_fases é None")
    
    print("\n=== MODELO EM PRODUÇÃO ===")
    print(f"  Passo de tempo (dt): {dt} s")
    print(f"  Número de casos testados: {len(casos_teste)}")
    print(f"  Período mais longo: {2 * np.pi / 1.0:.2f} s")
    print(f"  Período mais curto: {2 * np.pi / 6.0:.2f} s")
    
    fig_completo.show()
    fig_fases.show()
    
    return None