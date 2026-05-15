# nodes saída x,v; entrada [x0, v0, ω, t]

"""
Nodes do pipeline MLP.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any, Tuple

from .model import MLP
from oscilador_harmonico.utils import cria_grafico_previsoes_mlp


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


def visualiza_distribuicao_dados_separado(base_oscilador: pd.DataFrame, parameters: Dict[str, Any]) -> None:
    """
    Node separado para visualizar a distribuição dos dados no espaço de fases.
    Carrega os dados novamente e faz a divisão apenas para visualização.
    Não interfere no pipeline principal de treinamento.
    
    Args:
        base_oscilador: DataFrame com a base consolidada
        parameters: Parâmetros do pipeline
    """
    from oscilador_harmonico.utils import cria_grafico_distribuicao_dados
    
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
    
    fig.write_html("data/08_reporting/distribuicao_dados.html")
    print("Gráfico de distribuição salvo em data/08_reporting/distribuicao_dados.html")
    
    fig.show()
    
    return None

def cria_modelo_mlp_node(input_dim: int, output_dim: int, parameters: Dict[str, Any]) -> nn.Module:
    """Cria o modelo MLP."""
    mlp_config = parameters.get('mlp', {})
    
    hidden_dims = mlp_config.get('hidden_dims', [64, 128, 64])
    activation = mlp_config.get('activation', 'sigmoid')
    
    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=activation
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
    
    fig.write_html("data/08_reporting/previsoes_mlp.html")
    print("Gráfico de previsões salvo em data/08_reporting/previsoes_mlp.html")
    
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
    from oscilador_harmonico.utils import cria_grafico_previsoes_espaco_fases
    
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
    
    fig.write_html("data/08_reporting/previsoes_espaco_fases.html")
    print("Gráfico de previsões no espaço de fases salvo em data/08_reporting/previsoes_espaco_fases.html")
    
    fig.show()
    
    return None