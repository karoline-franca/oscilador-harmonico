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
    cria_grafico_real_previsto_mlp,
    cria_grafico_distribuicao_dados,
    cria_grafico_previsoes_espaco_fases,
    cria_grafico_interpolacao_pontual_mlp,
    cria_grafico_interpolacao_pontual_espaco_fases,
    cria_grafico_interpolacao_pontual_completo,
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
    
    # 70% treino, 20% validação, 10% teste
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_scaled, test_size=0.30, random_state=42  # 30% temporário
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.3333, random_state=42  # 10% do total (33.33% dos 30%)
    )
    
    print(f"  Treino: {X_train.shape}, Validação: {X_val.shape}, Teste: {X_test.shape}")
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, input_dim, output_dim, scaler_X, scaler_y


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
    
    # 70% treino, 20% validação, 10% teste
    _, y_temp, _, _ = train_test_split(
        y_combined, y_combined, test_size=0.30, random_state=42  # 30% temporário
    )
    y_val, y_test, _, _ = train_test_split(
        y_temp, y_temp, test_size=0.3333, random_state=42  # 10% do total (33.33% dos 30%)
    )
    y_train = y_combined[:len(y_combined) - len(y_temp)]
    
    y_pos_train = y_train[:, 0].reshape(-1, 1)
    y_vel_train = y_train[:, 1].reshape(-1, 1)
    y_pos_val = y_val[:, 0].reshape(-1, 1)
    y_vel_val = y_val[:, 1].reshape(-1, 1)
    y_pos_test = y_test[:, 0].reshape(-1, 1)
    y_vel_test = y_test[:, 1].reshape(-1, 1)
    
    fig = cria_grafico_distribuicao_dados(
        y_pos_train=y_pos_train,
        y_vel_train=y_vel_train,
        y_pos_val=y_pos_val,
        y_vel_val=y_vel_val,
        y_pos_test=y_pos_test,
        y_vel_test=y_vel_test,
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
    X_val: np.ndarray,
    y_val: np.ndarray,
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
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    print("\n=== INICIANDO TREINAMENTO DO MLP ===")
    print(f"  Entrada: (x0, v0, ω, t) -> Saída: (x, v)")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Função loss: MSE (Mean Squared Error)")
    
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
        
        # validação
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                epoch_val_loss += loss.item()
        
        epoch_val_loss /= len(val_loader)
        
        history['train_loss'].append(float(epoch_train_loss))
        history['val_loss'].append(float(epoch_val_loss))
        
        scheduler.step(epoch_val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
        
    return model, history


def avalia_mlp_node(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler
) -> Dict[str, float]:
    """Avalia o modelo MLP nos dados de validação."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions_scaled_val = model(X_val_tensor).cpu().numpy()
        predictions_scaled_test = model(X_test_tensor).cpu().numpy()
    
    # desnormaliza previsões
    predictions_val = scaler_y.inverse_transform(predictions_scaled_val)
    y_val_original = scaler_y.inverse_transform(y_val)
    predictions_test = scaler_y.inverse_transform(predictions_scaled_test)
    y_test_original = scaler_y.inverse_transform(y_test)

    rmse_pos_val = float(np.sqrt(mean_squared_error(y_val_original[:, 0], predictions_val[:, 0])))
    rmse_vel_val = float(np.sqrt(mean_squared_error(y_val_original[:, 1], predictions_val[:, 1])))
    r2_pos_val = float(r2_score(y_val_original[:, 0], predictions_val[:, 0]))
    r2_vel_val = float(r2_score(y_val_original[:, 1], predictions_val[:, 1]))
    
    rmse_pos_test = float(np.sqrt(mean_squared_error(y_test_original[:, 0], predictions_test[:, 0])))
    rmse_vel_test = float(np.sqrt(mean_squared_error(y_test_original[:, 1], predictions_test[:, 1])))
    r2_pos_test = float(r2_score(y_test_original[:, 0], predictions_test[:, 0]))
    r2_vel_test = float(r2_score(y_test_original[:, 1], predictions_test[:, 1]))

    metrics = {
        'rmse_posicao_val': rmse_pos_val,
        'rmse_velocidade_val': rmse_vel_val,
        'r2_posicao_val': r2_pos_val,
        'r2_velocidade_val': r2_vel_val,
        'rmse_posicao_test': rmse_pos_test,
        'rmse_velocidade_test': rmse_vel_test,
        'r2_posicao_test': r2_pos_test,
        'r2_velocidade_test': r2_vel_test,
    }
    
    print("\n=== AVALIAÇÃO DO MODELO MLP ===")
    print(f"  RMSE Posição Validação: {rmse_pos_val:.6f}")
    print(f"  RMSE Velocidade Validação: {rmse_vel_val:.6f}")
    print(f"  R² Posição Validação: {r2_pos_val:.4f}")
    print(f"  R² Velocidade Validação: {r2_vel_val:.4f}")
    print(f"  RMSE Posição Teste: {rmse_pos_test:.6f}")
    print(f"  RMSE Velocidade Teste: {rmse_vel_test:.6f}")
    print(f"  R² Posição Teste: {r2_pos_test:.4f}")
    print(f"  R² Velocidade Teste: {r2_vel_test:.4f}")

    return metrics


def visualiza_previsoes_mlp_node(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any]
) -> None:
    """
    Visualiza as previsões do modelo MLP nos dados de teste.
    
    Args:
        model: Modelo treinado
        X_test: Dados de teste
        y_test: Targets de teste
        scaler_y: Scaler dos targets
        parameters: Parâmetros do pipeline
    """
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_previsoes_mlp = f"{output_dir}/real_previsto_mlp.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).cpu().numpy()
    
    # desnormaliza previsões
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    fig = cria_grafico_real_previsto_mlp(
        predictions=predictions,
        y_true=y_test_original,
        titulo="Real vs Previsto - Dados de Teste"
    )
    
    fig.write_html(grafico_previsoes_mlp)
    print(f"Gráfico de previsões salvo em {grafico_previsoes_mlp}")
    
    fig.show()
    
    return None


def visualiza_previsoes_espaco_fases_node(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any]
) -> None:
    """
    Node: Visualiza as previsões do modelo no espaço de fases (Posição vs Velocidade).
    
    Args:
        model: Modelo treinado
        X_test: Dados de teste
        y_test: Targets de teste (posição e velocidade normalizadas)
        scaler_y: Scaler dos targets
        parameters: Parâmetros do pipeline
    """
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'default_v1')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_previsoes_espaco_fases = f"{output_dir}/previsoes_espaco_fases.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print("\n=== VISUALIZAÇÃO DAS PREVISÕES NO ESPAÇO DE FASES ===")
    print(f"  Amostras de teste: {len(X_test)}")
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).cpu().numpy()
    
    # desnormaliza previsões
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    y_pos_true = y_test_original[:, 0].reshape(-1, 1)
    y_vel_true = y_test_original[:, 1].reshape(-1, 1)
    y_pos_pred = predictions[:, 0].reshape(-1, 1)
    y_vel_pred = predictions[:, 1].reshape(-1, 1)
    
    # calcula métricas para exibição
    rmse_pos = np.sqrt(mean_squared_error(y_pos_true, y_pos_pred))
    rmse_vel = np.sqrt(mean_squared_error(y_vel_true, y_vel_pred))
    r2_pos = r2_score(y_pos_true, y_pos_pred)
    r2_vel = r2_score(y_vel_true, y_vel_pred)
    
    print(f"  RMSE Posição: {rmse_pos:.6f} m")
    print(f"  RMSE Velocidade: {rmse_vel:.6f} m/s")
    print(f"  R² Posição: {r2_pos:.4f}")
    print(f"  R² Velocidade: {r2_vel:.4f}")
    
    fig = cria_grafico_previsoes_espaco_fases(
        y_pos_true=y_pos_true,
        y_vel_true=y_vel_true,
        y_pos_pred=y_pos_pred,
        y_vel_pred=y_vel_pred,
        titulo="Previsões do Modelo no Espaço de Fases - Dados de Teste"
    )
    
    fig.write_html(grafico_previsoes_espaco_fases)
    print(f"\n  Gráfico de previsões no espaço de fases salvo em {grafico_previsoes_espaco_fases}")
    
    fig.show()
    
    return None


def interpolacoes_pontuais_tempo_mlp_node(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Node: Usa o modelo treinado para fazer interpolação entre pontos dos dados de teste.
    Faz previsões pontuais independentes em pontos que não estão no conjunto original de treino/validação.
    
    Args:
        model: Modelo MLP treinado
        X_test: Dados de teste (entradas) - usado como referência
        y_test: Dados de teste (saídas reais) - usado como referência
        scaler_X: Scaler das features de entrada
        scaler_y: Scaler dos targets
        parameters: Parâmetros do pipeline
        
    Returns:
        DataFrame com os dados interpolados temporalmente e previsões do modelo
    """
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_interpolacao_pontual = f"{output_dir}/interpolacoes_pontuais_real_previsto_mlp.html"
    grafico_interpolacao_pontual_espaco_fases = f"{output_dir}/interpolacao_pontual_espaco_fases.html"
    grafico_interpolacao_pontual_temporal = f"{output_dir}/interpolacao_pontual_v_x_vs_t.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print("\n=== INTERPOLAÇÃO PONTUAL ENTRE DADOS DE TESTE ===")
    
    # agrupa por condições iniciais e frequência (x0, v0, omega)
    # a interpolação é feita variando o tempo para um mesmo sistema
    
    X_test_df = pd.DataFrame(X_test, columns=['x0', 'v0', 'frequencia_angular', 'tempo'])
    
    # x0, v0, omega constantes para definir sistemas únicos e tempo variável para interpolação
    sistemas_unicos = X_test_df.groupby(['x0', 'v0', 'frequencia_angular']).size().reset_index()
    sistemas_unicos = sistemas_unicos.head(3)
    
    print(f"\n  Sistemas únicos encontrados: {len(sistemas_unicos)}")
    print("  Interpolando novos pontos de tempo para cada sistema...")
        
    todas_previsoes = []
    todos_reais_interpolados = []
    todas_condicoes = []
    
    # listas para o gráfico temporal
    tempos_lista = []
    posicoes_previstas_lista = []
    velocidades_previstas_lista = []
    posicoes_reais_lista = []
    velocidades_reais_lista = []
    casos_info_lista = []
    
    # Lista para armazenar todos os dados interpolados
    dados_interpolados = []
    
    for idx, row in sistemas_unicos.iterrows():
        x0 = row['x0']
        v0 = row['v0']
        omega = row['frequencia_angular']
        
        # filtra os pontos originais deste sistema
        mask = (np.abs(X_test[:, 0] - x0) < 1e-6) & \
               (np.abs(X_test[:, 1] - v0) < 1e-6) & \
               (np.abs(X_test[:, 2] - omega) < 1e-6)
        
        tempos_originais = X_test[mask, 3]
        pos_originais = y_test[mask, 0]
        vel_originais = y_test[mask, 1]
        
        if len(tempos_originais) < 5:
            continue
        
        # ordena por tempo
        idx_sort = np.argsort(tempos_originais)
        tempos_originais = tempos_originais[idx_sort]
        pos_originais = pos_originais[idx_sort]
        vel_originais = vel_originais[idx_sort]
        
        # cria tempos interpolados (pontos entre os tempos originais)
        t_min = tempos_originais.min()
        t_max = tempos_originais.max()
        
        # gera 100 pontos interpolados uniformemente
        tempos_interpolados = np.linspace(t_min, t_max, 100)
        
        # para validação, obtém os valores reais nesses tempos (via solução analítica do OHS)
        # solução analítica: x(t) = x0*cos(ωt) + (v0/ω)*sin(ωt)
        #                 v(t) = -x0*ω*sin(ωt) + v0*cos(ωt)
        pos_reais_interpolados = x0 * np.cos(omega * tempos_interpolados) + (v0 / omega) * np.sin(omega * tempos_interpolados)
        vel_reais_interpolados = -x0 * omega * np.sin(omega * tempos_interpolados) + v0 * np.cos(omega * tempos_interpolados)
        
        # prepara entrada para o modelo
        X_interpolado = np.zeros((len(tempos_interpolados), 4))
        X_interpolado[:, 0] = x0
        X_interpolado[:, 1] = v0
        X_interpolado[:, 2] = omega
        X_interpolado[:, 3] = tempos_interpolados
        
        # normaliza e faz previsão
        X_interpolado_scaled = scaler_X.transform(X_interpolado)
        X_tensor = torch.tensor(X_interpolado_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()
        
        pred = scaler_y.inverse_transform(pred_scaled)
        
        # armazena para métricas globais
        todas_previsoes.append(pred)
        todos_reais_interpolados.append(np.column_stack([pos_reais_interpolados, vel_reais_interpolados]))
        
        # armazena informações para gráfico temporal
        tempos_lista.append(tempos_interpolados)
        posicoes_previstas_lista.append(pred[:, 0])
        velocidades_previstas_lista.append(pred[:, 1])
        posicoes_reais_lista.append(pos_reais_interpolados)
        velocidades_reais_lista.append(vel_reais_interpolados)
        
        # escolhe uma cor da paleta baseada no índice
        cor = CORES_PALETA[idx % len(CORES_PALETA)]
        
        casos_info_lista.append({
            'x0': x0,
            'v0': v0,
            'omega': omega,
            'cor': cor
        })
        
        # armazena informações para gráficos adicionais
        todas_condicoes.append({
            'x0': x0,
            'v0': v0,
            'omega': omega,
            'tempos_originais': tempos_originais,
            'pos_originais': pos_originais,
            'vel_originais': vel_originais,
            'tempos_interpolados': tempos_interpolados,
            'pos_previstas': pred[:, 0],
            'vel_previstas': pred[:, 1],
            'pos_reais_interpolados': pos_reais_interpolados,
            'vel_reais_interpolados': vel_reais_interpolados
        })
        
        # Cria registros para o DataFrame interpolado temporalmente
        for k in range(len(tempos_interpolados)):
            # Encontra o tempo original mais próximo para referência
            idx_tempo_original = np.argmin(np.abs(tempos_originais - tempos_interpolados[k]))
            tempo_original_mais_proximo = tempos_originais[idx_tempo_original]
            pos_original_mais_proximo = pos_originais[idx_tempo_original]
            vel_original_mais_proximo = vel_originais[idx_tempo_original]
            
            dados_interpolados.append({
                'x0': x0,
                'v0': v0,
                'omega': omega,
                'tempo_original_mais_proximo': tempo_original_mais_proximo,
                'posicao_original_mais_proxima': pos_original_mais_proximo,
                'velocidade_original_mais_proxima': vel_original_mais_proximo,
                'tempo_interpolado': tempos_interpolados[k],
                'posicao_analitica': pos_reais_interpolados[k],
                'velocidade_analitica': vel_reais_interpolados[k],
                'posicao_prevista_mlp': pred[k, 0],
                'velocidade_prevista_mlp': pred[k, 1],
                'erro_posicao': pred[k, 0] - pos_reais_interpolados[k],
                'erro_velocidade': pred[k, 1] - vel_reais_interpolados[k],
                'erro_abs_posicao': abs(pred[k, 0] - pos_reais_interpolados[k]),
                'erro_abs_velocidade': abs(pred[k, 1] - vel_reais_interpolados[k]),
                'erro_rel_posicao_pct': (abs(pred[k, 0] - pos_reais_interpolados[k]) / (abs(pos_reais_interpolados[k]) + 1e-6)) * 100,
                'erro_rel_velocidade_pct': (abs(pred[k, 1] - vel_reais_interpolados[k]) / (abs(vel_reais_interpolados[k]) + 1e-6)) * 100,
                'delta_tempo': tempos_interpolados[k] - tempo_original_mais_proximo,
                't_min_sistema': t_min,
                't_max_sistema': t_max,
                'posicao_normalizada_tempo': (tempos_interpolados[k] - t_min) / (t_max - t_min)
            })
    
    if len(todas_previsoes) == 0:
        print("  ERRO: Nenhum sistema válido encontrado para interpolação")
        return pd.DataFrame()
        
    predictions_all = np.vstack(todas_previsoes)
    y_true_all = np.vstack(todos_reais_interpolados)
    
    rmse_pos = float(np.sqrt(mean_squared_error(y_true_all[:, 0], predictions_all[:, 0])))
    rmse_vel = float(np.sqrt(mean_squared_error(y_true_all[:, 1], predictions_all[:, 1])))
    r2_pos = float(r2_score(y_true_all[:, 0], predictions_all[:, 0]))
    r2_vel = float(r2_score(y_true_all[:, 1], predictions_all[:, 1]))
    
    print(f"\n  Total de pontos interpolados: {len(predictions_all)}")
    print(f"  RMSE Posição (vs solução analítica): {rmse_pos:.6f} m")
    print(f"  RMSE Velocidade (vs solução analítica): {rmse_vel:.6f} m/s")
    print(f"  R² Posição (vs solução analítica): {r2_pos:.4f}")
    print(f"  R² Velocidade (vs solução analítica): {r2_vel:.4f}")
    
    # ============================================
    # GRÁFICO 1: Real vs Previsto
    # ============================================
    
    fig1 = cria_grafico_interpolacao_pontual_mlp(
        predictions=predictions_all,
        y_true=y_true_all,
        titulo="Interpolação Pontual: MLP vs Solução Analítica - Dados de Teste"
    )
    
    fig1.write_html(grafico_interpolacao_pontual)
    print(f"\n  Gráfico de interpolação pontual (Real vs Previsto) salvo em {grafico_interpolacao_pontual}")
    
    # ============================================
    # GRÁFICO 2: Espaço de Fases
    # ============================================
    
    y_pos_true = y_true_all[:, 0].reshape(-1, 1)
    y_vel_true = y_true_all[:, 1].reshape(-1, 1)
    y_pos_pred = predictions_all[:, 0].reshape(-1, 1)
    y_vel_pred = predictions_all[:, 1].reshape(-1, 1)
    
    fig2 = cria_grafico_interpolacao_pontual_espaco_fases(
        y_pos_true=y_pos_true,
        y_vel_true=y_vel_true,
        y_pos_pred=y_pos_pred,
        y_vel_pred=y_vel_pred,
        titulo="Interpolação Pontual do Modelo no Espaço de Fases - MLP vs Solução Analítica"
    )
    
    fig2.write_html(grafico_interpolacao_pontual_espaco_fases)
    print(f"  Gráfico de interpolação pontual (Espaço de Fases) salvo em {grafico_interpolacao_pontual_espaco_fases}")
    
    # ============================================
    # GRÁFICO 3: Posição e Velocidade vs Tempo
    # ============================================
    
    fig3 = cria_grafico_interpolacao_pontual_completo(
        tempos_lista=tempos_lista,
        posicoes_previstas_lista=posicoes_previstas_lista,
        velocidades_previstas_lista=velocidades_previstas_lista,
        posicoes_reais_lista=posicoes_reais_lista,
        velocidades_reais_lista=velocidades_reais_lista,
        casos_info=casos_info_lista,
        titulo="Interpolação Pontual: MLP vs Solução Analítica - Posição e Velocidade vs Tempo"
    )
    
    fig3.write_html(grafico_interpolacao_pontual_temporal)
    print(f"  Gráfico de interpolação pontual (Posição/Velocidade vs Tempo) salvo em {grafico_interpolacao_pontual_temporal}")
    
    fig1.show()
    fig2.show()
    fig3.show()
    
    # Cria DataFrame com os dados interpolados temporalmente
    df_interpolado_tempo = pd.DataFrame(dados_interpolados)
    
    # Adiciona informações de metadados
    df_interpolado_tempo.attrs['rmse_posicao'] = rmse_pos
    df_interpolado_tempo.attrs['rmse_velocidade'] = rmse_vel
    df_interpolado_tempo.attrs['r2_posicao'] = r2_pos
    df_interpolado_tempo.attrs['r2_velocidade'] = r2_vel
    df_interpolado_tempo.attrs['total_pontos'] = len(predictions_all)
    df_interpolado_tempo.attrs['num_sistemas'] = len(sistemas_unicos)
    df_interpolado_tempo.attrs['pontos_por_sistema'] = 100
    
    print(f"\n  Base de dados com interpolação temporal gerada com {len(df_interpolado_tempo)} registros")
    print(f"  Colunas disponíveis: {list(df_interpolado_tempo.columns)}")
    
    return df_interpolado_tempo


def interpolacoes_pontuais_x0_v0_w_mlp_node(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Node: Usa o modelo treinado para fazer interpolação entre pontos dos dados de teste.
    Faz previsões pontuais independentes em pontos que não estão no conjunto original de treino/validação.
    
    Args:
        model: Modelo MLP treinado
        X_test: Dados de teste (entradas) - usado como referência
        y_test: Dados de teste (saídas reais) - usado como referência
        scaler_X: Scaler das features de entrada
        scaler_y: Scaler dos targets
        parameters: Parâmetros do pipeline
        
    Returns:
        DataFrame com os dados interpolados e previsões do modelo
    """
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_interpolacao_pontual = f"{output_dir}/interpolacoes_pontuais_real_previsto_mlp.html"
    grafico_interpolacao_pontual_espaco_fases = f"{output_dir}/interpolacao_pontual_espaco_fases.html"
    grafico_interpolacao_pontual_temporal = f"{output_dir}/interpolacao_pontual_v_x_vs_t.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print("\n=== INTERPOLAÇÃO PONTUAL ENTRE DADOS DE TESTE ===")
    
    # agrupa por condições iniciais e frequência (x0, v0, omega)
    # a interpolação é feita variando o tempo para um mesmo sistema
    
    X_test_df = pd.DataFrame(X_test, columns=['x0', 'v0', 'frequencia_angular', 'tempo'])
    
    # x0, v0, omega constantes para definir sistemas únicos e tempo variável para interpolação
    sistemas_unicos = X_test_df.groupby(['x0', 'v0', 'frequencia_angular']).size().reset_index()
    sistemas_unicos = sistemas_unicos.head(3)
    
    print(f"\n  Sistemas únicos encontrados: {len(sistemas_unicos)}")
    print("  Interpolando novos pontos de tempo para cada sistema...")
        
    todas_previsoes = []
    todos_reais_interpolados = []
    todas_condicoes = []
    
    # listas para o gráfico temporal
    tempos_lista = []
    posicoes_previstas_lista = []
    velocidades_previstas_lista = []
    posicoes_reais_lista = []
    velocidades_reais_lista = []
    casos_info_lista = []
    
    # Lista para armazenar todos os dados interpolados
    dados_interpolados = []
    
    for idx, row in sistemas_unicos.iterrows():
        x0 = row['x0']
        v0 = row['v0']
        omega = row['frequencia_angular']
        
        # filtra os pontos originais deste sistema
        mask = (np.abs(X_test[:, 0] - x0) < 1e-6) & \
               (np.abs(X_test[:, 1] - v0) < 1e-6) & \
               (np.abs(X_test[:, 2] - omega) < 1e-6)
        
        tempos_originais = X_test[mask, 3]
        pos_originais = y_test[mask, 0]
        vel_originais = y_test[mask, 1]
        
        if len(tempos_originais) < 5:
            continue
        
        # ordena por tempo
        idx_sort = np.argsort(tempos_originais)
        tempos_originais = tempos_originais[idx_sort]
        pos_originais = pos_originais[idx_sort]
        vel_originais = vel_originais[idx_sort]
        
        # mantém os tempos originais, interpola entre condições iniciais
        # para diferentes sistemas
        
        # encontra sistemas vizinhos com valores diferentes de x0, v0, omega
        outros_sistemas = sistemas_unicos[~((sistemas_unicos['x0'] == x0) & 
                                            (sistemas_unicos['v0'] == v0) & 
                                            (sistemas_unicos['frequencia_angular'] == omega))]
        
        if len(outros_sistemas) == 0:
            continue
        
        # para cada tempo original, cria pontos interpolados entre as condições iniciais
        tempos_interpolados = tempos_originais.copy()
        
        # gera fatores de interpolação (entre 0 e 1)
        # interpola entre o sistema atual (alpha=1) e outro sistema (alpha=0)
        alphas = np.linspace(0, 1, 10)  # 10 pontos interpolados entre os sistemas
        
        for alpha in alphas:
            # escolhe um sistema vizinho aleatório para interpolar
            sistema_vizinho = outros_sistemas.iloc[np.random.randint(0, len(outros_sistemas))]
            
            x0_vizinho = sistema_vizinho['x0']
            v0_vizinho = sistema_vizinho['v0']
            omega_vizinho = sistema_vizinho['frequencia_angular']
            
            # interpola as condições iniciais
            x0_interp = (1 - alpha) * x0_vizinho + alpha * x0
            v0_interp = (1 - alpha) * v0_vizinho + alpha * v0
            omega_interp = (1 - alpha) * omega_vizinho + alpha * omega
            
            # calcula solução analítica para os parâmetros interpolados
            pos_reais_interpolados = x0_interp * np.cos(omega_interp * tempos_interpolados) + \
                                     (v0_interp / omega_interp) * np.sin(omega_interp * tempos_interpolados)
            vel_reais_interpolados = -x0_interp * omega_interp * np.sin(omega_interp * tempos_interpolados) + \
                                     v0_interp * np.cos(omega_interp * tempos_interpolados)
            
            # prepara entrada para o modelo
            X_interpolado = np.zeros((len(tempos_interpolados), 4))
            X_interpolado[:, 0] = x0_interp
            X_interpolado[:, 1] = v0_interp
            X_interpolado[:, 2] = omega_interp
            X_interpolado[:, 3] = tempos_interpolados
            
            # normaliza e faz previsão
            X_interpolado_scaled = scaler_X.transform(X_interpolado)
            X_tensor = torch.tensor(X_interpolado_scaled, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                pred_scaled = model(X_tensor).cpu().numpy()
            
            pred = scaler_y.inverse_transform(pred_scaled)
            
            # armazena para métricas globais
            todas_previsoes.append(pred)
            todos_reais_interpolados.append(np.column_stack([pos_reais_interpolados, vel_reais_interpolados]))
            
            # armazena informações para gráfico temporal
            tempos_lista.append(tempos_interpolados)
            posicoes_previstas_lista.append(pred[:, 0])
            velocidades_previstas_lista.append(pred[:, 1])
            posicoes_reais_lista.append(pos_reais_interpolados)
            velocidades_reais_lista.append(vel_reais_interpolados)
            
            # escolhe uma cor da paleta baseada no índice
            cor = CORES_PALETA[idx % len(CORES_PALETA)]
            
            casos_info_lista.append({
                'x0': x0_interp,
                'v0': v0_interp,
                'omega': omega_interp,
                'cor': cor
            })
            
            # armazena informações para gráficos adicionais
            todas_condicoes.append({
                'x0': x0_interp,
                'v0': v0_interp,
                'omega': omega_interp,
                'tempos_originais': tempos_originais,
                'pos_originais': pos_originais,
                'vel_originais': vel_originais,
                'tempos_interpolados': tempos_interpolados,
                'pos_previstas': pred[:, 0],
                'vel_previstas': pred[:, 1],
                'pos_reais_interpolados': pos_reais_interpolados,
                'vel_reais_interpolados': vel_reais_interpolados
            })
            
            # cria registros para o DataFrame interpolado
            for k in range(len(tempos_interpolados)):
                dados_interpolados.append({
                    'x0_original': x0,
                    'v0_original': v0,
                    'omega_original': omega,
                    'x0_vizinho': x0_vizinho,
                    'v0_vizinho': v0_vizinho,
                    'omega_vizinho': omega_vizinho,
                    'alpha_interpolacao': alpha,
                    'x0_interpolado': x0_interp,
                    'v0_interpolado': v0_interp,
                    'omega_interpolado': omega_interp,
                    'tempo': tempos_interpolados[k],
                    'posicao_analitica': pos_reais_interpolados[k],
                    'velocidade_analitica': vel_reais_interpolados[k],
                    'posicao_prevista_mlp': pred[k, 0],
                    'velocidade_prevista_mlp': pred[k, 1],
                    'erro_posicao': pred[k, 0] - pos_reais_interpolados[k],
                    'erro_velocidade': pred[k, 1] - vel_reais_interpolados[k],
                    'erro_abs_posicao': abs(pred[k, 0] - pos_reais_interpolados[k]),
                    'erro_abs_velocidade': abs(pred[k, 1] - vel_reais_interpolados[k])
                })
    
    if len(todas_previsoes) == 0:
        print("  ERRO: Nenhum sistema válido encontrado para interpolação")
        return pd.DataFrame()
        
    predictions_all = np.vstack(todas_previsoes)
    y_true_all = np.vstack(todos_reais_interpolados)
    
    rmse_pos = float(np.sqrt(mean_squared_error(y_true_all[:, 0], predictions_all[:, 0])))
    rmse_vel = float(np.sqrt(mean_squared_error(y_true_all[:, 1], predictions_all[:, 1])))
    r2_pos = float(r2_score(y_true_all[:, 0], predictions_all[:, 0]))
    r2_vel = float(r2_score(y_true_all[:, 1], predictions_all[:, 1]))
    
    print(f"\n  Total de pontos interpolados: {len(predictions_all)}")
    print(f"  RMSE Posição (vs solução analítica): {rmse_pos:.6f} m")
    print(f"  RMSE Velocidade (vs solução analítica): {rmse_vel:.6f} m/s")
    print(f"  R² Posição (vs solução analítica): {r2_pos:.4f}")
    print(f"  R² Velocidade (vs solução analítica): {r2_vel:.4f}")
    
    # ============================================
    # GRÁFICO 1: Real vs Previsto
    # ============================================
    
    fig1 = cria_grafico_interpolacao_pontual_mlp(
        predictions=predictions_all,
        y_true=y_true_all,
        titulo="Interpolação Pontual: MLP vs Solução Analítica - Dados de Teste"
    )
    
    fig1.write_html(grafico_interpolacao_pontual)
    print(f"\n  Gráfico de interpolação pontual (Real vs Previsto) salvo em {grafico_interpolacao_pontual}")
    
    # ============================================
    # GRÁFICO 2: Espaço de Fases
    # ============================================
    
    y_pos_true = y_true_all[:, 0].reshape(-1, 1)
    y_vel_true = y_true_all[:, 1].reshape(-1, 1)
    y_pos_pred = predictions_all[:, 0].reshape(-1, 1)
    y_vel_pred = predictions_all[:, 1].reshape(-1, 1)
    
    fig2 = cria_grafico_interpolacao_pontual_espaco_fases(
        y_pos_true=y_pos_true,
        y_vel_true=y_vel_true,
        y_pos_pred=y_pos_pred,
        y_vel_pred=y_vel_pred,
        titulo="Interpolação Pontual do Modelo no Espaço de Fases - MLP vs Solução Analítica"
    )
    
    fig2.write_html(grafico_interpolacao_pontual_espaco_fases)
    print(f"  Gráfico de interpolação pontual (Espaço de Fases) salvo em {grafico_interpolacao_pontual_espaco_fases}")
    
    # ============================================
    # GRÁFICO 3: Posição e Velocidade vs Tempo
    # ============================================
    
    fig3 = cria_grafico_interpolacao_pontual_completo(
        tempos_lista=tempos_lista,
        posicoes_previstas_lista=posicoes_previstas_lista,
        velocidades_previstas_lista=velocidades_previstas_lista,
        posicoes_reais_lista=posicoes_reais_lista,
        velocidades_reais_lista=velocidades_reais_lista,
        casos_info=casos_info_lista,
        titulo="Interpolação Pontual: MLP vs Solução Analítica - Posição e Velocidade vs Tempo"
    )
    
    fig3.write_html(grafico_interpolacao_pontual_temporal)
    print(f"  Gráfico de interpolação pontual (Posição/Velocidade vs Tempo) salvo em {grafico_interpolacao_pontual_temporal}")
    
    fig1.show()
    fig2.show()
    fig3.show()
    
    df_interpolado = pd.DataFrame(dados_interpolados)
    
    # adiciona informações de metadados
    df_interpolado.attrs['rmse_posicao'] = rmse_pos
    df_interpolado.attrs['rmse_velocidade'] = rmse_vel
    df_interpolado.attrs['r2_posicao'] = r2_pos
    df_interpolado.attrs['r2_velocidade'] = r2_vel
    df_interpolado.attrs['total_pontos'] = len(predictions_all)
    df_interpolado.attrs['num_sistemas'] = len(sistemas_unicos)
    
    print(f"\n  Base de dados interpolada gerada com {len(df_interpolado)} registros")
    print(f"  Colunas disponíveis: {list(df_interpolado.columns)}")
    
    return df_interpolado