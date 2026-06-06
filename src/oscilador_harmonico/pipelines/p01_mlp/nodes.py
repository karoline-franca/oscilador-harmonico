# nodes saída x,v; entrada [x0, v0, t]

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
    cria_grafico_interpolacao_completo,
    cria_grafico_interpolacao_espaco_fases,
    cria_grafico_interpolacao_pontual_mlp,
    cria_grafico_interpolacao_pontual_espaco_fases,
    cria_grafico_interpolacao_pontual_completo,
    cria_grafico_interpolacao_entre_trajetorias_espaco_fases,
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
    
    Entrada: [x0, v0, tempo]
    Saída: [posicao, velocidade]
    
    Nota: A frequência angular é constante para todos os dados.
    """
    
    for col in base_oscilador.columns:
        if base_oscilador[col].dtype == 'object':
            try:
                base_oscilador[col] = pd.to_numeric(base_oscilador[col].astype(str).str.replace(',', '.'), errors='coerce')
            except:
                pass
    
    if 'id_trajetoria' in base_oscilador.columns:
        base_oscilador['id_trajetoria'] = base_oscilador['id_trajetoria'].astype(str)
        if (base_oscilador['id_trajetoria'] == 'nan').any():
            print("  AVISO: Valores 'nan' encontrados em id_trajetoria. Recriando identificadores...")
            base_oscilador['id_trajetoria'] = 'sistema_' + base_oscilador['sistema_id'].astype(str) + '_condicao_' + base_oscilador['simulacao_id'].astype(str)
    
    if base_oscilador[['x0', 'v0', 'tempo']].isnull().any().any():
        print("  AVISO: Valores NaN detectados nas colunas numéricas!")
        base_oscilador = base_oscilador.dropna(subset=['x0', 'v0', 'tempo'])
    
    frequencia_angular_unica = base_oscilador['frequencia_angular'].iloc[0] if len(base_oscilador) > 0 else 5.0
    print(f"\n=== BASE DE DADOS ===")
    print(f"  Frequência angular do sistema: {frequencia_angular_unica} rad/s")
    print(f"  Total de linhas: {len(base_oscilador)}")
    
    if 'id_trajetoria' in base_oscilador.columns:
        trajetorias_unicas = base_oscilador['id_trajetoria'].unique()
        print(f"  Total de trajetórias únicas: {len(trajetorias_unicas)}")
        
        if len(trajetorias_unicas) == 1 and 'nan' in str(trajetorias_unicas[0]).lower():
            print("  AVISO: id_trajetoria ainda com problemas. Recriando baseado em x0, v0...")
            base_oscilador['id_trajetoria'] = 'x0_' + base_oscilador['x0'].round(6).astype(str) + \
                                              '_v0_' + base_oscilador['v0'].round(6).astype(str)
            trajetorias_unicas = base_oscilador['id_trajetoria'].unique()
            print(f"  Nova contagem de trajetórias: {len(trajetorias_unicas)}")
    else:
        print("  ERRO: Coluna 'id_trajetoria' não encontrada!")
        print("  Criando id_trajetoria baseado em x0, v0")
        base_oscilador['id_trajetoria'] = 'x0_' + base_oscilador['x0'].round(6).astype(str) + \
                                          '_v0_' + base_oscilador['v0'].round(6).astype(str)
        trajetorias_unicas = base_oscilador['id_trajetoria'].unique()
        print(f"  Total de trajetórias criadas: {len(trajetorias_unicas)}")
    
    features_entrada = ['x0', 'v0', 'tempo']
    features_saida = ['posicao', 'velocidade']
    
    print(f"\n=== PREPARAÇÃO DOS DADOS ===")
    
    if len(trajetorias_unicas) < 2:
        print(f"\n  ERRO: Apenas {len(trajetorias_unicas)} trajetória(s) encontrada(s)!")
        print("  Não é possível dividir em treino/validação/teste.")
        print("  Verifique se a base foi gerada corretamente com múltiplas condições iniciais.")
        print("\n  Criando divisão artificial para debug...")
        trajetorias_train = trajetorias_unicas[:max(1, len(trajetorias_unicas)//2)]
        trajetorias_val = []
        trajetorias_test = []
        
        if len(trajetorias_unicas) == 1:
            trajetorias_train = trajetorias_unicas
            trajetorias_val = trajetorias_unicas
            trajetorias_test = trajetorias_unicas
    else:
        # divide as trajetórias em treino, validação e teste (70-20-10)
        trajetorias_train, trajetorias_temp = train_test_split(
            trajetorias_unicas, test_size=0.30, random_state=42
        )
        trajetorias_val, trajetorias_test = train_test_split(
            trajetorias_temp, test_size=0.3333, random_state=42
        )
    
    print(f"  Trajetórias de treino: {len(trajetorias_train)}")
    print(f"  Trajetórias de validação: {len(trajetorias_val)}")
    print(f"  Trajetórias de teste: {len(trajetorias_test)}")
    
    # seleciona os dados de cada conjunto baseado nas trajetórias
    dados_train = base_oscilador[base_oscilador['id_trajetoria'].isin(trajetorias_train)]
    dados_val = base_oscilador[base_oscilador['id_trajetoria'].isin(trajetorias_val)]
    dados_test = base_oscilador[base_oscilador['id_trajetoria'].isin(trajetorias_test)]
    
    X_raw_train = dados_train[features_entrada].values.astype(np.float32)
    y_raw_train = dados_train[features_saida].values.astype(np.float32)
    
    X_raw_val = dados_val[features_entrada].values.astype(np.float32)
    y_raw_val = dados_val[features_saida].values.astype(np.float32)
    
    X_raw_test = dados_test[features_entrada].values.astype(np.float32)
    y_raw_test = dados_test[features_saida].values.astype(np.float32)
        
    print(f"  Amostras de treino: {X_raw_train.shape[0]}")
    print(f"  Amostras de validação: {X_raw_val.shape[0]}")
    print(f"  Amostras de teste: {X_raw_test.shape[0]}")
        
    # normalização padrão das variáveis de entrada e saída
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled_train = scaler_X.fit_transform(X_raw_train)
    X_scaled_val = scaler_X.transform(X_raw_val)
    X_scaled_test = scaler_X.transform(X_raw_test)
    
    y_scaled_train = scaler_y.fit_transform(y_raw_train)
    y_scaled_val = scaler_y.transform(y_raw_val)
    y_scaled_test = scaler_y.transform(y_raw_test)
    
    input_dim = X_raw_train.shape[1]
    output_dim = y_raw_train.shape[1]
    
    print(f"  Dimensão entrada: {input_dim} (x0, v0, t)")
    print(f"  Dimensão saída: {output_dim} (x, v)")
    
    return (X_scaled_train, y_scaled_train, 
            X_scaled_val, y_scaled_val, 
            X_scaled_test, y_scaled_test, 
            input_dim, output_dim, scaler_X, scaler_y,
            trajetorias_train, trajetorias_val, trajetorias_test)


def visualiza_distribuicao_dados_separado(
    base_oscilador: pd.DataFrame, 
    parameters: Dict[str, Any]
) -> None:
    """
    Node separado para visualizar a distribuição dos dados no espaço de fases.
    Carrega os dados novamente e faz a divisão por trajetória apenas para visualização.
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
    
    # obtém lista única de trajetórias
    trajetorias_unicas = base_oscilador['id_trajetoria'].unique()
    
    # divide as trajetórias em treino, validação e teste (70-20-10)
    trajetorias_train, trajetorias_temp = train_test_split(
        trajetorias_unicas, test_size=0.30, random_state=42
    )
    trajetorias_val, trajetorias_test = train_test_split(
        trajetorias_temp, test_size=0.3333, random_state=42
    )
        
    # seleciona os dados de cada conjunto baseado nas trajetórias
    dados_train = base_oscilador[base_oscilador['id_trajetoria'].isin(trajetorias_train)]
    dados_val = base_oscilador[base_oscilador['id_trajetoria'].isin(trajetorias_val)]
    dados_test = base_oscilador[base_oscilador['id_trajetoria'].isin(trajetorias_test)]
    
    y_pos_train = dados_train['posicao'].values.astype(np.float32).reshape(-1, 1)
    y_vel_train = dados_train['velocidade'].values.astype(np.float32).reshape(-1, 1)
    y_pos_val = dados_val['posicao'].values.astype(np.float32).reshape(-1, 1)
    y_vel_val = dados_val['velocidade'].values.astype(np.float32).reshape(-1, 1)
    y_pos_test = dados_test['posicao'].values.astype(np.float32).reshape(-1, 1)
    y_vel_test = dados_test['velocidade'].values.astype(np.float32).reshape(-1, 1)
    
    fig = cria_grafico_distribuicao_dados(
        y_pos_train=y_pos_train,
        y_vel_train=y_vel_train,
        y_pos_val=y_pos_val,
        y_vel_val=y_vel_val,
        y_pos_test=y_pos_test,
        y_vel_test=y_vel_test,
        titulo="Distribuição dos Dados - Espaço de Fases (Por Trajetória)"
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
    print(f"  Input dim: {input_dim} (x0, v0, t)")
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
    print(f"  Entrada: (x0, v0, t) -> Saída: (x, v)")
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


def avalia_metricas_mlp_node(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler
) -> Dict[str, float]:
    """Avalia o modelo MLP nos dados de validação e teste."""

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
        titulo="Real vs Previsto - Dados de Teste (Por Trajetória)"
    )
    
    fig.write_html(grafico_previsoes_mlp)    
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
        titulo="Previsões do Modelo no Espaço de Fases - Dados de Teste (Por Trajetória)"
    )
    
    fig.write_html(grafico_previsoes_espaco_fases)    
    fig.show()
    
    return None


def interpola_trajetorias_avulsas_node(
    model: nn.Module,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any]
) -> None:
    """
    Node: Usa o modelo treinado para fazer interpolações e prever trajetórias completas
    para novas condições iniciais não vistas durante o treinamento.
    Nota: A frequência angular é fixa para todos os casos.
    
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
    
    grafico_interpolacao_completa = f"{output_dir}/interpolacao_avulsa_v_x_vs_t.html"
    grafico_interpolacao_espaco_fases = f"{output_dir}/interpolacao_avulsa_espaco_fases.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    sim_params = parameters.get('simulation', {})
    dt = sim_params.get('dt', 0.01)
    intervals = parameters.get('intervals', {})
    omega_fixo = intervals.get('omega', 5.0)
    
    # casos de teste para interpolação (apenas condições iniciais variam)
    casos_teste = [
        {
            "nome": "Caso 1",
            "x0": 0.3,
            "v0": 0.0,
            "omega": omega_fixo,
            "t_final": 2 * np.pi / omega_fixo,
            "cor": CORES_PALETA[0]
        },
        {
            "nome": "Caso 2",
            "x0": -0.3,
            "v0": 1.0,
            "omega": omega_fixo,
            "t_final": 2 * np.pi / omega_fixo,
            "cor": CORES_PALETA[1]
        },
        {
            "nome": "Caso 3",
            "x0": 0.5,
            "v0": -1.0,
            "omega": omega_fixo,
            "t_final": 2 * np.pi / omega_fixo,
            "cor": CORES_PALETA[2]
        },
        {
            "nome": "Caso 4",
            "x0": -0.1,
            "v0": 0.5,
            "omega": omega_fixo,
            "t_final": 2 * np.pi / omega_fixo,
            "cor": CORES_PALETA[3]
        },

    ]
    
    # gera os nomes das legendas dinamicamente
    for caso in casos_teste:
        caso["nome_legenda"] = (
            f"{caso['nome']}: x0={caso['x0']:.1f} m, "
            f"v0={caso['v0']:.1f} m/s, "
            f"ω={omega_fixo:.1f} rad/s, "
            f"T={caso['t_final']:.2f} s"
        )
    
    tempos_lista = []
    posicoes_lista = []
    velocidades_lista = []
    
    for caso in casos_teste:
        t_max = caso["t_final"]
        tempos = np.arange(0, t_max + dt, dt)
        
        # prepara as entradas (apenas x0, v0, tempo)
        X_caso = np.zeros((len(tempos), 3))
        X_caso[:, 0] = caso["x0"]
        X_caso[:, 1] = caso["v0"]
        X_caso[:, 2] = tempos
        
        # normaliza e faz previsão (todos os pontos da mesma trajetória juntos)
        X_caso_scaled = scaler_X.transform(X_caso)
        X_tensor = torch.tensor(X_caso_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            predictions_scaled = model(X_tensor).cpu().numpy()
        
        predictions = scaler_y.inverse_transform(predictions_scaled)
        
        tempos_lista.append(tempos)
        posicoes_lista.append(predictions[:, 0])
        velocidades_lista.append(predictions[:, 1])
        
    fig_completo = cria_grafico_interpolacao_completo(
        tempos_lista=tempos_lista,
        posicoes_lista=posicoes_lista,
        velocidades_lista=velocidades_lista,
        casos_info=casos_teste,
        titulo="Interpolação Avulsa: Posição e Velocidade vs Tempo (Por Trajetória)"
    )
    
    fig_fases = cria_grafico_interpolacao_espaco_fases(
        posicoes_lista=posicoes_lista,
        velocidades_lista=velocidades_lista,
        casos_info=casos_teste,
        titulo="Interpolação Avulsa: Espaço de Fases (Por Trajetória)"
    )
    
    fig_completo.write_html(grafico_interpolacao_completa)
    fig_fases.write_html(grafico_interpolacao_espaco_fases)
    
    print("\n=== MODELO PARA INTERPOLAÇÃO ===")
    print(f"  Passo de tempo (dt): {dt} s")
    print(f"  Período: {2 * np.pi / omega_fixo:.2f} s")
    print(f"  Número de casos testados: {len(casos_teste)}")
    
    fig_completo.show()
    fig_fases.show()
    
    return None

def interpolacoes_pontuais_mlp_node(
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
    Nota: A interpolação é feita dentro da mesma trajetória, variando apenas o tempo.
    A frequência angular é constante.
    
    Args:
        model: Modelo MLP treinado
        X_test: Dados de teste NORMALIZADOS
        y_test: Dados de teste NORMALIZADOS
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
    
    intervals = parameters.get('intervals', {})
    omega_fixo = intervals.get('omega', 5.0)
    
    print("\n=== INTERPOLAÇÃO PONTUAL ===")
    print("  A interpolação é feita variando o tempo para uma mesma trajetória (x0, v0 constantes)")
    
    X_test_original = scaler_X.inverse_transform(X_test)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    X_test_df = pd.DataFrame(X_test_original, columns=['x0', 'v0', 'tempo'])
    
    # identifica trajetórias únicas (x0, v0 constantes) usando dados originais
    trajetorias_unicas = X_test_df.groupby(['x0', 'v0']).size().reset_index()
    trajetorias_unicas = trajetorias_unicas
    
    print(f"\n  Trajetórias únicas encontradas: {len(trajetorias_unicas)}")
        
    todas_previsoes = []
    todos_reais_interpolados = []
    
    # listas para o gráfico temporal
    tempos_lista = []
    posicoes_previstas_lista = []
    velocidades_previstas_lista = []
    posicoes_reais_lista = []
    velocidades_reais_lista = []
    casos_info_lista = []
    dados_interpolados = []
    
    for idx, row in trajetorias_unicas.iterrows():
        x0 = row['x0']
        v0 = row['v0']
        
        # filtra os pontos originais desta trajetória específica (usando dados originais)
        mask = (np.abs(X_test_original[:, 0] - x0) < 1e-6) & \
               (np.abs(X_test_original[:, 1] - v0) < 1e-6)
        
        tempos_originais = X_test_original[mask, 2]
        pos_originais = y_test_original[mask, 0]
        vel_originais = y_test_original[mask, 1]
        
        if len(tempos_originais) < 5:
            print(f"    Poucos pontos ({len(tempos_originais)} < 5). Pulando...")
            continue
        
        # ordena por tempo
        idx_sort = np.argsort(tempos_originais)
        tempos_originais = tempos_originais[idx_sort]
        pos_originais = pos_originais[idx_sort]
        vel_originais = vel_originais[idx_sort]
        
        # cria tempos interpolados entre os pontos originais da mesma trajetória
        t_min = tempos_originais.min()
        t_max = tempos_originais.max()
        
        # gera pontos de tempo interpolados (mais finos que os originais)
        num_pontos_interpolados = 100
        tempos_interpolados = np.linspace(t_min, t_max, num_pontos_interpolados)
        
        # para validação, obtém os valores reais nesses tempos (via solução analítica do OHS)
        # usando a frequência angular fixa do sistema
        pos_reais_interpolados = x0 * np.cos(omega_fixo * tempos_interpolados) + \
                                 (v0 / omega_fixo) * np.sin(omega_fixo * tempos_interpolados)
        vel_reais_interpolados = -x0 * omega_fixo * np.sin(omega_fixo * tempos_interpolados) + \
                                 v0 * np.cos(omega_fixo * tempos_interpolados)
        
        # prepara entrada para o modelo (apenas x0, v0, tempo)
        X_interpolado = np.zeros((len(tempos_interpolados), 3))
        X_interpolado[:, 0] = x0
        X_interpolado[:, 1] = v0
        X_interpolado[:, 2] = tempos_interpolados
        
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
            'omega': omega_fixo,
            'cor': cor
        })
        
        # registros para o DataFrame interpolado
        for k in range(len(tempos_interpolados)):
            dados_interpolados.append({
                'id_trajetoria': f"x0_{x0:.3f}_v0_{v0:.3f}",
                'x0': x0,
                'v0': v0,
                'omega': omega_fixo,
                'tempo_interpolado': tempos_interpolados[k],
                'posicao_analitica': pos_reais_interpolados[k],
                'velocidade_analitica': vel_reais_interpolados[k],
                'posicao_prevista_mlp': pred[k, 0],
                'velocidade_prevista_mlp': pred[k, 1],
                'erro_posicao': pred[k, 0] - pos_reais_interpolados[k],
                'erro_velocidade': pred[k, 1] - vel_reais_interpolados[k],
                'erro_abs_posicao': abs(pred[k, 0] - pos_reais_interpolados[k]),
                'erro_abs_velocidade': abs(pred[k, 1] - vel_reais_interpolados[k]),
            })
    
    if len(todas_previsoes) == 0:
        print("  ERRO: Nenhuma trajetória válida encontrada para interpolação")
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
        titulo="Interpolação Pontual: Solução Analítica vs MLP - Dados de Teste (Por Trajetória)"
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
        titulo="Interpolação Pontual: MLP vs Solução Analítica - Espaço de Fases (Por Trajetória)"
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
        titulo="Interpolação Pontual: MLP vs Solução Analítica - Posição e Velocidade vs Tempo (Por Trajetória)"
    )
    
    fig3.write_html(grafico_interpolacao_pontual_temporal)
    print(f"  Gráfico de interpolação pontual (Posição/Velocidade vs Tempo) salvo em {grafico_interpolacao_pontual_temporal}")
    
    fig1.show()
    fig2.show()
    fig3.show()
    
    df_interpolado = pd.DataFrame(dados_interpolados)
    
    df_interpolado.attrs['rmse_posicao'] = rmse_pos
    df_interpolado.attrs['rmse_velocidade'] = rmse_vel
    df_interpolado.attrs['r2_posicao'] = r2_pos
    df_interpolado.attrs['r2_velocidade'] = r2_vel
    df_interpolado.attrs['total_pontos'] = len(predictions_all)
    df_interpolado.attrs['num_trajetorias'] = len(trajetorias_unicas)
    df_interpolado.attrs['pontos_por_trajetoria'] = 100
    df_interpolado.attrs['omega_fixo'] = omega_fixo
    
    print(f"\n  Base de dados com interpolação temporal dentro de cada trajetória gerada com {len(df_interpolado)} registros")
    
    return df_interpolado


def interpola_entre_trajetorias_mlp_node(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Node: Usa o modelo treinado para fazer interpolação entre trajetórias.
    Para cada instante de tempo, interpola entre duas trajetórias diferentes (variando x0 e v0).
    Não mistura dados de treino/validação/teste pois usa apenas dados de teste.
    """
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_interpolacao_entre_trajetorias = f"{output_dir}/interpolacoes_entre_trajetorias_real_previsto_mlp.html"
    grafico_interpolacao_entre_trajetorias_espaco_fases = f"{output_dir}/interpolacao_entre_trajetorias_espaco_fases.html"
    grafico_interpolacao_entre_trajetorias_temporal = f"{output_dir}/interpolacao_entre_trajetorias_v_x_vs_t.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    intervals = parameters.get('intervals', {})
    omega_fixo = intervals.get('omega', 5.0)
    
    seed = parameters.get('seed', 42)
    np.random.seed(seed)
    
    print("\n=== INTERPOLAÇÃO ENTRE TRAJETÓRIAS ===")
    print(f"  Frequência: {omega_fixo} rad/s")
    print("  Para cada instante de tempo, interpola entre duas trajetórias diferentes")
    
    X_test_original = scaler_X.inverse_transform(X_test)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    X_test_df = pd.DataFrame(X_test_original, columns=['x0', 'v0', 'tempo'])
    
    # identifica trajetórias únicas (x0, v0 constantes) usando dados originais
    trajetorias_unicas = X_test_df.groupby(['x0', 'v0']).size().reset_index()
    trajetorias_unicas = trajetorias_unicas
    
    print(f"\n  Trajetórias únicas encontradas: {len(trajetorias_unicas)}")
    
    if len(trajetorias_unicas) < 2:
        print("  ERRO: Precisamos de pelo menos 2 trajetórias para interpolação")
        return pd.DataFrame()
    
    # ============================================================
    # TRAJETÓRIAS QUE DELIMITAM UM INTERVALO
    # ============================================================
    # calcula a "energia" ou "amplitude" para cada trajetória
    # amplitude máxima sqrt(x0^2 + (v0/ω)^2)
    trajetorias_unicas['amplitude'] = np.sqrt(
        trajetorias_unicas['x0']**2 + (trajetorias_unicas['v0'] / omega_fixo)**2
    )
    
    trajetorias_unicas = trajetorias_unicas.sort_values('amplitude').reset_index(drop=True)
    
    # seleciona uma trajetória com amplitude pequena e outra com amplitude grande
    n_traj = len(trajetorias_unicas)
    # idx_pequena = 0  # menor amplitude
    # idx_grande = n_traj - 1  # maior amplitude
    
    # aleatório mas diferentes em amplitude
    idx_pequena = np.random.choice(n_traj // 2)  # primeira metade (amplitudes menores)
    idx_grande = np.random.choice(range(n_traj // 2, n_traj))  # segunda metade (amplitudes maiores)
    
    traj1 = trajetorias_unicas.iloc[idx_pequena]
    traj2 = trajetorias_unicas.iloc[idx_grande]
    
    x0_1, v0_1 = traj1['x0'], traj1['v0']
    x0_2, v0_2 = traj2['x0'], traj2['v0']
    
    print(f"\n  Trajetória 1 (menor amplitude - {traj1['amplitude']:.3f}): x0={x0_1:.3f}, v0={v0_1:.3f}")
    print(f"  Trajetória 2 (maior amplitude - {traj2['amplitude']:.3f}): x0={x0_2:.3f}, v0={v0_2:.3f}")
    
    tempos_unicos = np.sort(np.unique(X_test_original[:, 2]))
    print(f"  Instantes de tempo únicos entre {tempos_unicos.min():.3f} s e {tempos_unicos.max():.3f} s")
    
    alphas = np.linspace(0, 1, 3)
    print(f"  Fatores de interpolação: {alphas}")
    
    todas_previsoes = []
    todos_reais_interpolados = []
    
    tempos_lista = []
    posicoes_previstas_lista = []
    velocidades_previstas_lista = []
    posicoes_reais_lista = []
    velocidades_reais_lista = []
    casos_info_lista = []
    dados_interpolados = []
    
    for alpha in alphas:
        x0_interp = (1 - alpha) * x0_1 + alpha * x0_2
        v0_interp = (1 - alpha) * v0_1 + alpha * v0_2
        amplitude_interp = np.sqrt(x0_interp**2 + (v0_interp / omega_fixo)**2)
        if alpha == 0.5:
            print(f"  Amplitude interpolada para α={alpha:.1f}: {amplitude_interp:.3f}")
        
        X_interpolado = np.zeros((len(tempos_unicos), 3))
        X_interpolado[:, 0] = x0_interp
        X_interpolado[:, 1] = v0_interp
        X_interpolado[:, 2] = tempos_unicos
        
        X_interpolado_scaled = scaler_X.transform(X_interpolado)
        X_tensor = torch.tensor(X_interpolado_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()
        
        pred = scaler_y.inverse_transform(pred_scaled)
        
        pos_analitico = x0_interp * np.cos(omega_fixo * tempos_unicos) + \
                        (v0_interp / omega_fixo) * np.sin(omega_fixo * tempos_unicos)
        vel_analitico = -x0_interp * omega_fixo * np.sin(omega_fixo * tempos_unicos) + \
                        v0_interp * np.cos(omega_fixo * tempos_unicos)
        
        todas_previsoes.append(pred)
        todos_reais_interpolados.append(np.column_stack([pos_analitico, vel_analitico]))
        
        tempos_lista.append(tempos_unicos)
        posicoes_previstas_lista.append(pred[:, 0])
        velocidades_previstas_lista.append(pred[:, 1])
        posicoes_reais_lista.append(pos_analitico)
        velocidades_reais_lista.append(vel_analitico)
        
        cor_idx = int(alpha * (len(CORES_PALETA) - 1))
        cor = CORES_PALETA[cor_idx]
        
        casos_info_lista.append({
            'alpha': alpha,
            'x0': x0_interp,
            'v0': v0_interp,
            'omega': omega_fixo,
            'cor': cor
        })
        
        for k in range(len(tempos_unicos)):
            dados_interpolados.append({
                'alpha_interpolacao': alpha,
                'x0_original_1': x0_1,
                'v0_original_1': v0_1,
                'x0_original_2': x0_2,
                'v0_original_2': v0_2,
                'x0_interpolado': x0_interp,
                'v0_interpolado': v0_interp,
                'omega': omega_fixo,
                'tempo': tempos_unicos[k],
                'posicao_analitica': pos_analitico[k],
                'velocidade_analitica': vel_analitico[k],
                'posicao_prevista_mlp': pred[k, 0],
                'velocidade_prevista_mlp': pred[k, 1],
                'erro_posicao': pred[k, 0] - pos_analitico[k],
                'erro_velocidade': pred[k, 1] - vel_analitico[k],
                'erro_abs_posicao': abs(pred[k, 0] - pos_analitico[k]),
                'erro_abs_velocidade': abs(pred[k, 1] - vel_analitico[k]),
                'erro_rel_posicao_pct': (abs(pred[k, 0] - pos_analitico[k]) / (abs(pos_analitico[k]) + 1e-6)) * 100,
                'erro_rel_velocidade_pct': (abs(pred[k, 1] - vel_analitico[k]) / (abs(vel_analitico[k]) + 1e-6)) * 100,
            })
    
    if len(todas_previsoes) == 0:
        print("  ERRO: Nenhuma interpolação realizada")
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
        titulo="Interpolação entre Trajetórias: Solução Analítica vs MLP"
    )
    
    fig1.write_html(grafico_interpolacao_entre_trajetorias)
    
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
        titulo="Interpolação entre Trajetórias: MLP vs Solução Analítica - Espaço de Fases"
    )
    
    fig2.write_html(grafico_interpolacao_entre_trajetorias_espaco_fases)
    
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
        titulo="Interpolação entre Trajetórias: MLP vs Solução Analítica - Posição e Velocidade vs Tempo"
    )
    
    fig3.write_html(grafico_interpolacao_entre_trajetorias_temporal)
    
    fig1.show()
    fig2.show()
    fig3.show()
    
    df_interpolado = pd.DataFrame(dados_interpolados)
    
    df_interpolado.attrs['rmse_posicao'] = rmse_pos
    df_interpolado.attrs['rmse_velocidade'] = rmse_vel
    df_interpolado.attrs['r2_posicao'] = r2_pos
    df_interpolado.attrs['r2_velocidade'] = r2_vel
    df_interpolado.attrs['total_pontos'] = len(predictions_all)
    df_interpolado.attrs['num_trajetorias'] = 2
    df_interpolado.attrs['num_alpha'] = len(alphas)
    df_interpolado.attrs['num_tempos'] = len(tempos_unicos)
    df_interpolado.attrs['omega_fixo'] = omega_fixo
    
    print(f"\n  Base de dados com interpolação entre trajetórias gerada com {len(df_interpolado)} registros")
    print(f"  - Trajetória 1: (x0={x0_1:.3f}, v0={v0_1:.3f}) - Amplitude: {np.sqrt(x0_1**2 + (v0_1/omega_fixo)**2):.3f}")
    print(f"  - Trajetória 2: (x0={x0_2:.3f}, v0={v0_2:.3f}) - Amplitude: {np.sqrt(x0_2**2 + (v0_2/omega_fixo)**2):.3f}")
    print(f"  - {len(alphas)} níveis de interpolação")
    print(f"  - {len(tempos_unicos)} instantes de tempo por trajetória")

    # ========================================================================
    # GRÁFICO 4: Trajetórias Originais e Interpoladas no Espaço de Fases
    # ========================================================================
    
    # trajetória 1
    mask_traj1 = (np.abs(X_test_original[:, 0] - x0_1) < 1e-6) & \
                 (np.abs(X_test_original[:, 1] - v0_1) < 1e-6)
    idx_traj1 = np.argsort(X_test_original[mask_traj1, 2])
    traj1_pos = y_test_original[mask_traj1, 0][idx_traj1]
    traj1_vel = y_test_original[mask_traj1, 1][idx_traj1]
    
    # trajetória 2
    mask_traj2 = (np.abs(X_test_original[:, 0] - x0_2) < 1e-6) & \
                 (np.abs(X_test_original[:, 1] - v0_2) < 1e-6)
    idx_traj2 = np.argsort(X_test_original[mask_traj2, 2])
    traj2_pos = y_test_original[mask_traj2, 0][idx_traj2]
    traj2_vel = y_test_original[mask_traj2, 1][idx_traj2]
    
    # interpolações
    interpolacoes_para_grafico = []
    alphas_unicos = np.sort(df_interpolado['alpha_interpolacao'].unique())
    
    for alpha in alphas_unicos:
        if alpha == 0 or alpha == 1:
            continue
        
        mask_alpha = df_interpolado['alpha_interpolacao'] == alpha
        dados_alpha = df_interpolado[mask_alpha].sort_values('tempo')
        
        x0_interp = dados_alpha['x0_interpolado'].iloc[0]
        v0_interp = dados_alpha['v0_interpolado'].iloc[0]
        
        interpolacoes_para_grafico.append({
            'alpha': alpha,
            'posicoes': dados_alpha['posicao_prevista_mlp'].values,
            'velocidades': dados_alpha['velocidade_prevista_mlp'].values,
            'x0_interp': x0_interp,
            'v0_interp': v0_interp
        })
    
    casos_info_grafico = [{
        'x0_1': x0_1,
        'v0_1': v0_1,
        'x0_2': x0_2,
        'v0_2': v0_2
    }]
    
    fig4 = cria_grafico_interpolacao_entre_trajetorias_espaco_fases(
        trajetoria1_pos=traj1_pos,
        trajetoria1_vel=traj1_vel,
        trajetoria2_pos=traj2_pos,
        trajetoria2_vel=traj2_vel,
        interpolacoes_lista=interpolacoes_para_grafico,
        casos_info=casos_info_grafico,
        titulo="Interpolação entre Trajetórias no Espaço de Fases"
    )
    
    grafico_entre_trajetorias_espaco_fases = f"{output_dir}/interpolacao_entre_trajetorias_espaco_fases_detalhado.html"
    fig4.write_html(grafico_entre_trajetorias_espaco_fases)
    
    fig4.show()

    return df_interpolado