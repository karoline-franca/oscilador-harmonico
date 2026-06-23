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
    cria_grafico_interpolacao_completo,
    cria_grafico_interpolacao_espaco_fases,
    cria_grafico_interpolacao_pontual_mlp,
    cria_grafico_interpolacao_pontual_espaco_fases,
    cria_grafico_interpolacao_pontual_completo,
    cria_grafico_interpolacao_entre_trajetorias_espaco_fases,
    cria_grafico_interpolacao_trajetorias_espaco_fases,
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
    
    Nota: Divide os dados por trajetória para evitar mistura de trajetórias diferentes.
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
    
    if base_oscilador[['x0', 'v0', 'frequencia_angular', 'tempo']].isnull().any().any():
        print("  AVISO: Valores NaN detectados nas colunas numéricas!")
        base_oscilador = base_oscilador.dropna(subset=['x0', 'v0', 'frequencia_angular', 'tempo'])
    
    print(f"\n=== BASE DE DADOS ===")
    print(f"  Total de linhas: {len(base_oscilador)}")
    
    if 'id_trajetoria' in base_oscilador.columns:
        trajetorias_unicas = base_oscilador['id_trajetoria'].unique()
        print(f"  Total de trajetórias únicas: {len(trajetorias_unicas)}")
        
        if len(trajetorias_unicas) == 1 and 'nan' in str(trajetorias_unicas[0]).lower():
            print("  AVISO: id_trajetoria ainda com problemas. Recriando baseado em x0, v0, ω...")
            base_oscilador['id_trajetoria'] = 'x0_' + base_oscilador['x0'].round(6).astype(str) + \
                                              '_v0_' + base_oscilador['v0'].round(6).astype(str) + \
                                              '_omega_' + base_oscilador['frequencia_angular'].round(6).astype(str)
            trajetorias_unicas = base_oscilador['id_trajetoria'].unique()
            print(f"  Nova contagem de trajetórias: {len(trajetorias_unicas)}")
    else:
        print("  ERRO: Coluna 'id_trajetoria' não encontrada!")
        print("  Criando id_trajetoria baseado em x0, v0, ω")
        base_oscilador['id_trajetoria'] = 'x0_' + base_oscilador['x0'].round(6).astype(str) + \
                                          '_v0_' + base_oscilador['v0'].round(6).astype(str) + \
                                          '_omega_' + base_oscilador['frequencia_angular'].round(6).astype(str)
        trajetorias_unicas = base_oscilador['id_trajetoria'].unique()
        print(f"  Total de trajetórias criadas: {len(trajetorias_unicas)}")
    
    features_entrada = ['x0', 'v0', 'frequencia_angular', 'tempo']
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
    
    print(f"  Dimensão entrada: {input_dim} (x0, v0, ω, t)")
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


def visualiza_previsoes_teste_node(
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
    
    grafico_previsoes_mlp = f"{output_dir}/real_previsto_teste.html"
    
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


def visualiza_previsoes_espaco_fases_teste_node(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any]
) -> None:
    """
    Node: Visualiza as previsões do modelo no espaço de fases (Posição vs Velocidade).
    
    Args:
        model: Modelo treinado
        X_test: Dados de teste
        y_test: Targets de teste
        scaler_X: Scaler das features de entrada
        scaler_y: Scaler dos targets
        parameters: Parâmetros do pipeline
    """
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'default_v1')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_previsoes_espaco_fases = f"{output_dir}/previsoes_espaco_fases_teste.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print("\n=== VISUALIZAÇÃO DAS PREVISÕES NO ESPAÇO DE FASES ===")
    print(f"  Amostras de teste: {len(X_test)}")
    
    X_test_original = scaler_X.inverse_transform(X_test)
    frequencias_teste = X_test_original[:, 2]
    frequencias_unicas = np.unique(frequencias_teste)
    print(f"  Frequências encontradas nos dados de teste: {frequencias_unicas}")
    
    X_test_df = pd.DataFrame(X_test_original, columns=['x0', 'v0', 'omega', 'tempo'])
    
    # identifica trajetórias únicas (x0, v0, ω constantes)
    trajetorias_unicas = X_test_df.groupby(['x0', 'v0', 'omega']).size().reset_index()
    total_trajetorias = len(trajetorias_unicas)
    
    # seleciona 2 trajetórias aleatórias baseado em (x0, v0, ω)
    np.random.seed(42)
    indices_selecionados = np.random.choice(total_trajetorias, size=min(2, total_trajetorias), replace=False)
    trajetorias_selecionadas = trajetorias_unicas.iloc[indices_selecionados]
    
    print(f"\n  Total de trajetórias disponíveis: {total_trajetorias}")
    print(f"  Visualizando apenas {len(trajetorias_selecionadas)} trajetória(s) selecionada(s) aleatoriamente:")
    for idx, row in trajetorias_selecionadas.iterrows():
        print(f"    - x0={row['x0']:.3f}, v0={row['v0']:.3f}, ω={row['omega']:.3f} rad/s")
    
    # filtra os dados para incluir apenas as trajetórias selecionadas
    mask_selecionadas = np.zeros(len(X_test_original), dtype=bool)
    for _, row in trajetorias_selecionadas.iterrows():
        mask = (np.abs(X_test_original[:, 0] - row['x0']) < 1e-6) & \
               (np.abs(X_test_original[:, 1] - row['v0']) < 1e-6) & \
               (np.abs(X_test_original[:, 2] - row['omega']) < 1e-6)
        mask_selecionadas = mask_selecionadas | mask
    
    X_test_filtrado = X_test[mask_selecionadas]
    y_test_filtrado = y_test[mask_selecionadas]
    X_test_original_filtrado = X_test_original[mask_selecionadas]
    frequencias_teste_filtradas = X_test_original_filtrado[:, 2]
    
    print(f"  Amostras após filtro: {len(X_test_filtrado)}")
    
    X_test_tensor = torch.tensor(X_test_filtrado, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).cpu().numpy()
    
    # desnormaliza previsões
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_test_original_filtrado = scaler_y.inverse_transform(y_test_filtrado)
    
    y_pos_true = y_test_original_filtrado[:, 0].reshape(-1, 1)
    y_vel_true = y_test_original_filtrado[:, 1].reshape(-1, 1)
    y_pos_pred = predictions[:, 0].reshape(-1, 1)
    y_vel_pred = predictions[:, 1].reshape(-1, 1)
    
    # calcula métricas gerais para exibição (apenas sobre os sistemas selecionados)
    rmse_pos = np.sqrt(mean_squared_error(y_pos_true, y_pos_pred))
    rmse_vel = np.sqrt(mean_squared_error(y_vel_true, y_vel_pred))
    r2_pos = r2_score(y_pos_true, y_pos_pred)
    r2_vel = r2_score(y_vel_true, y_vel_pred)
    
    print(f"\n  Métricas globais (apenas para os {len(trajetorias_selecionadas)} sistemas selecionados):")
    print(f"    RMSE Posição: {rmse_pos:.6f} m")
    print(f"    RMSE Velocidade: {rmse_vel:.6f} m/s")
    print(f"    R² Posição: {r2_pos:.4f}")
    print(f"    R² Velocidade: {r2_vel:.4f}")
    
    fig = cria_grafico_previsoes_espaco_fases(
        y_pos_true=y_pos_true,
        y_vel_true=y_vel_true,
        y_pos_pred=y_pos_pred,
        y_vel_pred=y_vel_pred,
        frequencias=frequencias_teste_filtradas,
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
    para novas condições iniciais e frequências não vistas durante o treinamento.
    
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
    
    print("\n=== INICIANDO MODELO PARA INTERPOLAÇÃO ===")
    
    # casos de teste para interpolação
    casos_teste = [
        {
            "nome": "Caso 1",
            "x0": -0.3,
            "v0": 1.0,
            "omega": 1.0,
            "t_final": 2 * np.pi / 1.0,
            "cor": CORES_PALETA[1]
        },
        {
            "nome": "Caso 2",
            "x0": 0.5,
            "v0": -1.0,
            "omega": 1.0,
            "t_final": 2 * np.pi / 1.0,
            "cor": CORES_PALETA[2]
        },
        {
            "nome": "Caso 3",
            "x0": -0.1,
            "v0": 0.5,
            "omega": 4.0,
            "t_final": 2 * np.pi / 4.0,
            "cor": CORES_PALETA[3]
        },
        {
            "nome": "Caso 4",
            "x0": 0.0,
            "v0": -1.0,
            "omega": 4.0,
            "t_final": 2 * np.pi / 4.0,
            "cor": CORES_PALETA[4]
        },
    ]
    
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
    
    for caso in casos_teste:
        t_max = caso["t_final"]
        num_passos = int(np.ceil(t_max / dt))
        tempos = np.linspace(0, t_max, num_passos + 1)  # +1 para incluir t_max
        
        print(f"  Processando {caso['nome']}: {len(tempos)} passos temporais, dt={dt:.3f}s")
        
        # prepara as entradas (x0, v0, ω, tempo)
        X_caso = np.zeros((len(tempos), 4))
        X_caso[:, 0] = caso["x0"]
        X_caso[:, 1] = caso["v0"]
        X_caso[:, 2] = caso["omega"]
        X_caso[:, 3] = tempos
        
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
        
    fig_completo.show()
    fig_fases.show()
    
    return None


def interpolacoes_pontuais_mlp_node(
    model: nn.Module,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Node: Usa o modelo treinado para fazer interpolação entre pontos gerados aleatoriamente.
    Faz previsões pontuais independentes em pontos que não estão no conjunto original de treino/validação/teste.
    Nota: A interpolação é feita dentro da mesma trajetória, variando apenas o tempo.
    Cada sistema é tratado separadamente sem mistura entre sistemas.
    
    Args:
        model: Modelo MLP treinado
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
    
    print("\n=== INTERPOLAÇÃO PONTUAL ===")
    print("\n  A interpolação é feita variando o tempo para uma mesma trajetória")
    
    # ============================================
    # GERAÇÃO DE DADOS ALEATÓRIOS
    # ============================================
    
    intervals = parameters.get('intervals', {})
    seed = parameters.get('seed', 42)
    
    # limites dos intervalos dos parâmetros
    omega_min = intervals.get('omega_min', 1.0)
    omega_max = intervals.get('omega_max', 5.0)
    x0_min = intervals.get('x0_min', -0.5)
    x0_max = intervals.get('x0_max', 0.5)
    v0_min = intervals.get('v0_min', -1.0)
    v0_max = intervals.get('v0_max', 1.0)
    
    # fixa a semente para reprodutibilidade
    np.random.seed(seed)
    
    # número de trajetórias a serem geradas
    num_trajetorias = 2
    
    # gera condições iniciais e frequências aleatórias
    x0_values = np.random.uniform(x0_min, x0_max, num_trajetorias)
    v0_values = np.random.uniform(v0_min, v0_max, num_trajetorias)
    omega_values = np.random.uniform(omega_min, omega_max, num_trajetorias)
    while len(np.unique(omega_values)) < num_trajetorias:
        omega_values = np.random.uniform(omega_min, omega_max, num_trajetorias)

    # número de pontos por trajetória
    num_pontos_por_trajetoria = 1000
    
    # estrutura para armazenar as trajetórias
    trajetorias_unicas = []
    
    print(f"\n  Configuração da interpolação:")
    print(f"    Pontos por trajetória: {num_pontos_por_trajetoria}")
    
    for idx in range(num_trajetorias):
        x0 = x0_values[idx]
        v0 = v0_values[idx]
        omega = omega_values[idx]
        
        # calcula o período do sistema
        T = 2 * np.pi / omega
        
        # define o tempo máximo como 1 período
        tempo_maximo = T
        
        # calcula o passo temporal
        dt_interpolacao = tempo_maximo / (num_pontos_por_trajetoria - 1)
        
        print(f"\n  Trajetória {idx+1}:")
        print(f"    x0={x0:.3f}, v0={v0:.3f}, ω={omega:.3f} rad/s")
        print(f"    Período: {T:.3f} s")
        print(f"    Tempo máximo: {tempo_maximo:.3f} s")
        print(f"    Passo temporal: {dt_interpolacao:.6f} s")
        
        trajetorias_unicas.append({
            'x0': x0,
            'v0': v0,
            'omega': omega,
            'T': T,
            'tempo_maximo': tempo_maximo,
            'dt': dt_interpolacao
        })
    
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
    
    for idx, traj in enumerate(trajetorias_unicas):
        x0 = traj['x0']
        v0 = traj['v0']
        omega = traj['omega']
        tempo_maximo = traj['tempo_maximo']
                
        # gera tempos interpolados para 1 período completo
        tempos_interpolados = np.linspace(0, tempo_maximo, num_pontos_por_trajetoria)
                
        # solução analítica
        pos_reais_interpolados = x0 * np.cos(omega * tempos_interpolados) + \
                                 (v0 / omega) * np.sin(omega * tempos_interpolados)
        vel_reais_interpolados = -x0 * omega * np.sin(omega * tempos_interpolados) + \
                                 v0 * np.cos(omega * tempos_interpolados)
        
        # prepara entrada para o modelo (x0, v0, omega, tempo)
        X_interpolado = np.zeros((len(tempos_interpolados), 4))
        X_interpolado[:, 0] = x0
        X_interpolado[:, 1] = v0
        X_interpolado[:, 2] = omega
        X_interpolado[:, 3] = tempos_interpolados
        
        # normaliza e faz previsão (todos os pontos da mesma trajetória juntos)
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
        
        cor = CORES_PALETA[idx % len(CORES_PALETA)]
        
        casos_info_lista.append({
            'x0': x0,
            'v0': v0,
            'omega': omega,
            'cor': cor
        })
        
        for k in range(len(tempos_interpolados)):
            dados_interpolados.append({
                'id_trajetoria': f"x0_{x0:.3f}_v0_{v0:.3f}_omega_{omega:.3f}",
                'x0': x0,
                'v0': v0,
                'omega': omega,
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
    
    print(f"\n  RMSE Posição (vs solução analítica): {rmse_pos:.6f} m")
    print(f"  RMSE Velocidade (vs solução analítica): {rmse_vel:.6f} m/s")
    print(f"  R² Posição (vs solução analítica): {r2_pos:.4f}")
    print(f"  R² Velocidade (vs solução analítica): {r2_vel:.4f}")
    
    # ============================================
    # GRÁFICO 1: Real vs Previsto
    # ============================================
    
    fig1 = cria_grafico_interpolacao_pontual_mlp(
        predictions=predictions_all,
        y_true=y_true_all,
        titulo="Interpolação Pontual: Solução Analítica vs MLP - Dados Gerados Aleatoriamente (Por Trajetória)"
    )
    
    fig1.write_html(grafico_interpolacao_pontual)
    
    # ============================================
    # GRÁFICO 2: Espaço de Fases
    # ============================================
    
    y_pos_true = y_true_all[:, 0].reshape(-1, 1)
    y_vel_true = y_true_all[:, 1].reshape(-1, 1)
    y_pos_pred = predictions_all[:, 0].reshape(-1, 1)
    y_vel_pred = predictions_all[:, 1].reshape(-1, 1)
    
    # gera frequências para o gráfico de espaço de fases
    frequencias_all = []
    for traj in trajetorias_unicas:
        frequencias_all.extend([traj['omega']] * num_pontos_por_trajetoria)
    
    frequencias_all = np.array(frequencias_all)
    
    fig2 = cria_grafico_interpolacao_pontual_espaco_fases(
        y_pos_true=y_pos_true,
        y_vel_true=y_vel_true,
        y_pos_pred=y_pos_pred,
        y_vel_pred=y_vel_pred,
        frequencias=frequencias_all,
        titulo="Interpolação Pontual: MLP vs Solução Analítica - Espaço de Fases (Dados Gerados Aleatoriamente)"
    )
        
    fig2.write_html(grafico_interpolacao_pontual_espaco_fases)
    
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
        titulo="Interpolação Pontual: MLP vs Solução Analítica - Posição e Velocidade vs Tempo (Dados Gerados Aleatoriamente)"
    )
    
    fig3.write_html(grafico_interpolacao_pontual_temporal)
    
    fig1.show()
    fig2.show()
    fig3.show()
    
    df_interpolado = pd.DataFrame(dados_interpolados)
    
    df_interpolado.attrs['rmse_posicao'] = rmse_pos
    df_interpolado.attrs['rmse_velocidade'] = rmse_vel
    df_interpolado.attrs['r2_posicao'] = r2_pos
    df_interpolado.attrs['r2_velocidade'] = r2_vel
    df_interpolado.attrs['total_pontos'] = len(predictions_all)
    df_interpolado.attrs['num_trajetorias'] = num_trajetorias
    df_interpolado.attrs['pontos_por_trajetoria'] = num_pontos_por_trajetoria
    df_interpolado.attrs['omega_min'] = omega_min
    df_interpolado.attrs['omega_max'] = omega_max
    df_interpolado.attrs['x0_min'] = x0_min
    df_interpolado.attrs['x0_max'] = x0_max
    df_interpolado.attrs['v0_min'] = v0_min
    df_interpolado.attrs['v0_max'] = v0_max
    
    return df_interpolado


def interpola_entre_trajetorias_mlp_node(
    model: nn.Module,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Node: Usa o modelo treinado para fazer interpolação entre trajetórias.
    Para cada instante de tempo, interpola entre duas trajetórias diferentes (variando x0, v0 e ω).
    Não mistura dados de treino/validação/teste pois usa dados gerados aleatoriamente.
    Garante que cada sistema é tratado separadamente e que a interpolação respeita os passos temporais.
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
    
    seed = parameters.get('seed', 42)
    np.random.seed(seed)
    
    print("\n=== INTERPOLAÇÃO ENTRE TRAJETÓRIAS ===")
    print("\n  Para cada instante de tempo, interpola entre duas trajetórias diferentes")
    
    # ============================================
    # GERAÇÃO DE DADOS ALEATÓRIOS
    # ============================================
    
    # limites dos intervalos dos parâmetros
    omega_min = intervals.get('omega_min', 1.0)
    omega_max = intervals.get('omega_max', 5.0)
    x0_min = intervals.get('x0_min', -0.5)
    x0_max = intervals.get('x0_max', 0.5)
    v0_min = intervals.get('v0_min', -1.0)
    v0_max = intervals.get('v0_max', 1.0)
    
    # número de sistemas diferentes
    num_sistemas = 2
    num_trajetorias_por_sistema = 2  # cada sistema precisa de 2 trajetórias para interpolação
    
    # gera frequências diferentes para cada sistema
    frequencias_sistemas = np.random.uniform(omega_min, omega_max, num_sistemas)
    while len(np.unique(frequencias_sistemas)) < num_sistemas:
        frequencias_sistemas = np.random.uniform(omega_min, omega_max, num_sistemas)
    
    # número de pontos por trajetória
    num_pontos_por_trajetoria = 1000
    
    # estrutura para armazenar os sistemas
    sistemas_dados = {}
    trajetorias_unicas_geradas = []
    
    for sistema_idx, omega_sistema in enumerate(frequencias_sistemas):
        print(f"\n  Sistema {sistema_idx + 1}: ω={omega_sistema:.3f} rad/s")
        
        # calcula o período do sistema
        T = 2 * np.pi / omega_sistema
        tempo_maximo = T  # 1 período completo
        
        # calcula o passo temporal
        dt_interpolacao = tempo_maximo / (num_pontos_por_trajetoria - 1)
        
        print(f"    Período: {T:.3f} s")
        print(f"    Passo temporal: {dt_interpolacao:.6f} s")
        
        # gera tempos para este sistema
        tempos_unicos = np.linspace(0, tempo_maximo, num_pontos_por_trajetoria)
        
        # gera 2 trajetórias com condições iniciais diferentes para este sistema
        x0_values = np.random.uniform(x0_min, x0_max, num_trajetorias_por_sistema)
        v0_values = np.random.uniform(v0_min, v0_max, num_trajetorias_por_sistema)
        
        # garante que as trajetórias sejam diferentes
        while len(np.unique(np.column_stack([x0_values, v0_values]), axis=0)) < num_trajetorias_por_sistema:
            x0_values = np.random.uniform(x0_min, x0_max, num_trajetorias_por_sistema)
            v0_values = np.random.uniform(v0_min, v0_max, num_trajetorias_por_sistema)
        
        trajetorias_sistema = []
        for traj_idx in range(num_trajetorias_por_sistema):
            x0 = x0_values[traj_idx]
            v0 = v0_values[traj_idx]
            
            print(f"    Trajetória {traj_idx + 1}: x0={x0:.3f}, v0={v0:.3f}")
            
            trajetorias_sistema.append({
                'x0': x0,
                'v0': v0,
                'omega': omega_sistema,
                'tempos': tempos_unicos
            })
            
            trajetorias_unicas_geradas.append({
                'x0': x0,
                'v0': v0,
                'omega': omega_sistema
            })
        
        sistemas_dados[omega_sistema] = {
            'sistema_id': sistema_idx + 1,
            'x0_1': trajetorias_sistema[0]['x0'],
            'v0_1': trajetorias_sistema[0]['v0'],
            'x0_2': trajetorias_sistema[1]['x0'],
            'v0_2': trajetorias_sistema[1]['v0'],
            'omega': omega_sistema,
            'tempos_unicos': tempos_unicos,
            'dt': dt_interpolacao,
            'T': T,
            'traj1': trajetorias_sistema[0],
            'traj2': trajetorias_sistema[1]
        }
    
    trajetorias_unicas = pd.DataFrame(trajetorias_unicas_geradas)
    
    if len(trajetorias_unicas) < 2:
        print("  ERRO: Precisamos de pelo menos 2 trajetórias para interpolação")
        return pd.DataFrame()
    
    # ========================================================================================
    # SELECIONA 2 SISTEMAS DIFERENTES ALEATORIAMENTE
    # ========================================================================================
    
    # agrupa trajetórias por frequência
    trajetorias_por_frequencia = {}
    for idx, row in trajetorias_unicas.iterrows():
        omega = row['omega']
        if omega not in trajetorias_por_frequencia:
            trajetorias_por_frequencia[omega] = []
        trajetorias_por_frequencia[omega].append((idx, row))
    
    # filtra frequências que têm pelo menos 2 trajetórias com condições iniciais diferentes
    frequencias_validas = [omega for omega, trajs in trajetorias_por_frequencia.items() if len(trajs) >= 2]
    
    if len(frequencias_validas) < 2:
        print("  ERRO: Precisamos de pelo menos 2 frequências diferentes com pelo menos 2 trajetórias cada")
        return pd.DataFrame()
    
    # seleciona 2 frequências diferentes aleatoriamente
    frequencias_selecionadas = np.random.choice(frequencias_validas, 2, replace=False)
    
    sistemas_dados_selecionados = {}
    
    for sistema_idx, omega_sistema in enumerate(frequencias_selecionadas):
        
        trajetorias_mesma_freq = trajetorias_por_frequencia[omega_sistema]
        
        # seleciona 2 trajetórias diferentes com a MESMA frequência para este sistema
        indices_selecionados = np.random.choice(len(trajetorias_mesma_freq), 2, replace=False)
        traj1_info = trajetorias_mesma_freq[indices_selecionados[0]]
        traj2_info = trajetorias_mesma_freq[indices_selecionados[1]]
        traj1 = traj1_info[1]
        traj2 = traj2_info[1]
        
        x0_1, v0_1, omega_1 = traj1['x0'], traj1['v0'], traj1['omega']
        x0_2, v0_2, omega_2 = traj2['x0'], traj2['v0'], traj2['omega']
        
        # usa os tempos do sistema (todos têm o mesmo tamanho pois geramos com num_pontos_por_trajetoria)
        tempos_unicos_sistema = sistemas_dados[omega_sistema]['tempos_unicos']
        
        sistemas_dados_selecionados[omega_sistema] = {
            'sistema_id': sistema_idx + 1,
            'x0_1': x0_1,
            'v0_1': v0_1,
            'x0_2': x0_2,
            'v0_2': v0_2,
            'omega': omega_1,
            'tempos_unicos': tempos_unicos_sistema,
            'dt': sistemas_dados[omega_sistema]['dt'],
            'T': sistemas_dados[omega_sistema]['T'],
            'traj1_info': traj1_info,
            'traj2_info': traj2_info
        }
    
    alphas = np.linspace(0, 1, 4)

    todas_previsoes = []
    todos_reais_interpolados = []
    
    tempos_lista = []
    posicoes_previstas_lista = []
    velocidades_previstas_lista = []
    posicoes_reais_lista = []
    velocidades_reais_lista = []
    casos_info_lista = []
    dados_interpolados = []
    
    # processa cada sistema separadamente
    for sistema_idx, (omega_sistema, dados_sistema) in enumerate(sistemas_dados_selecionados.items()):
        
        x0_1 = dados_sistema['x0_1']
        v0_1 = dados_sistema['v0_1']
        x0_2 = dados_sistema['x0_2']
        v0_2 = dados_sistema['v0_2']
        omega = dados_sistema['omega']
        tempos_unicos = dados_sistema['tempos_unicos']
        
        for alpha in alphas:
            x0_interp = (1 - alpha) * x0_1 + alpha * x0_2
            v0_interp = (1 - alpha) * v0_1 + alpha * v0_2
            omega_interp = omega
            
            # cada alpha gera uma nova trajetória interpolada
            X_interpolado = np.zeros((len(tempos_unicos), 4))
            X_interpolado[:, 0] = x0_interp
            X_interpolado[:, 1] = v0_interp
            X_interpolado[:, 2] = omega_interp
            X_interpolado[:, 3] = tempos_unicos
            
            X_interpolado_scaled = scaler_X.transform(X_interpolado)
            X_tensor = torch.tensor(X_interpolado_scaled, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                pred_scaled = model(X_tensor).cpu().numpy()
            
            pred = scaler_y.inverse_transform(pred_scaled)
            
            # solução analítica com a frequência interpolada
            pos_analitico = x0_interp * np.cos(omega_interp * tempos_unicos) + \
                            (v0_interp / omega_interp) * np.sin(omega_interp * tempos_unicos)
            vel_analitico = -x0_interp * omega_interp * np.sin(omega_interp * tempos_unicos) + \
                            v0_interp * np.cos(omega_interp * tempos_unicos)
            
            todas_previsoes.append(pred)
            todos_reais_interpolados.append(np.column_stack([pos_analitico, vel_analitico]))
            
            tempos_lista.append(tempos_unicos)
            posicoes_previstas_lista.append(pred[:, 0])
            velocidades_previstas_lista.append(pred[:, 1])
            posicoes_reais_lista.append(pos_analitico)
            velocidades_reais_lista.append(vel_analitico)
            
            cor_idx = int(alpha * (len(CORES_PALETA) - 1))
            cor = CORES_PALETA[cor_idx % len(CORES_PALETA)]
            
            casos_info_lista.append({
                'sistema': f"Sistema_{dados_sistema['sistema_id']}_ω={omega:.2f}",
                'alpha': alpha,
                'x0': x0_interp,
                'v0': v0_interp,
                'omega': omega_interp,
                'cor': cor
            })
            
            for k in range(len(tempos_unicos)):
                dados_interpolados.append({
                    'sistema_id': dados_sistema['sistema_id'],
                    'frequencia_sistema': omega,
                    'alpha_interpolacao': alpha,
                    'x0_original_1': x0_1,
                    'v0_original_1': v0_1,
                    'omega_original_1': omega,
                    'x0_original_2': x0_2,
                    'v0_original_2': v0_2,
                    'omega_original_2': omega,
                    'x0_interpolado': x0_interp,
                    'v0_interpolado': v0_interp,
                    'omega_interpolado': omega_interp,
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
    
    print(f"\n  RMSE Posição (vs solução analítica): {rmse_pos:.6f} m")
    print(f"  RMSE Velocidade (vs solução analítica): {rmse_vel:.6f} m/s")
    print(f"  R² Posição (vs solução analítica): {r2_pos:.4f}")
    print(f"  R² Velocidade (vs solução analítica): {r2_vel:.4f}")
    
    # ============================================
    # GRÁFICO 1: Real vs Previsto
    # ============================================
    
    fig1 = cria_grafico_interpolacao_pontual_mlp(
        predictions=predictions_all,
        y_true=y_true_all,
        titulo="Interpolação entre Trajetórias: Solução Analítica vs MLP - Dados Gerados Aleatoriamente"
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
        titulo="Interpolação entre Trajetórias: MLP vs Solução Analítica - Espaço de Fases (Dados Gerados Aleatoriamente)"
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
        titulo="Interpolação entre Trajetórias: MLP vs Solução Analítica - Posição e Velocidade vs Tempo (Dados Gerados Aleatoriamente)"
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
    df_interpolado.attrs['num_sistemas'] = len(frequencias_selecionadas)
    df_interpolado.attrs['num_trajetorias_por_sistema'] = 2
    df_interpolado.attrs['num_alpha'] = len(alphas)
    df_interpolado.attrs['num_tempos'] = num_pontos_por_trajetoria
    df_interpolado.attrs['omega_min'] = omega_min
    df_interpolado.attrs['omega_max'] = omega_max
    df_interpolado.attrs['x0_min'] = x0_min
    df_interpolado.attrs['x0_max'] = x0_max
    df_interpolado.attrs['v0_min'] = v0_min
    df_interpolado.attrs['v0_max'] = v0_max

    # ========================================================================
    # GRÁFICO 4: Trajetórias Originais e Interpoladas no Espaço de Fases
    # ========================================================================
    
    # diretório específico para os gráficos de espaço de fases por sistema
    graficos_sistemas_dir = f"{output_dir}/espaco_fases_por_sistema"
    os.makedirs(graficos_sistemas_dir, exist_ok=True)
    
    # para cada sistema gera gráficos individuais
    for sistema_idx, (omega_sistema, dados_sistema) in enumerate(sistemas_dados_selecionados.items()):
        
        x0_1 = dados_sistema['x0_1']
        v0_1 = dados_sistema['v0_1']
        x0_2 = dados_sistema['x0_2']
        v0_2 = dados_sistema['v0_2']
        omega = dados_sistema['omega']
        tempos_unicos = dados_sistema['tempos_unicos']
        
        # gera trajetórias analiticamente para o gráfico
        pos_traj1 = x0_1 * np.cos(omega * tempos_unicos) + \
                    (v0_1 / omega) * np.sin(omega * tempos_unicos)
        vel_traj1 = -x0_1 * omega * np.sin(omega * tempos_unicos) + \
                    v0_1 * np.cos(omega * tempos_unicos)
        
        pos_traj2 = x0_2 * np.cos(omega * tempos_unicos) + \
                    (v0_2 / omega) * np.sin(omega * tempos_unicos)
        vel_traj2 = -x0_2 * omega * np.sin(omega * tempos_unicos) + \
                    v0_2 * np.cos(omega * tempos_unicos)
        
        # interpolações para o sistema
        interpolacoes_sistema = []
        alphas_unicos = np.sort(df_interpolado[(df_interpolado['sistema_id'] == dados_sistema['sistema_id'])]['alpha_interpolacao'].unique())
        
        for alpha in alphas_unicos:
            if alpha == 0 or alpha == 1:
                continue
            
            mask_alpha = (df_interpolado['sistema_id'] == dados_sistema['sistema_id']) & (df_interpolado['alpha_interpolacao'] == alpha)
            dados_alpha = df_interpolado[mask_alpha].sort_values('tempo')
            
            if len(dados_alpha) > 0:
                interpolacoes_sistema.append({
                    'alpha': alpha,
                    'posicoes': dados_alpha['posicao_prevista_mlp'].values,
                    'velocidades': dados_alpha['velocidade_prevista_mlp'].values,
                    'x0_interp': dados_alpha['x0_interpolado'].iloc[0],
                    'v0_interp': dados_alpha['v0_interpolado'].iloc[0],
                    'omega_interp': dados_alpha['omega_interpolado'].iloc[0]
                })
        
        casos_info_sistema = [{
            'x0_1': x0_1,
            'v0_1': v0_1,
            'omega_1': omega,
            'x0_2': x0_2,
            'v0_2': v0_2,
            'omega_2': omega
        }]
        
        fig_sistema = cria_grafico_interpolacao_entre_trajetorias_espaco_fases(
            trajetoria1_pos=pos_traj1,
            trajetoria1_vel=vel_traj1,
            trajetoria2_pos=pos_traj2,
            trajetoria2_vel=vel_traj2,
            interpolacoes_lista=interpolacoes_sistema,
            casos_info=casos_info_sistema,
            titulo=f"Interpolação entre Trajetórias - Sistema {dados_sistema['sistema_id']} (ω={omega:.3f} rad/s) - Dados Gerados Aleatoriamente"
        )
        
        nome_grafico = f"{graficos_sistemas_dir}/interpolacao_entre_trajetorias_espaco_fases_sistema_{dados_sistema['sistema_id']}_omega_{omega:.3f}.html"
        fig_sistema.write_html(nome_grafico)
        fig_sistema.show()

    return df_interpolado


def interpola_trajetorias_mlp_node(
    model: nn.Module,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Node: Usa o modelo treinado para gerar diferentes condições iniciais a partir de uma trajetória base.
    A partir de uma trajetória escolhida aleatoriamente, gera novas condições iniciais variando x0 e v0.
    Não mistura dados de treino/validação/teste pois usa dados gerados aleatoriamente.
    """
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_interpolacao_trajetorias = f"{output_dir}/interpolacoes_trajetorias_real_previsto_mlp.html"
    grafico_interpolacao_trajetorias_espaco_fases = f"{output_dir}/interpolacao_trajetorias_espaco_fases.html"
    grafico_interpolacao_trajetorias_temporal = f"{output_dir}/interpolacao_trajetorias_v_x_vs_t.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    intervals = parameters.get('intervals', {})
    
    x0_min = intervals.get('x0_min', -0.5)
    x0_max = intervals.get('x0_max', 0.5)
    v0_min = intervals.get('v0_min', -1.0)
    v0_max = intervals.get('v0_max', 1.0)
    omega_min = intervals.get('omega_min', 1.0)
    omega_max = intervals.get('omega_max', 5.0)
    
    seed = parameters.get('seed', 42)
    np.random.seed(seed)
    
    print("\n=== GERAÇÃO DE CONDIÇÕES INICIAIS A PARTIR DE TRAJETÓRIA BASE ===")
    print("\n  Gera novas condições iniciais variando x0, v0 e ω dentro dos limites de treino do modelo")
    
    # ============================================
    # GERAÇÃO DE DADOS ALEATÓRIOS
    # ============================================
    
    # número de sistemas diferentes
    num_sistemas = 2
    num_variacoes = 2  # número de novas condições iniciais por sistema
    num_pontos_por_trajetoria = 1000
        
    # gera frequências diferentes para cada sistema
    frequencias_sistemas = np.random.uniform(omega_min, omega_max, num_sistemas)
    while len(np.unique(frequencias_sistemas)) < num_sistemas:
        frequencias_sistemas = np.random.uniform(omega_min, omega_max, num_sistemas)
    
    # estrutura para armazenar os sistemas
    sistemas_dados = {}
    trajetorias_unicas_geradas = []
    X_test_original_gerado = []
    y_test_original_gerado = []
    
    for sistema_idx, omega_sistema in enumerate(frequencias_sistemas):
        
        # calcula o período do sistema
        T = 2 * np.pi / omega_sistema
        tempo_maximo = T  # 1 período completo
        
        # calcula o passo temporal
        dt_interpolacao = tempo_maximo / (num_pontos_por_trajetoria - 1)
                
        # gera tempos para este sistema
        tempos_unicos = np.linspace(0, tempo_maximo, num_pontos_por_trajetoria)
        
        # gera uma trajetória base aleatória para este sistema
        x0_base = np.random.uniform(x0_min, x0_max)
        v0_base = np.random.uniform(v0_min, v0_max)
        amplitude_base = np.sqrt(x0_base**2 + (v0_base / omega_sistema)**2)
                
        # gera novas condições iniciais para este sistema
        np.random.seed(seed + sistema_idx)
        x0_variacoes = np.random.uniform(x0_min, x0_max, num_variacoes)
        v0_variacoes = np.random.uniform(v0_min, v0_max, num_variacoes)
        
        variacoes = []
        for i in range(num_variacoes):
            amplitude = np.sqrt(x0_variacoes[i]**2 + (v0_variacoes[i] / omega_sistema)**2)
            variacoes.append({
                'x0': x0_variacoes[i],
                'v0': v0_variacoes[i],
                'amplitude': amplitude
            })
        
        trajetorias_unicas_geradas.append({
            'x0': x0_base,
            'v0': v0_base,
            'omega': omega_sistema
        })
        
        for t in tempos_unicos:
            X_test_original_gerado.append([x0_base, v0_base, omega_sistema, t])
            pos = x0_base * np.cos(omega_sistema * t) + (v0_base / omega_sistema) * np.sin(omega_sistema * t)
            vel = -x0_base * omega_sistema * np.sin(omega_sistema * t) + v0_base * np.cos(omega_sistema * t)
            y_test_original_gerado.append([pos, vel])
        
        # armazena dados deste sistema
        sistemas_dados[omega_sistema] = {
            'sistema_id': sistema_idx + 1,
            'omega': omega_sistema,
            'x0_base': x0_base,
            'v0_base': v0_base,
            'amplitude_base': amplitude_base,
            'tempos_unicos': tempos_unicos,
            'variacoes': variacoes,
            'T': T,
            'dt': dt_interpolacao
        }
        
    trajetorias_unicas = pd.DataFrame(trajetorias_unicas_geradas)
    
    if len(trajetorias_unicas) < 2:
        print("  ERRO: Precisamos de pelo menos 2 sistemas diferentes para interpolação")
        return pd.DataFrame()
    
    # ========================================================================
    # SELECIONA SISTEMAS DIFERENTES
    # ========================================================================
    
    # agrupa trajetórias por frequência
    trajetorias_por_frequencia = {}
    for idx, row in trajetorias_unicas.iterrows():
        omega = row['omega']
        if omega not in trajetorias_por_frequencia:
            trajetorias_por_frequencia[omega] = []
        trajetorias_por_frequencia[omega].append((idx, row))
    
    # obtém todas as frequências disponíveis
    frequencias_disponiveis = list(trajetorias_por_frequencia.keys())
    
    if len(frequencias_disponiveis) < 2:
        print("  ERRO: Precisamos de pelo menos 2 frequências diferentes")
        return pd.DataFrame()
    
    # seleciona 2 frequências diferentes aleatoriamente
    frequencias_selecionadas = np.random.choice(frequencias_disponiveis, 2, replace=False)
        
    sistemas_dados_selecionados = {}
    
    for sistema_idx, omega_sistema in enumerate(frequencias_selecionadas):
        print(f"\n  ============================================================")
        print(f"  Sistema {sistema_idx + 1}: ω = {omega_sistema:.3f} rad/s")
        print(f"  ============================================================")
        
        trajetorias_sistema = trajetorias_por_frequencia[omega_sistema]
        
        # seleciona uma trajetória base aleatoriamente para este sistema
        idx_base = np.random.choice(len(trajetorias_sistema))
        traj_base_info = trajetorias_sistema[idx_base]
        traj_base = traj_base_info[1]
        x0_base, v0_base = traj_base['x0'], traj_base['v0']
        amplitude_base = np.sqrt(x0_base**2 + (v0_base / omega_sistema)**2)
        
        print(f"\n  Trajetória Base Selecionada:")
        print(f"    x0 = {x0_base:.3f} m")
        print(f"    v0 = {v0_base:.3f} m/s")
        
        # obtém os tempos únicos da trajetória base
        tempos_unicos = sistemas_dados[omega_sistema]['tempos_unicos']
        
        print(f"    Período: {tempos_unicos.max():.3f} s")
        print(f"    Passo temporal: {sistemas_dados[omega_sistema]['dt']:.6f} s")
        
        np.random.seed(seed + sistema_idx)
        x0_variacoes = np.random.uniform(x0_min, x0_max, num_variacoes)
        v0_variacoes = np.random.uniform(v0_min, v0_max, num_variacoes)
        
        variacoes = []
        for i in range(num_variacoes):
            amplitude = np.sqrt(x0_variacoes[i]**2 + (v0_variacoes[i] / omega_sistema)**2)
            variacoes.append({
                'x0': x0_variacoes[i],
                'v0': v0_variacoes[i],
                'amplitude': amplitude
            })
            print(f"    Interpolação {i+1}: x0={x0_variacoes[i]:.3f} m, v0={v0_variacoes[i]:.3f} m/s")
        
        sistemas_dados_selecionados[omega_sistema] = {
            'sistema_id': sistema_idx + 1,
            'omega': omega_sistema,
            'x0_base': x0_base,
            'v0_base': v0_base,
            'amplitude_base': amplitude_base,
            'tempos_unicos': tempos_unicos,
            'variacoes': variacoes,
            'T': sistemas_dados[omega_sistema]['T'],
            'dt': sistemas_dados[omega_sistema]['dt']
        }
    
    # ========================================================================
    # PROCESSAMENTO DAS PREVISÕES PARA CADA SISTEMA
    # ========================================================================
    
    todas_previsoes = []
    todos_reais_interpolados = []
    
    tempos_lista = []
    posicoes_previstas_lista = []
    velocidades_previstas_lista = []
    posicoes_reais_lista = []
    velocidades_reais_lista = []
    casos_info_lista = []
    dados_interpolados = []
    
    # processa cada sistema separadamente
    for sistema_idx, (omega_sistema, dados_sistema) in enumerate(sistemas_dados_selecionados.items()):
        
        tempos_unicos = dados_sistema['tempos_unicos']
        variacoes = dados_sistema['variacoes']
        omega = dados_sistema['omega']
        
        for var_idx, var in enumerate(variacoes):
            x0_novo = var['x0']
            v0_novo = var['v0']
                        
            # prepara entrada para o modelo com 4 features (x0, v0, omega, tempo)
            X_novo = np.zeros((len(tempos_unicos), 4))
            X_novo[:, 0] = x0_novo
            X_novo[:, 1] = v0_novo
            X_novo[:, 2] = omega  # inclui a frequência do sistema
            X_novo[:, 3] = tempos_unicos
            
            # normaliza e faz previsão
            X_novo_scaled = scaler_X.transform(X_novo)
            X_tensor = torch.tensor(X_novo_scaled, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                pred_scaled = model(X_tensor).cpu().numpy()
            
            pred = scaler_y.inverse_transform(pred_scaled)
            
            # solução analítica para validação
            pos_analitico = x0_novo * np.cos(omega * tempos_unicos) + \
                            (v0_novo / omega) * np.sin(omega * tempos_unicos)
            vel_analitico = -x0_novo * omega * np.sin(omega * tempos_unicos) + \
                            v0_novo * np.cos(omega * tempos_unicos)
            
            todas_previsoes.append(pred)
            todos_reais_interpolados.append(np.column_stack([pos_analitico, vel_analitico]))
            
            tempos_lista.append(tempos_unicos)
            posicoes_previstas_lista.append(pred[:, 0])
            velocidades_previstas_lista.append(pred[:, 1])
            posicoes_reais_lista.append(pos_analitico)
            velocidades_reais_lista.append(vel_analitico)
            
            cor_idx = (sistema_idx * num_variacoes + var_idx) % len(CORES_PALETA)
            cor = CORES_PALETA[cor_idx]
            
            casos_info_lista.append({
                'sistema_id': dados_sistema['sistema_id'],
                'omega': omega,
                'x0': x0_novo,
                'v0': v0_novo,
                'cor': cor,
                'variation_id': var_idx
            })
            
            for k in range(len(tempos_unicos)):
                dados_interpolados.append({
                    'sistema_id': dados_sistema['sistema_id'],
                    'frequencia': omega,
                    'variacao_id': var_idx,
                    'x0': x0_novo,
                    'v0': v0_novo,
                    'omega': omega,
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
        print("  ERRO: Nenhuma previsão realizada")
        return pd.DataFrame()
    
    predictions_all = np.vstack(todas_previsoes)
    y_true_all = np.vstack(todos_reais_interpolados)
    
    rmse_pos = float(np.sqrt(mean_squared_error(y_true_all[:, 0], predictions_all[:, 0])))
    rmse_vel = float(np.sqrt(mean_squared_error(y_true_all[:, 1], predictions_all[:, 1])))
    r2_pos = float(r2_score(y_true_all[:, 0], predictions_all[:, 0]))
    r2_vel = float(r2_score(y_true_all[:, 1], predictions_all[:, 1]))
    
    print(f"\n  RMSE Posição (vs solução analítica): {rmse_pos:.6f} m")
    print(f"  RMSE Velocidade (vs solução analítica): {rmse_vel:.6f} m/s")
    print(f"  R² Posição (vs solução analítica): {r2_pos:.4f}")
    print(f"  R² Velocidade (vs solução analítica): {r2_vel:.4f}")
    
    # ============================================
    # GRÁFICO 1: Real vs Previsto
    # ============================================
    
    fig1 = cria_grafico_interpolacao_pontual_mlp(
        predictions=predictions_all,
        y_true=y_true_all,
        titulo="Novas Condições Iniciais: Solução Analítica vs MLP - Dados Gerados Aleatoriamente"
    )
    
    fig1.write_html(grafico_interpolacao_trajetorias)
    
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
        titulo="Novas Condições Iniciais: MLP vs Solução Analítica - Espaço de Fases (Dados Gerados Aleatoriamente)"
    )
    
    fig2.write_html(grafico_interpolacao_trajetorias_espaco_fases)
    
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
        titulo="Novas Condições Iniciais: MLP vs Solução Analítica - Posição e Velocidade vs Tempo (Dados Gerados Aleatoriamente)"
    )
    
    fig3.write_html(grafico_interpolacao_trajetorias_temporal)
    
    fig1.show()
    fig2.show()
    fig3.show()
    
    df_interpolado = pd.DataFrame(dados_interpolados)
    
    df_interpolado.attrs['rmse_posicao'] = rmse_pos
    df_interpolado.attrs['rmse_velocidade'] = rmse_vel
    df_interpolado.attrs['r2_posicao'] = r2_pos
    df_interpolado.attrs['r2_velocidade'] = r2_vel
    df_interpolado.attrs['total_pontos'] = len(predictions_all)
    df_interpolado.attrs['num_sistemas'] = len(frequencias_selecionadas)
    df_interpolado.attrs['num_variacoes_por_sistema'] = num_variacoes
    df_interpolado.attrs['num_tempos_por_sistema'] = num_pontos_por_trajetoria
    df_interpolado.attrs['omega_min'] = omega_min
    df_interpolado.attrs['omega_max'] = omega_max
    df_interpolado.attrs['x0_min'] = x0_min
    df_interpolado.attrs['x0_max'] = x0_max
    df_interpolado.attrs['v0_min'] = v0_min
    df_interpolado.attrs['v0_max'] = v0_max
    
    # ========================================================================
    # GRÁFICO 4: Trajetória Base e Novas Condições Iniciais no Espaço de Fases
    # ========================================================================
    
    graficos_sistemas_dir = f"{output_dir}/espaco_fases_trajetorias_por_sistema"
    os.makedirs(graficos_sistemas_dir, exist_ok=True)
    
    # para cada sistema gera gráficos individuais
    for omega_sistema, dados_sistema in sistemas_dados_selecionados.items():
        
        x0_base = dados_sistema['x0_base']
        v0_base = dados_sistema['v0_base']
        omega = dados_sistema['omega']
        tempos_unicos = dados_sistema['tempos_unicos']
        
        # gera trajetória base analiticamente
        pos_base = x0_base * np.cos(omega * tempos_unicos) + \
                   (v0_base / omega) * np.sin(omega * tempos_unicos)
        vel_base = -x0_base * omega * np.sin(omega * tempos_unicos) + \
                   v0_base * np.cos(omega * tempos_unicos)
        
        # novas trajetórias para este sistema
        novas_trajetorias_sistema = []
        
        for var_idx in range(num_variacoes):
            mask_var = (df_interpolado['sistema_id'] == dados_sistema['sistema_id']) & (df_interpolado['variacao_id'] == var_idx)
            dados_var = df_interpolado[mask_var].sort_values('tempo')
            
            if len(dados_var) > 0:
                x0_var = dados_var['x0'].iloc[0]
                v0_var = dados_var['v0'].iloc[0]
                
                novas_trajetorias_sistema.append({
                    'variacao_id': var_idx,
                    'posicoes': dados_var['posicao_prevista_mlp'].values,
                    'velocidades': dados_var['velocidade_prevista_mlp'].values,
                    'x0': x0_var,
                    'v0': v0_var
                })
        
        casos_info_sistema = {
            'x0_base': x0_base,
            'v0_base': v0_base
        }
        
        fig_sistema = cria_grafico_interpolacao_trajetorias_espaco_fases(
            trajetoria_base_pos=pos_base,
            trajetoria_base_vel=vel_base,
            novas_trajetorias_lista=novas_trajetorias_sistema,
            casos_info=casos_info_sistema,
            titulo=f"Sistema {dados_sistema['sistema_id']} (ω={omega:.3f} rad/s): Trajetória Base vs Novas Condições Iniciais - Dados Gerados Aleatoriamente"
        )
        
        nome_grafico = f"{graficos_sistemas_dir}/trajetoria_base_vs_novas_condicoes_sistema_{dados_sistema['sistema_id']}_omega_{omega:.3f}.html"
        fig_sistema.write_html(nome_grafico)
        fig_sistema.show()

    return df_interpolado