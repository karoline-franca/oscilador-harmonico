"""
Nodes do pipeline MLP para previsão de trajetórias completas do oscilador de FitzHugh-Nagumo.
Entrada: [v0, w0] (potencial, recuperação)
Saída: Trajetória completa [v_0, w_0, v_1, w_1, ..., v_N, w_N]
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
from scipy.interpolate import interp1d
from typing import Dict, Any, Tuple
from .model import MLP
from oscilador_fitz_hugh_nagumo.pipelines.p00_data_generating.fhn import OsciladorFitzHughNagumo
from oscilador_fitz_hugh_nagumo.utils import (
    CORES_PALETA,
    cria_grafico_distribuicao_amplitudes,
    cria_grafico_distribuicao_dados,
    cria_grafico_historico_treinamento,
    cria_grafico_real_previsto_mlp,
    cria_grafico_previsoes_espaco_fases,
    cria_grafico_interpolacao_completo,
    cria_grafico_interpolacao_espaco_fases,
    cria_grafico_interpolacao_pontual_mlp,
    cria_grafico_interpolacao_pontual_espaco_fases,
    cria_grafico_interpolacao_pontual_completo,
    cria_grafico_interpolacao_entre_trajetorias_espaco_fases,
    cria_grafico_interpolacao_trajetorias_espaco_fases,
)


def fixa_sementes(seed: int = 42):
    """Fixa todas as sementes para reprodutibilidade."""
    
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepara_dados_mlp_node(base_oscilador: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple:
    """
    Prepara os dados para treinamento do MLP para o oscilador de FitzHugh-Nagumo.
    
    Entrada: [v0, w0] (potencial, recuperação)
    Saída: Trajetória completa [v_0, w_0, v_1, w_1, ..., v_N, w_N]
    
    O tempo é usado apenas para organizar os pontos da trajetória,
    mas não é uma feature de entrada.
    """

    seed = parameters.get('seed', 42)
    fixa_sementes(seed)
    
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
    
    if base_oscilador[['v0', 'w0']].isnull().any().any():
        print("  AVISO: Valores NaN detectados nas colunas numéricas!")
        base_oscilador = base_oscilador.dropna(subset=['v0', 'w0'])
    
    # parâmetros do sistema - ponto de equilíbrio (calculado a partir das equações)
    if 'potencial_eq' in base_oscilador.columns and 'recuperacao_eq' in base_oscilador.columns:
        v_eq = base_oscilador['potencial_eq'].iloc[0] if len(base_oscilador) > 0 else 0.0
        w_eq = base_oscilador['recuperacao_eq'].iloc[0] if len(base_oscilador) > 0 else 0.0
    else:
        v_eq = 0.0  # valor padrão
        w_eq = 0.0  # valor padrão
    
    # parâmetros do sistema - epsilon, a, b, I
    if 'parametro_epsilon' in base_oscilador.columns:
        epsilon = base_oscilador['parametro_epsilon'].iloc[0] if len(base_oscilador) > 0 else 0.08
    else:
        epsilon = 0.08
    
    if 'parametro_a' in base_oscilador.columns:
        a = base_oscilador['parametro_a'].iloc[0] if len(base_oscilador) > 0 else 0.7
    else:
        a = 0.7
    
    if 'parametro_b' in base_oscilador.columns:
        b = base_oscilador['parametro_b'].iloc[0] if len(base_oscilador) > 0 else 0.8
    else:
        b = 0.8
    
    if 'parametro_I' in base_oscilador.columns:
        I = base_oscilador['parametro_I'].iloc[0] if len(base_oscilador) > 0 else 0.5
    else:
        I = 0.5

    if 'parametro_R' in base_oscilador.columns:
        R = base_oscilador['parametro_R'].iloc[0] if len(base_oscilador) > 0 else 0.1
    else:
        R = 0.1

    print(f"\n=== BASE DE DADOS ===")
    print(f"  Parâmetros do sistema: ε={epsilon:.4f}, a={a:.2f}, b={b:.2f}, I={I:.2f}, R={R:.2f}")
    print(f"  Ponto de equilíbrio: v*={v_eq:.3f}, w*={w_eq:.3f}")
    print(f"  Total de linhas da base: {len(base_oscilador)}")
    
    if 'id_trajetoria' in base_oscilador.columns:
        trajetorias_unicas = base_oscilador['id_trajetoria'].unique()
        print(f"  Total de trajetórias únicas: {len(trajetorias_unicas)}")
        
        if len(trajetorias_unicas) == 1 and 'nan' in str(trajetorias_unicas[0]).lower():
            print("  AVISO: id_trajetoria ainda com problemas. Recriando baseado em v0, w0...")
            base_oscilador['id_trajetoria'] = 'v0_' + base_oscilador['v0'].round(6).astype(str) + \
                                              '_w0_' + base_oscilador['w0'].round(6).astype(str)
            trajetorias_unicas = base_oscilador['id_trajetoria'].unique()
            print(f"  Nova contagem de trajetórias: {len(trajetorias_unicas)}")
    else:
        print("  ERRO: Coluna 'id_trajetoria' não encontrada!")
        print("  Criando id_trajetoria baseado em v0, w0")
        base_oscilador['id_trajetoria'] = 'v0_' + base_oscilador['v0'].round(6).astype(str) + \
                                          '_w0_' + base_oscilador['w0'].round(6).astype(str)
        trajetorias_unicas = base_oscilador['id_trajetoria'].unique()
        print(f"  Total de trajetórias criadas: {len(trajetorias_unicas)}")
    
    # obtém o número de pontos por trajetória a partir dos dados
    primeiro_grupo = base_oscilador[base_oscilador['id_trajetoria'] == trajetorias_unicas[0]].sort_values('tempo')
    num_timesteps = len(primeiro_grupo)
    
    print(f"\n=== PREPARAÇÃO DOS DADOS ===")
    
    X_list = []  # [v0, w0] para cada trajetória
    y_list = []  # trajetória completa intercalada para cada trajetória
    tempos_list = []  # tempos para referência
    
    for traj_id in trajetorias_unicas:
        grupo = base_oscilador[base_oscilador['id_trajetoria'] == traj_id].sort_values('tempo')
        
        if len(grupo) != num_timesteps:
            print(f"  AVISO: Trajetória {traj_id} tem {len(grupo)} pontos, pulando...")
            continue
        
        # entrada: [v0, w0] - potencial e recuperação iniciais
        v0 = grupo['v0'].iloc[0]
        w0 = grupo['w0'].iloc[0]
        X_list.append([v0, w0])
        
        # saída: trajetória completa intercalada [v0, w0, v1, w1, ..., vN, wN]
        potencial = grupo['potencial'].values
        recuperacao = grupo['recuperacao'].values
        trajetoria = np.column_stack([potencial, recuperacao]).flatten()
        y_list.append(trajetoria)
        
        # tempos para referência
        tempos_list.append(grupo['tempo'].values)
    
    X_raw = np.array(X_list, dtype=np.float32)
    y_raw = np.array(y_list, dtype=np.float32)
    tempos_referencia = np.array(tempos_list[0]) if tempos_list else np.array([])
    
    print(f"\n  Trajetórias válidas: {len(X_raw)}")
    print(f"  Dimensão entrada: {X_raw.shape[1]} (v0, w0)")
    print(f"  Dimensão saída: {y_raw.shape[1]} (2N)")
    print(f"  Nós de saída do modelo por trajetória: {num_timesteps}")
    
    # treino, validação e teste (70/20/10)
    n_trajetorias = len(X_raw)
    indices = np.random.permutation(n_trajetorias)
    n_train = int(0.7 * n_trajetorias)
    n_val = int(0.2 * n_trajetorias)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    X_train = X_raw[train_indices]
    y_train = y_raw[train_indices]
    X_val = X_raw[val_indices]
    y_val = y_raw[val_indices]
    X_test = X_raw[test_indices]
    y_test = y_raw[test_indices]
    
    trajetorias_train = np.array(trajetorias_unicas)[train_indices]
    trajetorias_val = np.array(trajetorias_unicas)[val_indices]
    trajetorias_test = np.array(trajetorias_unicas)[test_indices]
    
    print(f"\n  Trajetórias de treino: {len(X_train)}")
    print(f"  Trajetórias de validação: {len(X_val)}")
    print(f"  Trajetórias de teste: {len(X_test)}")
    print(f"  Amostras totais de treino: {len(X_train) * num_timesteps}")
    print(f"  Amostras totais de validação: {len(X_val) * num_timesteps}")
    print(f"  Amostras totais de teste: {len(X_test) * num_timesteps}")
    
    # normalização das variáveis de entrada e saída
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled_train = scaler_X.fit_transform(X_train)
    X_scaled_val = scaler_X.transform(X_val)
    X_scaled_test = scaler_X.transform(X_test)
    
    y_scaled_train = scaler_y.fit_transform(y_train)
    y_scaled_val = scaler_y.transform(y_val)
    y_scaled_test = scaler_y.transform(y_test)
    
    input_dim = X_raw.shape[1]
    output_dim = y_raw.shape[1]
    
    print(f"\n  Dimensão entrada: {input_dim} (v0, w0)")
    print(f"  Dimensão saída: {output_dim} (2N)")
    
    return (X_scaled_train, y_scaled_train, 
            X_scaled_val, y_scaled_val, 
            X_scaled_test, y_scaled_test, 
            input_dim, output_dim, scaler_X, scaler_y,
            trajetorias_train, trajetorias_val, trajetorias_test,
            num_timesteps, tempos_referencia)


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
    grafico_distribuicao_amplitudes = f"{output_dir}/distribuicao_amplitudes.html"
    
    base_oscilador = base_oscilador[base_oscilador['sistema_id'] == 0].copy()
    
    for col in base_oscilador.columns:
        if base_oscilador[col].dtype == 'object':
            try:
                base_oscilador[col] = base_oscilador[col].astype(str).str.replace(',', '.').astype(float)
            except:
                pass
    
    # obtém lista única de trajetórias
    trajetorias_unicas = base_oscilador['id_trajetoria'].unique()
    
    # ============================================
    # CÁLCULO DAS AMPLITUDES PARA VISUALIZAÇÃO
    # ============================================
    
    if 'potencial_eq' in base_oscilador.columns and 'recuperacao_eq' in base_oscilador.columns:
        potencial_eq = base_oscilador['potencial_eq'].iloc[0] if len(base_oscilador) > 0 else 0.0
        recuperacao_eq = base_oscilador['recuperacao_eq'].iloc[0] if len(base_oscilador) > 0 else 0.0
        
        amplitudes = {}
        for traj_id in trajetorias_unicas:
            grupo = base_oscilador[base_oscilador['id_trajetoria'] == traj_id].iloc[0]
            v0 = grupo['v0']
            w0 = grupo['w0']
            # distância euclidiana do ponto de equilíbrio
            amplitude = np.sqrt((v0 - potencial_eq)**2 + (w0 - recuperacao_eq)**2)
            amplitudes[traj_id] = amplitude
    else:
        amplitudes = {}
        for traj_id in trajetorias_unicas:
            grupo = base_oscilador[base_oscilador['id_trajetoria'] == traj_id]
            potencial = grupo['potencial'].values
            recuperacao = grupo['recuperacao'].values
            amp_potencial = np.max(np.abs(potencial))
            amp_recuperacao = np.max(np.abs(recuperacao))
            amplitude = np.sqrt(amp_potencial**2 + amp_recuperacao**2)
            amplitudes[traj_id] = amplitude
    
    # ordena de forma ascendente as trajetórias por amplitude
    trajetorias_ordenadas = sorted(amplitudes.items(), key=lambda x: x[1])
    amplitudes_ordenadas = [t[1] for t in trajetorias_ordenadas]
    
    print(f"\n=== DISTRIBUIÇÃO DAS TRAJETÓRIAS POR AMPLITUDE ===")
    print(f"  Amplitude mínima: {amplitudes_ordenadas[0]:.4f}")
    print(f"  Amplitude máxima: {amplitudes_ordenadas[-1]:.4f}")
    print(f"  Amplitude mediana: {amplitudes_ordenadas[len(amplitudes_ordenadas)//2]:.4f}")
    
    # ============================================
    # GRÁFICO: Distribuição das Amplitudes
    # ============================================
    
    fig_amp = cria_grafico_distribuicao_amplitudes(
        amplitudes=np.array(amplitudes_ordenadas),
        amplitude_limite_internas=None,
        titulo="Distribuição das Amplitudes das Trajetórias - Oscilador de FitzHugh-Nagumo"
    )
    
    fig_amp.write_html(grafico_distribuicao_amplitudes)
    fig_amp.show()
    
    # ============================================
    # DIVISÃO DOS DADOS
    # ============================================
    
    # treino, validação e teste (70/20/10)
    trajetorias_train, trajetorias_temp = train_test_split(
        trajetorias_unicas, test_size=0.30, random_state=42
    )
    trajetorias_val, trajetorias_test = train_test_split(
        trajetorias_temp, test_size=0.3333, random_state=42
    )
    
    print(f"\n=== DIVISÃO DOS DADOS ===")
    print(f"  Trajetórias de treino: {len(trajetorias_train)}")
    print(f"  Trajetórias de validação: {len(trajetorias_val)}")
    print(f"  Trajetórias de teste: {len(trajetorias_test)}")
        
    dados_train = base_oscilador[base_oscilador['id_trajetoria'].isin(trajetorias_train)]
    dados_val = base_oscilador[base_oscilador['id_trajetoria'].isin(trajetorias_val)]
    dados_test = base_oscilador[base_oscilador['id_trajetoria'].isin(trajetorias_test)]
    
    y_potencial_train = dados_train['potencial'].values.astype(np.float32).reshape(-1, 1)
    y_recuperacao_train = dados_train['recuperacao'].values.astype(np.float32).reshape(-1, 1)
    y_potencial_val = dados_val['potencial'].values.astype(np.float32).reshape(-1, 1)
    y_recuperacao_val = dados_val['recuperacao'].values.astype(np.float32).reshape(-1, 1)
    y_potencial_test = dados_test['potencial'].values.astype(np.float32).reshape(-1, 1)
    y_recuperacao_test = dados_test['recuperacao'].values.astype(np.float32).reshape(-1, 1)
    
    # ============================================
    # GRÁFICO: Distribuição no Espaço de Fases
    # ============================================
    
    fig = cria_grafico_distribuicao_dados(
        y_potencial_train=y_potencial_train,
        y_recuperacao_train=y_recuperacao_train,
        y_potencial_val=y_potencial_val,
        y_recuperacao_val=y_recuperacao_val,
        y_potencial_test=y_potencial_test,
        y_recuperacao_test=y_recuperacao_test,
        titulo="Distribuição dos Dados no Espaço de Fases - Oscilador de FitzHugh-Nagumo"
    )
    
    fig.write_html(grafico_distribuicao_dados) 
    fig.show()
    
    return None


def cria_modelo_mlp_node(input_dim: int, output_dim: int, parameters: Dict[str, Any]) -> nn.Module:
    """Cria o modelo MLP para previsão de trajetórias completas do oscilador de FitzHugh-Nagumo."""

    mlp_config = parameters.get('mlp', {})
    seed = parameters.get('seed', 42)
    
    fixa_sementes(seed)
    
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
    print(f"  Dimensão entrada: {input_dim} (v0, w0)")
    print(f"  Camadas ocultas: {hidden_dims}")
    print(f"  Dimensão saída: {output_dim} (2N)")
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
    """Treina o modelo MLP para prever trajetórias completas do oscilador de FitzHugh-Nagumo."""

    mlp_config = parameters.get('mlp', {})
    seed = parameters.get('seed', 42)
    
    batch_size = mlp_config.get('batch_size', 512)
    epochs = mlp_config.get('epochs', 500)
    learning_rate = mlp_config.get('learning_rate', 0.005)
    weight_decay = 0.0001
        
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_historico_loss = f"{output_dir}/historico_treinamento_loss.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    model = model.to(device)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        generator=generator
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    print("\n=== INICIANDO TREINAMENTO DO MLP ===")
    print(f"  Entrada: (v0, w0) -> Saída: (trajetória completa potencial/recuperação)")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Função loss: RMSE (Root Mean Squared Error)")
    print(f"  Seed: {seed}")
    
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
    
    # ============================================
    # GRÁFICO: Histórico de Treinamento
    # ============================================
    
    fig = cria_grafico_historico_treinamento(
        history=history,
        titulo="Evolução da Função de Custo durante o Treinamento do MLP - Oscilador de FitzHugh-Nagumo"
    )
    
    fig.write_html(grafico_historico_loss)
    
    print(f"\n=== TREINAMENTO CONCLUÍDO ===")
    print(f"  Loss final de treino: {history['train_loss'][-1]:.6f}")
    print(f"  Loss final de validação: {history['val_loss'][-1]:.6f}")
    
    fig.show()
    
    return model, history

def avalia_metricas_mlp_node(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler,
) -> Dict[str, float]:
    """
    Avalia o modelo MLP nos dados de validação e teste para o oscilador de FitzHugh-Nagumo.
    
    Avalia tanto a trajetória completa quanto cada ponto individualmente.
    """

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
    
    # separa potencial e recuperação das trajetórias
    potencial_pred_val = predictions_val[:, 0::2]
    recuperacao_pred_val = predictions_val[:, 1::2]
    potencial_true_val = y_val_original[:, 0::2]
    recuperacao_true_val = y_val_original[:, 1::2]
    
    potencial_pred_test = predictions_test[:, 0::2]
    recuperacao_pred_test = predictions_test[:, 1::2]
    potencial_true_test = y_test_original[:, 0::2]
    recuperacao_true_test = y_test_original[:, 1::2]
    
    # avalia ponto a ponto
    rmse_potencial_val = float(np.sqrt(mean_squared_error(potencial_true_val.flatten(), potencial_pred_val.flatten())))
    rmse_recuperacao_val = float(np.sqrt(mean_squared_error(recuperacao_true_val.flatten(), recuperacao_pred_val.flatten())))
    r2_potencial_val = float(r2_score(potencial_true_val.flatten(), potencial_pred_val.flatten()))
    r2_recuperacao_val = float(r2_score(recuperacao_true_val.flatten(), recuperacao_pred_val.flatten()))
    
    rmse_potencial_test = float(np.sqrt(mean_squared_error(potencial_true_test.flatten(), potencial_pred_test.flatten())))
    rmse_recuperacao_test = float(np.sqrt(mean_squared_error(recuperacao_true_test.flatten(), recuperacao_pred_test.flatten())))
    r2_potencial_test = float(r2_score(potencial_true_test.flatten(), potencial_pred_test.flatten()))
    r2_recuperacao_test = float(r2_score(recuperacao_true_test.flatten(), recuperacao_pred_test.flatten()))

    metrics = {
        'rmse_potencial_val': rmse_potencial_val,
        'rmse_recuperacao_val': rmse_recuperacao_val,
        'r2_potencial_val': r2_potencial_val,
        'r2_recuperacao_val': r2_recuperacao_val,
        'rmse_potencial_test': rmse_potencial_test,
        'rmse_recuperacao_test': rmse_recuperacao_test,
        'r2_potencial_test': r2_potencial_test,
        'r2_recuperacao_test': r2_recuperacao_test,
    }
    
    print("\n=== AVALIAÇÃO DO MODELO MLP - OSCILADOR DE FITZHUGH-NAGUMO ===")
    print(f"  RMSE Potencial Validação: {rmse_potencial_val:.6f}")
    print(f"  RMSE Recuperação Validação: {rmse_recuperacao_val:.6f}")
    print(f"  R² Potencial Validação: {r2_potencial_val:.4f}")
    print(f"  R² Recuperação Validação: {r2_recuperacao_val:.4f}")
    print(f"  RMSE Potencial Teste: {rmse_potencial_test:.6f}")
    print(f"  RMSE Recuperação Teste: {rmse_recuperacao_test:.6f}")
    print(f"  R² Potencial Teste: {r2_potencial_test:.4f}")
    print(f"  R² Recuperação Teste: {r2_recuperacao_test:.4f}")

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
    Mostra trajetórias completas no espaço de fases.
    
    Args:
        model: Modelo treinado
        X_test: Dados de teste (condições iniciais)
        y_test: Targets de teste (trajetórias completas)
        scaler_y: Scaler dos targets
        parameters: Parâmetros do pipeline
        tempos_referencia: Array com os tempos para plotagem
    """
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    seed = parameters.get('seed', 42)
    
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
    
    rng = np.random.RandomState(seed)
    
    # separa algumas trajetórias para visualização
    num_trajetorias_vis = min(5, len(predictions))
    indices_vis = rng.choice(len(predictions), num_trajetorias_vis, replace=False)
    
    # dados para visualização ponto a ponto
    predictions_flat = []
    y_true_flat = []
    
    for idx in indices_vis:
        # para cada trajetória, extrai pontos individuais
        pred_traj = predictions[idx]
        true_traj = y_test_original[idx]
        
        # intercala potencial e recuperação (v0, w0, v1, w1, ...)
        pred_potencial = pred_traj[0::2]
        pred_recuperacao = pred_traj[1::2]
        true_potencial = true_traj[0::2]
        true_recuperacao = true_traj[1::2]
        
        for i in range(len(pred_potencial)):
            predictions_flat.append([pred_potencial[i], pred_recuperacao[i]])
            y_true_flat.append([true_potencial[i], true_recuperacao[i]])
    
    predictions_flat = np.array(predictions_flat)
    y_true_flat = np.array(y_true_flat)
    
    fig = cria_grafico_real_previsto_mlp(
        predictions=predictions_flat,
        y_true=y_true_flat,
        titulo="Real vs Previsto - Oscilador de FitzHugh-Nagumo (Dados de Teste)"
    )
    
    fig.write_html(grafico_previsoes_mlp)    
    fig.show()
    
    return None

def visualiza_previsoes_espaco_fases_node(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any],
    tempos_referencia: np.ndarray
) -> None:
    """
    Node: Visualiza as previsões do modelo no espaço de fases (Potencial vs Recuperação).
    Mostra trajetórias completas.
    
    Args:
        model: Modelo treinado
        X_test: Dados de teste (condições iniciais)
        y_test: Targets de teste (trajetórias completas)
        scaler_y: Scaler dos targets
        parameters: Parâmetros do pipeline
        tempos_referencia: Array com os tempos para plotagem
    """
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'default_v1')
    seed = parameters.get('seed', 42)

    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_previsoes_espaco_fases = f"{output_dir}/previsoes_espaco_fases.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print("\n=== VISUALIZAÇÃO DAS PREVISÕES NO ESPAÇO DE FASES ===")
    print(f"  Número de trajetórias de teste: {len(X_test)}")
    print(f"  Nós de saída do modelo por trajetória: {len(tempos_referencia)}")
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).cpu().numpy()
    
    # desnormaliza previsões
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    rng = np.random.RandomState(seed)
    
    # seleciona algumas trajetórias para visualização
    num_trajetorias_vis = min(10, len(predictions))
    indices_vis = rng.choice(len(predictions), num_trajetorias_vis, replace=False)
    
    # dados para o gráfico de espaço de fases
    y_potencial_true_list = []
    y_recuperacao_true_list = []
    y_potencial_pred_list = []
    y_recuperacao_pred_list = []
    
    for idx in indices_vis:
        pred_traj = predictions[idx]
        true_traj = y_test_original[idx]
        
        pred_potencial = pred_traj[0::2]
        pred_recuperacao = pred_traj[1::2]
        true_potencial = true_traj[0::2]
        true_recuperacao = true_traj[1::2]
        
        y_potencial_true_list.extend(true_potencial)
        y_recuperacao_true_list.extend(true_recuperacao)
        y_potencial_pred_list.extend(pred_potencial)
        y_recuperacao_pred_list.extend(pred_recuperacao)
    
    y_potencial_true = np.array(y_potencial_true_list).reshape(-1, 1)
    y_recuperacao_true = np.array(y_recuperacao_true_list).reshape(-1, 1)
    y_potencial_pred = np.array(y_potencial_pred_list).reshape(-1, 1)
    y_recuperacao_pred = np.array(y_recuperacao_pred_list).reshape(-1, 1)
    
    # calcula métricas para exibição
    rmse_potencial = np.sqrt(mean_squared_error(y_potencial_true, y_potencial_pred))
    rmse_recuperacao = np.sqrt(mean_squared_error(y_recuperacao_true, y_recuperacao_pred))
    r2_potencial = r2_score(y_potencial_true, y_potencial_pred)
    r2_recuperacao = r2_score(y_recuperacao_true, y_recuperacao_pred)
    
    print(f"  RMSE Potencial: {rmse_potencial:.6f}")
    print(f"  RMSE Recuperação: {rmse_recuperacao:.6f}")
    print(f"  R² Potencial: {r2_potencial:.4f}")
    print(f"  R² Recuperação: {r2_recuperacao:.4f}")
    
    fig = cria_grafico_previsoes_espaco_fases(
        y_pot_true=y_potencial_true,
        y_rec_true=y_recuperacao_true,
        y_pot_pred=y_potencial_pred,
        y_rec_pred=y_recuperacao_pred,
        titulo="Previsões do Modelo no Espaço de Fases - Oscilador de FitzHugh-Nagumo"
    )
    
    fig.write_html(grafico_previsoes_espaco_fases)    
    fig.show()
    
    return None


def interpola_trajetorias_avulsas_node(
    model: nn.Module,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any],
    tempos_referencia: np.ndarray = None
) -> None:
    """
    Node: Usa o modelo treinado para fazer interpolações e prever trajetórias completas
    para novas condições iniciais não vistas durante o treinamento.
    Nota: Os parâmetros do sistema (epsilon, a, b, I, R) são fixos para todos os casos.
    
    Args:
        model: Modelo MLP treinado (prevê trajetórias completas)
        scaler_X: Scaler das features de entrada
        scaler_y: Scaler dos targets
        parameters: Parâmetros do pipeline
        tempos_referencia: Array com os tempos para plotagem (opcional)
    """

    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_interpolacao_completa = f"{output_dir}/interpolacao_avulsa_potencial_recuperacao_vs_t.html"
    grafico_interpolacao_espaco_fases = f"{output_dir}/interpolacao_avulsa_espaco_fases.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    intervals = parameters.get('intervals', {})
    
    epsilon = intervals.get('parametro_epsilon', 0.08)
    a = parameters.get('a', 0.7)
    b = parameters.get('b', 0.8)
    I = parameters.get('I', 0.5)
    R = parameters.get('R', 0.1)
    # ponto de equilíbrio
    # v - v^3/3 - w + R * I = 0
    # w = (v + a)/b
    v_eq = 0.0
    w_eq = 0.0
    
    if tempos_referencia is None:
        # período aproximado para definir tempo de simulação
        # Para epsilon pequeno: T ≈ (3 - 2*ln(2))/epsilon
        if epsilon < 0.1:
            T_aprox = (3.0 - 2.0 * np.log(2.0)) / max(epsilon, 1e-10)
        else:
            T_aprox = 2.0 * np.pi / max(epsilon, 1e-10)
        
        sim_params = parameters.get('simulation', {})
        dt = sim_params.get('dt', 0.01)
        num_periodos = sim_params.get('num_periodos', 3)
        t_final = num_periodos * T_aprox
        tempos_referencia = np.arange(0, t_final + dt, dt)
        print(f"  Tempos de referência criados: {len(tempos_referencia)} pontos")
        print(f"  Período aproximado: {T_aprox:.4f} s")
        print(f"  Tempo final: {t_final:.4f} s")
    else:
        print(f"\n  Nós de saída do modelo por trajetória: {len(tempos_referencia)} pontos")
    
    num_timesteps = len(tempos_referencia)
    
    # casos de teste para interpolação (apenas condições iniciais variam)
    casos_teste = [
        {
            "nome": "Caso 1",
            "v0": 0.5,
            "w0": 0.0,
            "cor": CORES_PALETA[0]
        },
        {
            "nome": "Caso 2",
            "v0": 1.5,
            "w0": 0.0,
            "cor": CORES_PALETA[1]
        },
        {
            "nome": "Caso 3",
            "v0": 0.0,
            "w0": 1.0,
            "cor": CORES_PALETA[2]
        },
        {
            "nome": "Caso 4",
            "v0": 2.0,
            "w0": 0.5,
            "cor": CORES_PALETA[3]
        },
        {
            "nome": "Caso 5",
            "v0": -1.0,
            "w0": 0.0,
            "cor": CORES_PALETA[4]
        },
    ]
    
    # gera os nomes das legendas dinamicamente
    for caso in casos_teste:
        caso["nome_legenda"] = (
            f"{caso['nome']}: v0={caso['v0']:.1f}, "
            f"w0={caso['w0']:.1f}"
        )
        # informações do sistema
        caso["epsilon"] = epsilon
        caso["a"] = a
        caso["b"] = b
        caso["I"] = I
        caso["v_eq"] = v_eq
        caso["w_eq"] = w_eq
    
    tempos_lista = []
    potencial_lista = []
    recuperacao_lista = []
    
    for caso in casos_teste:
        # verifica se o número de timesteps é compatível com o modelo
        if len(tempos_referencia) != num_timesteps:
            print(f"  AVISO: {caso['nome']} - Ajustando tempos para {num_timesteps} pontos")
            tempos = np.linspace(tempos_referencia[0], tempos_referencia[-1], num_timesteps)
        else:
            tempos = tempos_referencia
        
        # entrada: [v0, w0] - potencial e recuperação iniciais
        X_caso = np.array([[caso["v0"], caso["w0"]]], dtype=np.float32)
        
        # normaliza a entrada
        X_caso_scaled = scaler_X.transform(X_caso)
        X_tensor = torch.tensor(X_caso_scaled, dtype=torch.float32).to(device)
        
        # previsão: trajetória completa
        with torch.no_grad():
            predictions_scaled = model(X_tensor).cpu().numpy()
        
        # desnormaliza a trajetória completa
        predictions = scaler_y.inverse_transform(predictions_scaled)
        
        # separa potencial e recuperação da trajetória completa
        potencial = predictions[0, 0::2]  # potencial (índices pares)
        recuperacao = predictions[0, 1::2]  # recuperação (índices ímpares)
        
        # se os tempos não têm o mesmo tamanho, ajusta
        if len(potencial) != len(tempos):
            print(f"  AVISO: {caso['nome']} - Ajustando tempos para {len(potencial)} pontos")
            tempos = np.linspace(tempos[0], tempos[-1], len(potencial))
        
        tempos_lista.append(tempos)
        potencial_lista.append(potencial)
        recuperacao_lista.append(recuperacao)
        
        caso["num_pontos"] = len(potencial)
        caso["dt"] = tempos[1] - tempos[0] if len(tempos) > 1 else 0
    
    print("\n=== INTERPOLAÇÃO DE TRAJETÓRIAS AVULSAS ===")
    print(f"  Parâmetros do sistema: ε={epsilon:.4f}, a={a:.2f}, b={b:.2f}, I={I:.2f}, R={R:.2f}")
    print(f"  Ponto de equilíbrio: v*={v_eq:.2f}, w*={w_eq:.2f}")
    for caso in casos_teste:
        print(f"    {caso['nome']}: v0={caso['v0']:.1f}, w0={caso['w0']:.1f}")
    
    fig_completo = cria_grafico_interpolacao_completo(
        tempos_lista=tempos_lista,
        potencial_lista=potencial_lista,
        recuperacao_lista=recuperacao_lista,
        casos_info=casos_teste,
        titulo="Interpolação Avulsa: Potencial e Recuperação vs Tempo - Oscilador de FitzHugh-Nagumo"
    )
    
    fig_fases = cria_grafico_interpolacao_espaco_fases(
        potencial_lista=potencial_lista,
        recuperacao_lista=recuperacao_lista,
        casos_info=casos_teste,
        titulo="Interpolação Avulsa: Espaço de Fases - Oscilador de FitzHugh-Nagumo"
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
    parameters: Dict[str, Any],
    tempos_referencia: np.ndarray = None
) -> pd.DataFrame:
    """
    Node: Usa o modelo treinado para fazer interpolação entre pontos de dados gerados aleatoriamente.
    Faz previsões de trajetórias completas a partir de condições iniciais aleatórias.
    Nota: A interpolação é feita dentro da mesma trajetória, variando apenas o tempo.
    Os parâmetros do sistema (epsilon, a, b, I, R) são constantes.
    
    Args:
        model: Modelo MLP treinado (prevê trajetórias completas)
        scaler_X: Scaler das features de entrada
        scaler_y: Scaler dos targets
        parameters: Parâmetros do pipeline
        tempos_referencia: Array com os tempos de referência (opcional)
        
    Returns:
        DataFrame com os dados interpolados e previsões do modelo
    """
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_interpolacao_pontual = f"{output_dir}/interpolacoes_pontuais_real_previsto_mlp.html"
    grafico_interpolacao_pontual_espaco_fases = f"{output_dir}/interpolacao_pontual_espaco_fases.html"
    grafico_interpolacao_pontual_temporal = f"{output_dir}/interpolacao_pontual_potencial_recuperacao_vs_t.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    intervals = parameters.get('intervals', {})
    seed = parameters.get('seed', 42)
    
    epsilon = intervals.get('parametro_epsilon', 0.08)
    a = parameters.get('a', 0.7)
    b = parameters.get('b', 0.8)
    I = parameters.get('I', 0.5)
    R = parameters.get('R', 0.1)
    
    # ponto de equilíbrio aproximado
    v_eq = 0.0
    w_eq = 0.0
    
    # fixa a semente para reprodutibilidade
    np.random.seed(seed)
    
    print("\n=== INTERPOLAÇÃO PONTUAL - OSCILADOR DE FITZHUGH-NAGUMO ===")
    print(f"  Parâmetros do sistema: ε={epsilon:.4f}, a={a:.2f}, b={b:.2f}, I={I:.2f}, R={R:.2f}")
    print(f"  Ponto de equilíbrio: v*={v_eq:.2f}, w*={w_eq:.2f}")
    print("\n  A interpolação é feita variando o tempo para uma mesma trajetória")
    
    # ============================================
    # GERAÇÃO DE DADOS ALEATÓRIOS
    # ============================================
    
    # limites dos intervalos dos parâmetros
    v0_min = intervals.get('v0_min', -3.0)
    v0_max = intervals.get('v0_max', 3.0)
    w0_min = intervals.get('w0_min', -3.0)
    w0_max = intervals.get('w0_max', 3.0)
    
    # número de trajetórias a serem geradas
    num_trajetorias = 2
    
    print(f"\n  Gerando {num_trajetorias} trajetórias aleatórias:")
    print(f"    v0 no intervalo [{v0_min:.3f}, {v0_max:.3f}]")
    print(f"    w0 no intervalo [{w0_min:.3f}, {w0_max:.3f}]")
    
    # gera condições iniciais aleatórias
    v0_values = np.random.uniform(v0_min, v0_max, num_trajetorias)
    w0_values = np.random.uniform(w0_min, w0_max, num_trajetorias)
    
    # número de pontos baseado nos tempos de referência ou padrão
    if tempos_referencia is not None:
        num_pontos_por_trajetoria = len(tempos_referencia)
        tempos_interpolados = tempos_referencia
        tempo_maximo = tempos_interpolados[-1]
        dt_interpolacao = tempos_interpolados[1] - tempos_interpolados[0] if len(tempos_interpolados) > 1 else 0.01
        print(f"\n  Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria} pontos")
        print(f"  Tempo máximo: {tempo_maximo:.3f} s")
        print(f"  Passo temporal: {dt_interpolacao:.6f} s")
    else:
        # período aproximado
        if epsilon < 0.1:
            T_aprox = (3.0 - 2.0 * np.log(2.0)) / max(epsilon, 1e-10)
        else:
            T_aprox = 2.0 * np.pi / max(epsilon, 1e-10)
        
        sim_params = parameters.get('simulation', {})
        dt = sim_params.get('dt', 0.01)
        num_periodos = sim_params.get('num_periodos', 3)
        tempo_maximo = num_periodos * T_aprox
        num_pontos_por_trajetoria = int(tempo_maximo / dt) + 1
        tempos_interpolados = np.linspace(0, tempo_maximo, num_pontos_por_trajetoria)
        dt_interpolacao = tempos_interpolados[1] - tempos_interpolados[0]
        
        print(f"\n  Configuração da interpolação:")
        print(f"    Período aproximado: {T_aprox:.3f} s")
        print(f"    Tempo máximo: {tempo_maximo:.3f} s")
        print(f"    Passo temporal: {dt_interpolacao:.6f} s")
        print(f"    Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria}")
    
    todas_previsoes = []
    todos_reais_interpolados = []
    
    # listas para o gráfico temporal
    tempos_lista = []
    potencial_previsto_lista = []
    recuperacao_previsto_lista = []
    potencial_real_lista = []
    recuperacao_real_lista = []
    casos_info_lista = []
    dados_interpolados = []
    
    for idx in range(num_trajetorias):
        v0 = v0_values[idx]
        w0 = w0_values[idx]
        
        print(f"\n  Processando trajetória {idx}: v0={v0:.3f}, w0={w0:.3f}")
        
        osc = OsciladorFitzHughNagumo(
            parametros_epsilon=[epsilon],
            a=a,
            b=b,
            I=I,
            R=R,
            device='cpu'
        )
        
        cond_curta = torch.tensor([[v0, w0]], dtype=torch.float32)
        solucao_curta = osc.resolve_multi_condicoes_sistemas(
            condicoes_iniciais=cond_curta,
            t_final=tempo_maximo,
            dt=dt_interpolacao
        )
        
        # valores reais da simulação
        potencial_real = solucao_curta['potencial'][:, 0, 0]
        recuperacao_real = solucao_curta['recuperacao'][:, 0, 0]
        tempos_reais = solucao_curta['tempo']
        
        if len(tempos_reais) != len(tempos_interpolados):
            interp_potencial = interp1d(tempos_reais, potencial_real, kind='linear', fill_value='extrapolate')
            interp_recuperacao = interp1d(tempos_reais, recuperacao_real, kind='linear', fill_value='extrapolate')
            potencial_real_interpolado = interp_potencial(tempos_interpolados)
            recuperacao_real_interpolado = interp_recuperacao(tempos_interpolados)
        else:
            potencial_real_interpolado = potencial_real
            recuperacao_real_interpolado = recuperacao_real
        
        X_interpolado = np.array([[v0, w0]], dtype=np.float32)
        
        X_interpolado_scaled = scaler_X.transform(X_interpolado)
        X_tensor = torch.tensor(X_interpolado_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()
        
        pred = scaler_y.inverse_transform(pred_scaled)
        
        # separa potencial e recuperação da trajetória completa
        # a saída está no formato: [v0, w0, v1, w1, ..., vN, wN]
        potencial_previsto = pred[0, 0::2]  # potencial (índices pares)
        recuperacao_previsto = pred[0, 1::2]  # recuperação (índices ímpares)
        
        if len(potencial_previsto) != len(tempos_interpolados):
            print(f"    AVISO: Ajustando tempos para {len(potencial_previsto)} pontos")
            tempos_ajustados = np.linspace(tempos_interpolados[0], tempos_interpolados[-1], len(potencial_previsto))
        else:
            tempos_ajustados = tempos_interpolados
        
        pred_pontos = np.column_stack([potencial_previsto, recuperacao_previsto])
        real_pontos = np.column_stack([potencial_real_interpolado, recuperacao_real_interpolado])
        
        if len(potencial_previsto) != len(potencial_real_interpolado):
            interp_potencial = interp1d(tempos_interpolados, potencial_real_interpolado, kind='linear', fill_value='extrapolate')
            interp_recuperacao = interp1d(tempos_interpolados, recuperacao_real_interpolado, kind='linear', fill_value='extrapolate')
            potencial_real_ajustado = interp_potencial(tempos_ajustados)
            recuperacao_real_ajustado = interp_recuperacao(tempos_ajustados)
            real_pontos = np.column_stack([potencial_real_ajustado, recuperacao_real_ajustado])
            tempos_para_grafico = tempos_ajustados
        else:
            tempos_para_grafico = tempos_interpolados
        
        todas_previsoes.append(pred_pontos)
        todos_reais_interpolados.append(real_pontos)
        
        tempos_lista.append(tempos_para_grafico)
        potencial_previsto_lista.append(potencial_previsto)
        recuperacao_previsto_lista.append(recuperacao_previsto)
        potencial_real_lista.append(potencial_real_interpolado[:len(tempos_para_grafico)])
        recuperacao_real_lista.append(recuperacao_real_interpolado[:len(tempos_para_grafico)])
        
        cor = CORES_PALETA[idx % len(CORES_PALETA)]
        
        casos_info_lista.append({
            'v0': v0,
            'w0': w0,
            'cor': cor
        })
        
        for k in range(len(tempos_para_grafico)):
            dados_interpolados.append({
                'id_trajetoria': f"v0_{v0:.3f}_w0_{w0:.3f}",
                'v0': v0,
                'w0': w0,
                'parametro_epsilon': epsilon,
                'parametro_a': a,
                'parametro_b': b,
                'parametro_I': I,
                'parametro_R': R,
                'potencial_eq': v_eq,
                'recuperacao_eq': w_eq,
                'tempo_interpolado': tempos_para_grafico[k],
                'potencial_real': real_pontos[k, 0],
                'recuperacao_real': real_pontos[k, 1],
                'potencial_previsto_mlp': pred_pontos[k, 0],
                'recuperacao_previsto_mlp': pred_pontos[k, 1],
                'erro_potencial': pred_pontos[k, 0] - real_pontos[k, 0],
                'erro_recuperacao': pred_pontos[k, 1] - real_pontos[k, 1],
                'erro_abs_potencial': abs(pred_pontos[k, 0] - real_pontos[k, 0]),
                'erro_abs_recuperacao': abs(pred_pontos[k, 1] - real_pontos[k, 1]),
            })
    
    if len(todas_previsoes) == 0:
        print("  ERRO: Nenhuma trajetória válida encontrada para interpolação")
        return pd.DataFrame()
        
    predictions_all = np.vstack(todas_previsoes)
    y_true_all = np.vstack(todos_reais_interpolados)
    
    rmse_potencial = float(np.sqrt(mean_squared_error(y_true_all[:, 0], predictions_all[:, 0])))
    rmse_recuperacao = float(np.sqrt(mean_squared_error(y_true_all[:, 1], predictions_all[:, 1])))
    r2_potencial = float(r2_score(y_true_all[:, 0], predictions_all[:, 0]))
    r2_recuperacao = float(r2_score(y_true_all[:, 1], predictions_all[:, 1]))
    
    print(f"\n  Total de pontos interpolados: {len(predictions_all)}")
    print(f"  RMSE Potencial (vs solução RK4): {rmse_potencial:.6f}")
    print(f"  RMSE Recuperação (vs solução RK4): {rmse_recuperacao:.6f}")
    print(f"  R² Potencial (vs solução RK4): {r2_potencial:.4f}")
    print(f"  R² Recuperação (vs solução RK4): {r2_recuperacao:.4f}")
    
    # ============================================
    # GRÁFICO 1: Real vs Previsto
    # ============================================
    
    fig1 = cria_grafico_interpolacao_pontual_mlp(
        predictions=predictions_all,
        y_true=y_true_all,
        titulo="Interpolação Pontual: RK4 vs MLP - Dados Gerados Aleatoriamente - Oscilador de FitzHugh-Nagumo"
    )
    
    fig1.write_html(grafico_interpolacao_pontual)
    
    # ============================================
    # GRÁFICO 2: Espaço de Fases
    # ============================================
    
    y_potencial_true = y_true_all[:, 0].reshape(-1, 1)
    y_recuperacao_true = y_true_all[:, 1].reshape(-1, 1)
    y_potencial_pred = predictions_all[:, 0].reshape(-1, 1)
    y_recuperacao_pred = predictions_all[:, 1].reshape(-1, 1)
    
    fig2 = cria_grafico_interpolacao_pontual_espaco_fases(
        y_pot_true=y_potencial_true,
        y_rec_true=y_recuperacao_true,
        y_pot_pred=y_potencial_pred,
        y_rec_pred=y_recuperacao_pred,
        titulo="Interpolação Pontual: MLP vs RK4 - Espaço de Fases - Oscilador de FitzHugh-Nagumo"
    )
    
    fig2.write_html(grafico_interpolacao_pontual_espaco_fases)
    
    # ============================================
    # GRÁFICO 3: Potencial e Recuperação vs Tempo
    # ============================================
    
    fig3 = cria_grafico_interpolacao_pontual_completo(
        tempos_lista=tempos_lista,
        potenciais_previstos_lista=potencial_previsto_lista,
        recuperacoes_previstos_lista=recuperacao_previsto_lista,
        potenciais_reais_lista=potencial_real_lista,
        recuperacoes_reais_lista=recuperacao_real_lista,
        casos_info=casos_info_lista,
        titulo="Interpolação Pontual: MLP vs RK4 - Potencial e Recuperação vs Tempo - Oscilador de FitzHugh-Nagumo"
    )
    
    fig3.write_html(grafico_interpolacao_pontual_temporal)
    
    fig1.show()
    fig2.show()
    fig3.show()
    
    df_interpolado = pd.DataFrame(dados_interpolados)
    
    df_interpolado.attrs['rmse_potencial'] = rmse_potencial
    df_interpolado.attrs['rmse_recuperacao'] = rmse_recuperacao
    df_interpolado.attrs['r2_potencial'] = r2_potencial
    df_interpolado.attrs['r2_recuperacao'] = r2_recuperacao
    df_interpolado.attrs['total_pontos'] = len(predictions_all)
    df_interpolado.attrs['num_trajetorias'] = num_trajetorias
    df_interpolado.attrs['pontos_por_trajetoria'] = num_pontos_por_trajetoria
    df_interpolado.attrs['parametro_epsilon'] = epsilon
    df_interpolado.attrs['parametro_a'] = a
    df_interpolado.attrs['parametro_b'] = b
    df_interpolado.attrs['parametro_I'] = I
    df_interpolado.attrs['parametro_R'] = R
    df_interpolado.attrs['potencial_eq'] = v_eq
    df_interpolado.attrs['recuperacao_eq'] = w_eq
    df_interpolado.attrs['tempo_maximo'] = tempo_maximo
    df_interpolado.attrs['dt_interpolacao'] = dt_interpolacao
    df_interpolado.attrs['v0_min'] = v0_min
    df_interpolado.attrs['v0_max'] = v0_max
    df_interpolado.attrs['w0_min'] = w0_min
    df_interpolado.attrs['w0_max'] = w0_max
    
    print(f"\n  Base de dados com interpolação temporal dentro de {num_trajetorias} trajetórias gerada com {len(df_interpolado)} registros")
    
    return df_interpolado


def interpola_entre_trajetorias_mlp_node(
    model: nn.Module,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any],
    tempos_referencia: np.ndarray = None
) -> pd.DataFrame:
    """
    Node: Usa o modelo treinado para fazer interpolação entre trajetórias.
    Para cada instante de tempo, interpola entre duas trajetórias diferentes (variando v0 e w0).
    Não mistura dados de treino/validação/teste pois usa dados gerados aleatoriamente.
    Agora prevê trajetórias completas a partir das condições iniciais.
    """
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_interpolacao_entre_trajetorias = f"{output_dir}/interpolacoes_entre_trajetorias_real_previsto_mlp.html"
    grafico_interpolacao_entre_trajetorias_espaco_fases = f"{output_dir}/interpolacao_entre_trajetorias_espaco_fases.html"
    grafico_interpolacao_entre_trajetorias_temporal = f"{output_dir}/interpolacao_entre_trajetorias_potencial_recuperacao_vs_t.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    intervals = parameters.get('intervals', {})
    
    epsilon = intervals.get('parametro_epsilon', 0.08)
    a = parameters.get('a', 0.7)
    b = parameters.get('b', 0.8)
    I = parameters.get('I', 0.5)
    R = parameters.get('R', 0.1)
    
    # ponto de equilíbrio aproximado
    v_eq = 0.0
    w_eq = 0.0
    
    seed = parameters.get('seed', 42)
    np.random.seed(seed)
    
    print("\n=== INTERPOLAÇÃO ENTRE TRAJETÓRIAS - OSCILADOR DE FITZHUGH-NAGUMO ===")
    print(f"  Parâmetros do sistema: ε={epsilon:.4f}, a={a:.2f}, b={b:.2f}, I={I:.2f}, R={R:.2f}")
    print(f"  Ponto de equilíbrio: v*={v_eq:.2f}, w*={w_eq:.2f}")
    print("\n  Para cada instante de tempo, interpola entre duas trajetórias diferentes")
    
    # ============================================
    # GERAÇÃO DE DADOS ALEATÓRIOS
    # ============================================
    
    # limites dos intervalos dos parâmetros
    v0_min = intervals.get('v0_min', -3.0)
    v0_max = intervals.get('v0_max', 3.0)
    w0_min = intervals.get('w0_min', -3.0)
    w0_max = intervals.get('w0_max', 3.0)
    
    # define o número de pontos baseado nos tempos de referência ou padrão
    if tempos_referencia is not None:
        num_pontos_por_trajetoria = len(tempos_referencia)
        tempos_unicos = tempos_referencia
        tempo_maximo = tempos_unicos[-1]
        dt_interpolacao = tempos_unicos[1] - tempos_unicos[0] if len(tempos_unicos) > 1 else 0.01
        print(f"\n  Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria} pontos")
        print(f"  Tempo máximo: {tempo_maximo:.3f} s")
    else:
        # período aproximado
        if epsilon < 0.1:
            T_aprox = (3.0 - 2.0 * np.log(2.0)) / max(epsilon, 1e-10)
        else:
            T_aprox = 2.0 * np.pi / max(epsilon, 1e-10)
        
        sim_params = parameters.get('simulation', {})
        dt = sim_params.get('dt', 0.01)
        num_periodos = sim_params.get('num_periodos', 3)
        tempo_maximo = num_periodos * T_aprox
        num_pontos_por_trajetoria = int(tempo_maximo / dt) + 1
        tempos_unicos = np.linspace(0, tempo_maximo, num_pontos_por_trajetoria)
        dt_interpolacao = tempos_unicos[1] - tempos_unicos[0]
        
        print(f"\n  Configuração da interpolação:")
        print(f"    Período aproximado: {T_aprox:.3f} s")
        print(f"    Tempo máximo: {tempo_maximo:.3f} s")
        print(f"    Passo temporal: {dt_interpolacao:.6f} s")
        print(f"    Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria}")
    
    # gera trajetórias com diferentes amplitudes (distância do equilíbrio)
    v0_candidates = np.random.uniform(v0_min, v0_max, 100)
    w0_candidates = np.random.uniform(w0_min, w0_max, 100)
    
    # calcula distância do equilíbrio (0,0)
    distancias = np.sqrt(v0_candidates**2 + w0_candidates**2)
    
    # trajetória de menor amplitude (mais próxima do equilíbrio)
    idx_pequena = np.argmin(distancias)
    v0_1 = v0_candidates[idx_pequena]
    w0_1 = w0_candidates[idx_pequena]
    
    # trajetória de maior amplitude (mais distante do equilíbrio)
    idx_grande = np.argmax(distancias)
    v0_2 = v0_candidates[idx_grande]
    w0_2 = w0_candidates[idx_grande]
    
    print(f"\n  Trajetória 1 (próxima ao equilíbrio): v0={v0_1:.3f}, w0={w0_1:.3f}")
    print(f"  Trajetória 2 (distante do equilíbrio): v0={v0_2:.3f}, w0={w0_2:.3f}")
    print(f"  Distâncias do equilíbrio: d1={distancias[idx_pequena]:.3f}, d2={distancias[idx_grande]:.3f}")
    
    # define os níveis de interpolação
    alphas = np.linspace(0, 1, 5)  # 5 níveis de interpolação
    
    todas_previsoes = []
    todos_reais_interpolados = []
    
    tempos_lista = []
    potencial_previsto_lista = []
    recuperacao_previsto_lista = []
    potencial_real_lista = []
    recuperacao_real_lista = []
    casos_info_lista = []
    dados_interpolados = []
    
    for alpha in alphas:
        v0_interp = (1 - alpha) * v0_1 + alpha * v0_2
        w0_interp = (1 - alpha) * w0_1 + alpha * w0_2
        
        # entrada para o modelo: v0, w0
        X_interpolado = np.array([[v0_interp, w0_interp]], dtype=np.float32)
        
        # normaliza e faz previsão
        X_interpolado_scaled = scaler_X.transform(X_interpolado)
        X_tensor = torch.tensor(X_interpolado_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()
        
        # desnormaliza a trajetória completa
        pred = scaler_y.inverse_transform(pred_scaled)
        
        # separa potencial e recuperação da trajetória completa
        # a saída está no formato: [v0, w0, v1, w1, ..., vN, wN]
        potencial_previsto = pred[0, 0::2]  # potencial (índices pares)
        recuperacao_previsto = pred[0, 1::2]  # recuperação (índices ímpares)
        
        # verifica se o número de pontos coincide com os tempos
        if len(potencial_previsto) != len(tempos_unicos):
            print(f"  AVISO: Ajustando tempos para {len(potencial_previsto)} pontos")
            tempos_ajustados = np.linspace(tempos_unicos[0], tempos_unicos[-1], len(potencial_previsto))
        else:
            tempos_ajustados = tempos_unicos
        
        osc = OsciladorFitzHughNagumo(
            parametros_epsilon=[epsilon],
            a=a,
            b=b,
            I=I,
            R=R,
            device='cpu'
        )
        
        cond_curta = torch.tensor([[v0_interp, w0_interp]], dtype=torch.float32)
        solucao_curta = osc.resolve_multi_condicoes_sistemas(
            condicoes_iniciais=cond_curta,
            t_final=tempo_maximo,
            dt=dt_interpolacao
        )
        
        potencial_real = solucao_curta['potencial'][:, 0, 0]
        recuperacao_real = solucao_curta['recuperacao'][:, 0, 0]
        tempos_reais = solucao_curta['tempo']
        
        if len(tempos_reais) != len(tempos_ajustados):
            interp_potencial = interp1d(tempos_reais, potencial_real, kind='linear', fill_value='extrapolate')
            interp_recuperacao = interp1d(tempos_reais, recuperacao_real, kind='linear', fill_value='extrapolate')
            potencial_real_ajustado = interp_potencial(tempos_ajustados)
            recuperacao_real_ajustado = interp_recuperacao(tempos_ajustados)
        else:
            potencial_real_ajustado = potencial_real
            recuperacao_real_ajustado = recuperacao_real
        
        pred_pontos = np.column_stack([potencial_previsto, recuperacao_previsto])
        real_pontos = np.column_stack([potencial_real_ajustado, recuperacao_real_ajustado])
        
        todas_previsoes.append(pred_pontos)
        todos_reais_interpolados.append(real_pontos)
        
        tempos_lista.append(tempos_ajustados)
        potencial_previsto_lista.append(potencial_previsto)
        recuperacao_previsto_lista.append(recuperacao_previsto)
        potencial_real_lista.append(potencial_real_ajustado)
        recuperacao_real_lista.append(recuperacao_real_ajustado)
        
        cor_idx = int(alpha * (len(CORES_PALETA) - 1))
        cor = CORES_PALETA[cor_idx]
        
        casos_info_lista.append({
            'alpha': alpha,
            'v0': v0_interp,
            'w0': w0_interp,
            'cor': cor
        })
        
        for k in range(len(tempos_ajustados)):
            dados_interpolados.append({
                'alpha_interpolacao': alpha,
                'v0_original_1': v0_1,
                'w0_original_1': w0_1,
                'v0_original_2': v0_2,
                'w0_original_2': w0_2,
                'v0_interpolado': v0_interp,
                'w0_interpolado': w0_interp,
                'parametro_epsilon': epsilon,
                'parametro_a': a,
                'parametro_b': b,
                'parametro_I': I,
                'parametro_R': R,
                'potencial_eq': v_eq,
                'recuperacao_eq': w_eq,
                'tempo': tempos_ajustados[k],
                'potencial_real': potencial_real_ajustado[k],
                'recuperacao_real': recuperacao_real_ajustado[k],
                'potencial_previsto_mlp': pred_pontos[k, 0],
                'recuperacao_previsto_mlp': pred_pontos[k, 1],
                'erro_potencial': pred_pontos[k, 0] - potencial_real_ajustado[k],
                'erro_recuperacao': pred_pontos[k, 1] - recuperacao_real_ajustado[k],
                'erro_abs_potencial': abs(pred_pontos[k, 0] - potencial_real_ajustado[k]),
                'erro_abs_recuperacao': abs(pred_pontos[k, 1] - recuperacao_real_ajustado[k]),
                'erro_rel_potencial_pct': (abs(pred_pontos[k, 0] - potencial_real_ajustado[k]) / (abs(potencial_real_ajustado[k]) + 1e-6)) * 100,
                'erro_rel_recuperacao_pct': (abs(pred_pontos[k, 1] - recuperacao_real_ajustado[k]) / (abs(recuperacao_real_ajustado[k]) + 1e-6)) * 100,
            })
    
    if len(todas_previsoes) == 0:
        print("  ERRO: Nenhuma interpolação realizada")
        return pd.DataFrame()
    
    predictions_all = np.vstack(todas_previsoes)
    y_true_all = np.vstack(todos_reais_interpolados)
    
    rmse_potencial = float(np.sqrt(mean_squared_error(y_true_all[:, 0], predictions_all[:, 0])))
    rmse_recuperacao = float(np.sqrt(mean_squared_error(y_true_all[:, 1], predictions_all[:, 1])))
    r2_potencial = float(r2_score(y_true_all[:, 0], predictions_all[:, 0]))
    r2_recuperacao = float(r2_score(y_true_all[:, 1], predictions_all[:, 1]))
    
    print(f"\n  Total de pontos interpolados: {len(predictions_all)}")
    print(f"  RMSE Potencial (vs solução RK4): {rmse_potencial:.6f}")
    print(f"  RMSE Recuperação (vs solução RK4): {rmse_recuperacao:.6f}")
    print(f"  R² Potencial (vs solução RK4): {r2_potencial:.4f}")
    print(f"  R² Recuperação (vs solução RK4): {r2_recuperacao:.4f}")
    
    # ============================================
    # GRÁFICO 1: Real vs Previsto
    # ============================================
    
    fig1 = cria_grafico_interpolacao_pontual_mlp(
        predictions=predictions_all,
        y_true=y_true_all,
        titulo="Interpolação entre Trajetórias: RK4 vs MLP - Oscilador de FitzHugh-Nagumo"
    )
    
    fig1.write_html(grafico_interpolacao_entre_trajetorias)
    
    # ============================================
    # GRÁFICO 2: Espaço de Fases
    # ============================================
    
    y_potencial_true = y_true_all[:, 0].reshape(-1, 1)
    y_recuperacao_true = y_true_all[:, 1].reshape(-1, 1)
    y_potencial_pred = predictions_all[:, 0].reshape(-1, 1)
    y_recuperacao_pred = predictions_all[:, 1].reshape(-1, 1)
    
    fig2 = cria_grafico_interpolacao_pontual_espaco_fases(
        y_pot_true=y_potencial_true,
        y_rec_true=y_recuperacao_true,
        y_pot_pred=y_potencial_pred,
        y_rec_pred=y_recuperacao_pred,
        titulo="Interpolação entre Trajetórias: MLP vs RK4 - Espaço de Fases - Oscilador de FitzHugh-Nagumo"
    )
    
    fig2.write_html(grafico_interpolacao_entre_trajetorias_espaco_fases)
    
    # ============================================
    # GRÁFICO 3: Potencial e Recuperação vs Tempo
    # ============================================
    
    fig3 = cria_grafico_interpolacao_pontual_completo(
        tempos_lista=tempos_lista,
        potenciais_previstos_lista=potencial_previsto_lista,
        recuperacoes_previstos_lista=recuperacao_previsto_lista,
        potenciais_reais_lista=potencial_real_lista,
        recuperacoes_reais_lista=recuperacao_real_lista,
        casos_info=casos_info_lista,
        titulo="Interpolação entre Trajetórias: MLP vs RK4 - Potencial e Recuperação vs Tempo - Oscilador de FitzHugh-Nagumo"
    )
    
    fig3.write_html(grafico_interpolacao_entre_trajetorias_temporal)
    
    fig1.show()
    fig2.show()
    fig3.show()
    
    df_interpolado = pd.DataFrame(dados_interpolados)
    
    df_interpolado.attrs['rmse_potencial'] = rmse_potencial
    df_interpolado.attrs['rmse_recuperacao'] = rmse_recuperacao
    df_interpolado.attrs['r2_potencial'] = r2_potencial
    df_interpolado.attrs['r2_recuperacao'] = r2_recuperacao
    df_interpolado.attrs['total_pontos'] = len(predictions_all)
    df_interpolado.attrs['num_trajetorias'] = 2
    df_interpolado.attrs['num_alpha'] = len(alphas)
    df_interpolado.attrs['num_tempos'] = len(tempos_ajustados)
    df_interpolado.attrs['parametro_epsilon'] = epsilon
    df_interpolado.attrs['parametro_a'] = a
    df_interpolado.attrs['parametro_b'] = b
    df_interpolado.attrs['parametro_I'] = I
    df_interpolado.attrs['parametro_R'] = R
    df_interpolado.attrs['potencial_eq'] = v_eq
    df_interpolado.attrs['recuperacao_eq'] = w_eq
    df_interpolado.attrs['v0_min'] = v0_min
    df_interpolado.attrs['v0_max'] = v0_max
    df_interpolado.attrs['w0_min'] = w0_min
    df_interpolado.attrs['w0_max'] = w0_max
    
    # ========================================================================
    # GRÁFICO 4: Trajetórias Originais e Interpoladas no Espaço de Fases
    # ========================================================================
    
    osc1 = OsciladorFitzHughNagumo(
        parametros_epsilon=[epsilon],
        a=a,
        b=b,
        I=I,
        R=R,
        device='cpu'
    )
    
    cond1 = torch.tensor([[v0_1, w0_1]], dtype=torch.float32)
    sol1 = osc1.resolve_multi_condicoes_sistemas(
        condicoes_iniciais=cond1,
        t_final=tempo_maximo,
        dt=dt_interpolacao
    )
    potencial_traj1 = sol1['potencial'][:, 0, 0]
    recuperacao_traj1 = sol1['recuperacao'][:, 0, 0]
    tempos_reais1 = sol1['tempo']
    
    interp_potencial1 = interp1d(tempos_reais1, potencial_traj1, kind='linear', fill_value='extrapolate')
    interp_recuperacao1 = interp1d(tempos_reais1, recuperacao_traj1, kind='linear', fill_value='extrapolate')
    potencial_traj1 = interp_potencial1(tempos_ajustados)
    recuperacao_traj1 = interp_recuperacao1(tempos_ajustados)
    
    cond2 = torch.tensor([[v0_2, w0_2]], dtype=torch.float32)
    sol2 = osc1.resolve_multi_condicoes_sistemas(
        condicoes_iniciais=cond2,
        t_final=tempo_maximo,
        dt=dt_interpolacao
    )
    potencial_traj2 = sol2['potencial'][:, 0, 0]
    recuperacao_traj2 = sol2['recuperacao'][:, 0, 0]
    tempos_reais2 = sol2['tempo']
    
    interp_potencial2 = interp1d(tempos_reais2, potencial_traj2, kind='linear', fill_value='extrapolate')
    interp_recuperacao2 = interp1d(tempos_reais2, recuperacao_traj2, kind='linear', fill_value='extrapolate')
    potencial_traj2 = interp_potencial2(tempos_ajustados)
    recuperacao_traj2 = interp_recuperacao2(tempos_ajustados)
    
    # interpolações
    interpolacoes_para_grafico = []
    alphas_unicos = np.sort(df_interpolado['alpha_interpolacao'].unique())
    
    for alpha in alphas_unicos:
        if alpha == 0 or alpha == 1:
            continue
        
        mask_alpha = df_interpolado['alpha_interpolacao'] == alpha
        dados_alpha = df_interpolado[mask_alpha].sort_values('tempo')
        
        v0_interp = dados_alpha['v0_interpolado'].iloc[0]
        w0_interp = dados_alpha['w0_interpolado'].iloc[0]
        
        interpolacoes_para_grafico.append({
            'alpha': alpha,
            'potenciais': dados_alpha['potencial_previsto_mlp'].values,
            'recuperacoes': dados_alpha['recuperacao_previsto_mlp'].values,
            'v0_interp': v0_interp,
            'w0_interp': w0_interp,
            'parametro_R': R
        })
    
    casos_info_grafico = [{
        'v0_1': v0_1,
        'w0_1': w0_1,
        'v0_2': v0_2,
        'w0_2': w0_2
    }]
    
    fig4 = cria_grafico_interpolacao_entre_trajetorias_espaco_fases(
        trajetoria1_pot=potencial_traj1,
        trajetoria1_rec=recuperacao_traj1,
        trajetoria2_pot=potencial_traj2,
        trajetoria2_rec=recuperacao_traj2,
        interpolacoes_lista=interpolacoes_para_grafico,
        casos_info=casos_info_grafico,
        titulo="Interpolação entre Trajetórias no Espaço de Fases - Oscilador de FitzHugh-Nagumo"
    )
    
    grafico_entre_trajetorias_espaco_fases = f"{output_dir}/interpolacao_entre_trajetorias_espaco_fases_detalhado.html"
    fig4.write_html(grafico_entre_trajetorias_espaco_fases)
    
    fig4.show()

    return df_interpolado


def interpola_trajetorias_mlp_node(
    model: nn.Module,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any],
    tempos_referencia: np.ndarray = None
) -> pd.DataFrame:
    """
    Node: Usa o modelo treinado para gerar diferentes condições iniciais a partir de uma trajetória base.
    A partir de uma trajetória escolhida aleatoriamente, gera novas condições iniciais variando v0 e w0.
    Não mistura dados de treino/validação/teste pois usa dados gerados aleatoriamente.
    Prevê trajetórias completas a partir das condições iniciais.
    """
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_interpolacao_trajetorias = f"{output_dir}/interpolacoes_trajetorias_real_previsto_mlp.html"
    grafico_interpolacao_trajetorias_espaco_fases = f"{output_dir}/interpolacao_trajetorias_espaco_fases.html"
    grafico_interpolacao_trajetorias_temporal = f"{output_dir}/interpolacao_trajetorias_potencial_recuperacao_vs_t.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    intervals = parameters.get('intervals', {})
    
    epsilon = intervals.get('parametro_epsilon', 0.08)
    a = parameters.get('a', 0.7)
    b = parameters.get('b', 0.8)
    I = parameters.get('I', 0.5)
    R = parameters.get('R', 0.1)
    
    # ponto de equilíbrio aproximado
    v_eq = 0.0
    w_eq = 0.0
    
    v0_min = intervals.get('v0_min', -3.0)
    v0_max = intervals.get('v0_max', 3.0)
    w0_min = intervals.get('w0_min', -3.0)
    w0_max = intervals.get('w0_max', 3.0)
    
    seed = parameters.get('seed', 42)
    np.random.seed(seed)
    
    print("\n=== GERAÇÃO DE CONDIÇÕES INICIAIS A PARTIR DA TRAJETÓRIA BASE - OSCILADOR DE FITZHUGH-NAGUMO ===")
    print(f"  Parâmetros do sistema: ε={epsilon:.4f}, a={a:.2f}, b={b:.2f}, I={I:.2f}, R={R:.2f}")
    print(f"  Ponto de equilíbrio: v*={v_eq:.2f}, w*={w_eq:.2f}")
    print("  Gerando novas condições iniciais variando v0 e w0 dentro dos limites de treino do modelo")
    
    # ============================================
    # GERAÇÃO DE DADOS ALEATÓRIOS
    # ============================================
    
    # define o número de pontos baseado nos tempos de referência ou padrão
    if tempos_referencia is not None:
        num_pontos_por_trajetoria = len(tempos_referencia)
        tempos_unicos = tempos_referencia
        tempo_maximo = tempos_unicos[-1]
        dt_interpolacao = tempos_unicos[1] - tempos_unicos[0] if len(tempos_unicos) > 1 else 0.01
        print(f"\n  Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria} pontos")
        print(f"  Tempo máximo: {tempo_maximo:.3f} s")
    else:
        # período aproximado
        if epsilon < 0.1:
            T_aprox = (3.0 - 2.0 * np.log(2.0)) / max(epsilon, 1e-10)
        else:
            T_aprox = 2.0 * np.pi / max(epsilon, 1e-10)
        
        sim_params = parameters.get('simulation', {})
        dt = sim_params.get('dt', 0.01)
        num_periodos = sim_params.get('num_periodos', 3)
        tempo_maximo = num_periodos * T_aprox
        num_pontos_por_trajetoria = int(tempo_maximo / dt) + 1
        tempos_unicos = np.linspace(0, tempo_maximo, num_pontos_por_trajetoria)
        dt_interpolacao = tempos_unicos[1] - tempos_unicos[0]
        
        print(f"\n  Configuração da interpolação:")
        print(f"    Período aproximado: {T_aprox:.3f} s")
        print(f"    Tempo máximo: {tempo_maximo:.3f} s")
        print(f"    Passo temporal: {dt_interpolacao:.6f} s")
        print(f"    Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria}")
    
    # gera uma trajetória base aleatória
    v0_base = np.random.uniform(v0_min, v0_max)
    w0_base = np.random.uniform(w0_min, w0_max)
    
    print(f"\n  Trajetória Base Selecionada:")
    print(f"    v0 = {v0_base:.3f}")
    print(f"    w0 = {w0_base:.3f}")
    print(f"    Distância do equilíbrio: {np.sqrt(v0_base**2 + w0_base**2):.3f}")
    print(f"    Tempo máximo: {tempo_maximo:.3f} s")
    
    num_variacoes = 5
    variacoes = []
    
    # geração aleatória dentro dos limites
    np.random.seed(seed)
    v0_variacoes = np.random.uniform(v0_min, v0_max, num_variacoes)
    w0_variacoes = np.random.uniform(w0_min, w0_max, num_variacoes)
        
    print(f"\n  Gerando {num_variacoes} novas condições iniciais:")
    for i in range(num_variacoes):
        distancia = np.sqrt(v0_variacoes[i]**2 + w0_variacoes[i]**2)
        variacoes.append({
            'v0': v0_variacoes[i],
            'w0': w0_variacoes[i],
            'distancia_eq': distancia
        })
        print(f"    Caso {i+1}: v0={v0_variacoes[i]:.3f}, w0={w0_variacoes[i]:.3f}, distância={distancia:.3f}")
    
    todas_previsoes = []
    todos_reais_interpolados = []
    
    tempos_lista = []
    potencial_previsto_lista = []
    recuperacao_previsto_lista = []
    potencial_real_lista = []
    recuperacao_real_lista = []
    casos_info_lista = []
    dados_interpolados = []
    
    # para cada nova condição inicial, faz a previsão
    for var_idx, var in enumerate(variacoes):
        v0_novo = var['v0']
        w0_novo = var['w0']
        
        # entrada para o modelo: v0, w0
        X_novo = np.array([[v0_novo, w0_novo]], dtype=np.float32)
        
        # normaliza e faz previsão
        X_novo_scaled = scaler_X.transform(X_novo)
        X_tensor = torch.tensor(X_novo_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()
        
        # desnormaliza a trajetória completa
        pred = scaler_y.inverse_transform(pred_scaled)
        
        # separa potencial e recuperação da trajetória completa
        # saída: [v0, w0, v1, w1, ..., vN, wN]
        potencial_previsto = pred[0, 0::2]  # potencial (índices pares)
        recuperacao_previsto = pred[0, 1::2]  # recuperação (índices ímpares)
        
        # verifica se o número de pontos coincide com os tempos
        if len(potencial_previsto) != len(tempos_unicos):
            print(f"  AVISO: Ajustando tempos para {len(potencial_previsto)} pontos")
            tempos_ajustados = np.linspace(tempos_unicos[0], tempos_unicos[-1], len(potencial_previsto))
        else:
            tempos_ajustados = tempos_unicos
        
        osc = OsciladorFitzHughNagumo(
            parametros_epsilon=[epsilon],
            a=a,
            b=b,
            I=I,
            R=R,
            device='cpu'
        )
        
        cond_curta = torch.tensor([[v0_novo, w0_novo]], dtype=torch.float32)
        solucao_curta = osc.resolve_multi_condicoes_sistemas(
            condicoes_iniciais=cond_curta,
            t_final=tempo_maximo,
            dt=dt_interpolacao
        )
        
        potencial_real = solucao_curta['potencial'][:, 0, 0]
        recuperacao_real = solucao_curta['recuperacao'][:, 0, 0]
        tempos_reais = solucao_curta['tempo']
        
        if len(tempos_reais) != len(tempos_ajustados):
            interp_potencial = interp1d(tempos_reais, potencial_real, kind='linear', fill_value='extrapolate')
            interp_recuperacao = interp1d(tempos_reais, recuperacao_real, kind='linear', fill_value='extrapolate')
            potencial_real_ajustado = interp_potencial(tempos_ajustados)
            recuperacao_real_ajustado = interp_recuperacao(tempos_ajustados)
        else:
            potencial_real_ajustado = potencial_real
            recuperacao_real_ajustado = recuperacao_real
        
        # métricas globais (ponto a ponto para compatibilidade)
        pred_pontos = np.column_stack([potencial_previsto, recuperacao_previsto])
        real_pontos = np.column_stack([potencial_real_ajustado, recuperacao_real_ajustado])
        
        todas_previsoes.append(pred_pontos)
        todos_reais_interpolados.append(real_pontos)
        
        tempos_lista.append(tempos_ajustados)
        potencial_previsto_lista.append(potencial_previsto)
        recuperacao_previsto_lista.append(recuperacao_previsto)
        potencial_real_lista.append(potencial_real_ajustado)
        recuperacao_real_lista.append(recuperacao_real_ajustado)
        
        cor = CORES_PALETA[var_idx % len(CORES_PALETA)]
        
        casos_info_lista.append({
            'v0': v0_novo,
            'w0': w0_novo,
            'cor': cor,
            'variation_id': var_idx
        })
        
        for k in range(len(tempos_ajustados)):
            dados_interpolados.append({
                'variacao_id': var_idx,
                'v0': v0_novo,
                'w0': w0_novo,
                'parametro_epsilon': epsilon,
                'parametro_a': a,
                'parametro_b': b,
                'parametro_I': I,
                'parametro_R': R,
                'potencial_eq': v_eq,
                'recuperacao_eq': w_eq,
                'tempo': tempos_ajustados[k],
                'potencial_real': potencial_real_ajustado[k],
                'recuperacao_real': recuperacao_real_ajustado[k],
                'potencial_previsto_mlp': pred_pontos[k, 0],
                'recuperacao_previsto_mlp': pred_pontos[k, 1],
                'erro_potencial': pred_pontos[k, 0] - potencial_real_ajustado[k],
                'erro_recuperacao': pred_pontos[k, 1] - recuperacao_real_ajustado[k],
                'erro_abs_potencial': abs(pred_pontos[k, 0] - potencial_real_ajustado[k]),
                'erro_abs_recuperacao': abs(pred_pontos[k, 1] - recuperacao_real_ajustado[k]),
                'erro_rel_potencial_pct': (abs(pred_pontos[k, 0] - potencial_real_ajustado[k]) / (abs(potencial_real_ajustado[k]) + 1e-6)) * 100,
                'erro_rel_recuperacao_pct': (abs(pred_pontos[k, 1] - recuperacao_real_ajustado[k]) / (abs(recuperacao_real_ajustado[k]) + 1e-6)) * 100,
            })
    
    if len(todas_previsoes) == 0:
        print("  ERRO: Nenhuma previsão realizada")
        return pd.DataFrame()
    
    predictions_all = np.vstack(todas_previsoes)
    y_true_all = np.vstack(todos_reais_interpolados)
    
    rmse_potencial = float(np.sqrt(mean_squared_error(y_true_all[:, 0], predictions_all[:, 0])))
    rmse_recuperacao = float(np.sqrt(mean_squared_error(y_true_all[:, 1], predictions_all[:, 1])))
    r2_potencial = float(r2_score(y_true_all[:, 0], predictions_all[:, 0]))
    r2_recuperacao = float(r2_score(y_true_all[:, 1], predictions_all[:, 1]))
    
    print(f"\n  Total de pontos previstos: {len(predictions_all)}")
    print(f"  RMSE Potencial (vs solução RK4): {rmse_potencial:.6f}")
    print(f"  RMSE Recuperação (vs solução RK4): {rmse_recuperacao:.6f}")
    print(f"  R² Potencial (vs solução RK4): {r2_potencial:.4f}")
    print(f"  R² Recuperação (vs solução RK4): {r2_recuperacao:.4f}")
    
    # ============================================
    # GRÁFICO 1: Real vs Previsto
    # ============================================
    
    fig1 = cria_grafico_interpolacao_pontual_mlp(
        predictions=predictions_all,
        y_true=y_true_all,
        titulo="Novas Condições Iniciais: RK4 vs MLP - Oscilador de FitzHugh-Nagumo"
    )
    
    fig1.write_html(grafico_interpolacao_trajetorias)
    
    # ============================================
    # GRÁFICO 2: Espaço de Fases
    # ============================================
    
    y_potencial_true = y_true_all[:, 0].reshape(-1, 1)
    y_recuperacao_true = y_true_all[:, 1].reshape(-1, 1)
    y_potencial_pred = predictions_all[:, 0].reshape(-1, 1)
    y_recuperacao_pred = predictions_all[:, 1].reshape(-1, 1)
    
    fig2 = cria_grafico_interpolacao_pontual_espaco_fases(
        y_pot_true=y_potencial_true,
        y_rec_true=y_recuperacao_true,
        y_pot_pred=y_potencial_pred,
        y_rec_pred=y_recuperacao_pred,
        titulo="Novas Condições Iniciais: MLP vs RK4 - Espaço de Fases - Oscilador de FitzHugh-Nagumo"
    )
    
    fig2.write_html(grafico_interpolacao_trajetorias_espaco_fases)
    
    # ============================================
    # GRÁFICO 3: Potencial e Recuperação vs Tempo
    # ============================================
    
    fig3 = cria_grafico_interpolacao_pontual_completo(
        tempos_lista=tempos_lista,
        potenciais_previstos_lista=potencial_previsto_lista,
        recuperacoes_previstos_lista=recuperacao_previsto_lista,
        potenciais_reais_lista=potencial_real_lista,
        recuperacoes_reais_lista=recuperacao_real_lista,
        casos_info=casos_info_lista,
        titulo="Novas Condições Iniciais: MLP vs RK4 - Potencial e Recuperação vs Tempo - Oscilador de FitzHugh-Nagumo"
    )
    
    fig3.write_html(grafico_interpolacao_trajetorias_temporal)
    
    fig1.show()
    fig2.show()
    fig3.show()
    
    df_interpolado = pd.DataFrame(dados_interpolados)
    
    df_interpolado.attrs['rmse_potencial'] = rmse_potencial
    df_interpolado.attrs['rmse_recuperacao'] = rmse_recuperacao
    df_interpolado.attrs['r2_potencial'] = r2_potencial
    df_interpolado.attrs['r2_recuperacao'] = r2_recuperacao
    df_interpolado.attrs['total_pontos'] = len(predictions_all)
    df_interpolado.attrs['num_variacoes'] = num_variacoes
    df_interpolado.attrs['num_tempos'] = len(tempos_ajustados)
    df_interpolado.attrs['parametro_epsilon'] = epsilon
    df_interpolado.attrs['parametro_a'] = a
    df_interpolado.attrs['parametro_b'] = b
    df_interpolado.attrs['parametro_I'] = I
    df_interpolado.attrs['parametro_R'] = R
    df_interpolado.attrs['potencial_eq'] = v_eq
    df_interpolado.attrs['recuperacao_eq'] = w_eq
    df_interpolado.attrs['v0_min'] = v0_min
    df_interpolado.attrs['v0_max'] = v0_max
    df_interpolado.attrs['w0_min'] = w0_min
    df_interpolado.attrs['w0_max'] = w0_max
    
    # ========================================================================
    # GRÁFICO 4: Trajetória Base e Novas Condições Iniciais no Espaço de Fases
    # ========================================================================
    
    # gera a trajetória base via RK4
    osc_base = OsciladorFitzHughNagumo(
        parametros_epsilon=[epsilon],
        a=a,
        b=b,
        I=I,
        R=R,
        device='cpu'
    )
    
    cond_base = torch.tensor([[v0_base, w0_base]], dtype=torch.float32)
    sol_base = osc_base.resolve_multi_condicoes_sistemas(
        condicoes_iniciais=cond_base,
        t_final=tempo_maximo,
        dt=dt_interpolacao
    )
    
    potencial_base = sol_base['potencial'][:, 0, 0]
    recuperacao_base = sol_base['recuperacao'][:, 0, 0]
    tempos_base = sol_base['tempo']
    
    if len(tempos_base) != len(tempos_ajustados):
        interp_potencial_base = interp1d(tempos_base, potencial_base, kind='linear', fill_value='extrapolate')
        interp_recuperacao_base = interp1d(tempos_base, recuperacao_base, kind='linear', fill_value='extrapolate')
        potencial_base = interp_potencial_base(tempos_ajustados)
        recuperacao_base = interp_recuperacao_base(tempos_ajustados)
    
    novas_trajetorias_para_grafico = []
    
    for var_idx in range(num_variacoes):
        mask_var = df_interpolado['variacao_id'] == var_idx
        dados_var = df_interpolado[mask_var].sort_values('tempo')
        
        if len(dados_var) > 0:
            v0_var = dados_var['v0'].iloc[0]
            w0_var = dados_var['w0'].iloc[0]
            
            novas_trajetorias_para_grafico.append({
                'variacao_id': var_idx,
                'potenciais': dados_var['potencial_previsto_mlp'].values,
                'recuperacoes': dados_var['recuperacao_previsto_mlp'].values,
                'v0': v0_var,
                'w0': w0_var
            })
    
    casos_info_grafico = {
        'v0_base': v0_base,
        'w0_base': w0_base,
    }
    
    fig4 = cria_grafico_interpolacao_trajetorias_espaco_fases(
        trajetoria_base_pot=potencial_base,
        trajetoria_base_rec=recuperacao_base,
        novas_trajetorias_lista=novas_trajetorias_para_grafico,
        casos_info=casos_info_grafico,
        titulo="Trajetória Base vs Novas Condições Iniciais no Espaço de Fases - Oscilador de FitzHugh-Nagumo"
    )
    
    grafico_novas_trajetorias = f"{output_dir}/trajetoria_base_vs_novas_condicoes.html"
    fig4.write_html(grafico_novas_trajetorias)
    
    fig4.show()

    return df_interpolado