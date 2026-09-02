"""
Nodes do pipeline MLP para previsão de trajetórias completas do oscilador de Lotka-Volterra.
Entrada: [x0, y0] (presas, predadores)
Saída: Trajetória completa [x_0, y_0, x_1, y_1, ..., x_N, y_N]
"""

import numpy as np
import pandas as pd
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import interp1d
from typing import Dict, Any, Tuple
from .model import MLP
from oscilador_lotka_volterra.pipelines.p00_data_generating.olv import OsciladorLotkaVolterra
from oscilador_lotka_volterra.utils import (
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
    Prepara os dados para treinamento do MLP para o oscilador de Lotka-Volterra.
    
    Entrada: [x0, y0] (presas, predadores)
    Saída: Trajetória completa [x_0, y_0, x_1, y_1, ..., x_N, y_N]
    
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
    
    if base_oscilador[['x0', 'y0']].isnull().any().any():
        print("  AVISO: Valores NaN detectados nas colunas numéricas!")
        base_oscilador = base_oscilador.dropna(subset=['x0', 'y0'])
    
    # parâmetros do sistema - ponto de equilíbrio
    if 'presas_eq' in base_oscilador.columns and 'predadores_eq' in base_oscilador.columns:
        x_eq = base_oscilador['presas_eq'].iloc[0] if len(base_oscilador) > 0 else 5.0
        y_eq = base_oscilador['predadores_eq'].iloc[0] if len(base_oscilador) > 0 else 4.0
    else:
        x_eq = 5.0  # valor padrão (c/d com c=1.0, d=0.2)
        y_eq = 4.0  # valor padrão (a/b com a=2.0, b=0.5)
    
    # parâmetros do sistema
    if 'taxa_crescimento_a' in base_oscilador.columns:
        a = base_oscilador['taxa_crescimento_a'].iloc[0] if len(base_oscilador) > 0 else 2.0
    else:
        a = 2.0
    
    if 'taxa_mortalidade_c' in base_oscilador.columns:
        c = base_oscilador['taxa_mortalidade_c'].iloc[0] if len(base_oscilador) > 0 else 1.0
    else:
        c = 1.0
    
    print(f"\n=== BASE DE DADOS ===")
    print(f"  Parâmetros do sistema: a={a:.3f}, c={c:.3f}")
    print(f"  Ponto de equilíbrio: x*={x_eq:.3f}, y*={y_eq:.3f}")
    print(f"  Total de linhas da base: {len(base_oscilador)}")
    
    if 'id_trajetoria' in base_oscilador.columns:
        trajetorias_unicas = base_oscilador['id_trajetoria'].unique()
        print(f"  Total de trajetórias únicas: {len(trajetorias_unicas)}")
        
        if len(trajetorias_unicas) == 1 and 'nan' in str(trajetorias_unicas[0]).lower():
            print("  AVISO: id_trajetoria ainda com problemas. Recriando baseado em x0, y0...")
            base_oscilador['id_trajetoria'] = 'x0_' + base_oscilador['x0'].round(6).astype(str) + \
                                              '_y0_' + base_oscilador['y0'].round(6).astype(str)
            trajetorias_unicas = base_oscilador['id_trajetoria'].unique()
            print(f"  Nova contagem de trajetórias: {len(trajetorias_unicas)}")
    else:
        print("  ERRO: Coluna 'id_trajetoria' não encontrada!")
        print("  Criando id_trajetoria baseado em x0, y0")
        base_oscilador['id_trajetoria'] = 'x0_' + base_oscilador['x0'].round(6).astype(str) + \
                                          '_y0_' + base_oscilador['y0'].round(6).astype(str)
        trajetorias_unicas = base_oscilador['id_trajetoria'].unique()
        print(f"  Total de trajetórias criadas: {len(trajetorias_unicas)}")
    
    # obtém o número de pontos por trajetória a partir dos dados
    primeiro_grupo = base_oscilador[base_oscilador['id_trajetoria'] == trajetorias_unicas[0]].sort_values('tempo')
    num_timesteps = len(primeiro_grupo)
    
    print(f"\n=== PREPARAÇÃO DOS DADOS ===")
    
    X_list = []  # [x0, y0] para cada trajetória
    y_list = []  # trajetória completa intercalada para cada trajetória
    tempos_list = []  # tempos para referência
    
    for traj_id in trajetorias_unicas:
        grupo = base_oscilador[base_oscilador['id_trajetoria'] == traj_id].sort_values('tempo')
        
        if len(grupo) != num_timesteps:
            print(f"  AVISO: Trajetória {traj_id} tem {len(grupo)} pontos, pulando...")
            continue
        
        # entrada: [x0, y0] - presas e predadores iniciais
        x0 = grupo['x0'].iloc[0]
        y0 = grupo['y0'].iloc[0]
        X_list.append([x0, y0])
        
        # saída: trajetória completa intercalada [x0, y0, x1, y1, ..., xN, yN]
        presas = grupo['presas'].values
        predadores = grupo['predadores'].values
        trajetoria = np.column_stack([presas, predadores]).flatten()
        y_list.append(trajetoria)
        
        # tempos para referência
        tempos_list.append(grupo['tempo'].values)
    
    X_raw = np.array(X_list, dtype=np.float32)
    y_raw = np.array(y_list, dtype=np.float32)
    tempos_referencia = np.array(tempos_list[0]) if tempos_list else np.array([])
    
    print(f"\n  Trajetórias válidas: {len(X_raw)}")
    print(f"  Dimensão entrada: {X_raw.shape[1]} (x0, y0)")
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
    
    print(f"\n  Dimensão entrada: {input_dim} (x0, y0)")
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
    
    if 'presas_eq' in base_oscilador.columns and 'predadores_eq' in base_oscilador.columns:
        presas_eq = base_oscilador['presas_eq'].iloc[0]
        predadores_eq = base_oscilador['predadores_eq'].iloc[0]
        
        amplitudes = {}
        for traj_id in trajetorias_unicas:
            grupo = base_oscilador[base_oscilador['id_trajetoria'] == traj_id].iloc[0]
            x0 = grupo['x0']
            y0 = grupo['y0']
            # distância euclidiana do ponto de equilíbrio
            amplitude = np.sqrt((x0 - presas_eq)**2 + (y0 - predadores_eq)**2)
            amplitudes[traj_id] = amplitude
    else:
        amplitudes = {}
        for traj_id in trajetorias_unicas:
            grupo = base_oscilador[base_oscilador['id_trajetoria'] == traj_id]
            presas = grupo['presas'].values
            predadores = grupo['predadores'].values
            amp_presas = np.max(presas) - np.min(presas)
            amp_predadores = np.max(predadores) - np.min(predadores)
            amplitude = np.sqrt(amp_presas**2 + amp_predadores**2)
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
    
    # Nota: O parâmetro omega não é mais utilizado, pois no Lotka-Volterra
    # a frequência não é constante. A amplitude é definida como distância do equilíbrio.
    fig_amp = cria_grafico_distribuicao_amplitudes(
        amplitudes=np.array(amplitudes_ordenadas),
        amplitude_limite_internas=None,
        titulo="Distribuição das Amplitudes das Trajetórias - Lotka-Volterra"
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
    
    y_presas_train = dados_train['presas'].values.astype(np.float32).reshape(-1, 1)
    y_predadores_train = dados_train['predadores'].values.astype(np.float32).reshape(-1, 1)
    y_presas_val = dados_val['presas'].values.astype(np.float32).reshape(-1, 1)
    y_predadores_val = dados_val['predadores'].values.astype(np.float32).reshape(-1, 1)
    y_presas_test = dados_test['presas'].values.astype(np.float32).reshape(-1, 1)
    y_predadores_test = dados_test['predadores'].values.astype(np.float32).reshape(-1, 1)
    
    # ============================================
    # GRÁFICO: Distribuição no Espaço de Fases
    # ============================================
    
    fig = cria_grafico_distribuicao_dados(
        y_pos_train=y_presas_train,
        y_vel_train=y_predadores_train,
        y_pos_val=y_presas_val,
        y_vel_val=y_predadores_val,
        y_pos_test=y_presas_test,
        y_vel_test=y_predadores_test,
        titulo="Distribuição dos Dados no Espaço de Fases - Lotka-Volterra"
    )
    
    fig.write_html(grafico_distribuicao_dados) 
    fig.show()
    
    return None


def cria_modelo_mlp_node(input_dim: int, output_dim: int, parameters: Dict[str, Any]) -> nn.Module:
    """Cria o modelo MLP para previsão de trajetórias completas do Lotka-Volterra."""

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
    print(f"  Dimensão entrada: {input_dim} (x0, y0)")
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
    """Treina o modelo MLP para prever trajetórias completas do Lotka-Volterra."""

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
    print(f"  Entrada: (x0, y0) -> Saída: (trajetória completa presas/predadores)")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Função loss: RMSE (Root Mean Squared Error)")
    
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
        titulo="Evolução da Função de Custo durante o Treinamento do MLP - Lotka-Volterra"
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
    Avalia o modelo MLP nos dados de validação e teste para o Lotka-Volterra.
    
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
    
    # separa presas e predadores das trajetórias
    presas_pred_val = predictions_val[:, 0::2]
    predadores_pred_val = predictions_val[:, 1::2]
    presas_true_val = y_val_original[:, 0::2]
    predadores_true_val = y_val_original[:, 1::2]
    
    presas_pred_test = predictions_test[:, 0::2]
    predadores_pred_test = predictions_test[:, 1::2]
    presas_true_test = y_test_original[:, 0::2]
    predadores_true_test = y_test_original[:, 1::2]
    
    # avalia ponto a ponto
    rmse_presas_val = float(np.sqrt(mean_squared_error(presas_true_val.flatten(), presas_pred_val.flatten())))
    rmse_predadores_val = float(np.sqrt(mean_squared_error(predadores_true_val.flatten(), predadores_pred_val.flatten())))
    r2_presas_val = float(r2_score(presas_true_val.flatten(), presas_pred_val.flatten()))
    r2_predadores_val = float(r2_score(predadores_true_val.flatten(), predadores_pred_val.flatten()))
    
    rmse_presas_test = float(np.sqrt(mean_squared_error(presas_true_test.flatten(), presas_pred_test.flatten())))
    rmse_predadores_test = float(np.sqrt(mean_squared_error(predadores_true_test.flatten(), predadores_pred_test.flatten())))
    r2_presas_test = float(r2_score(presas_true_test.flatten(), presas_pred_test.flatten()))
    r2_predadores_test = float(r2_score(predadores_true_test.flatten(), predadores_pred_test.flatten()))

    metrics = {
        'rmse_presas_val': rmse_presas_val,
        'rmse_predadores_val': rmse_predadores_val,
        'r2_presas_val': r2_presas_val,
        'r2_predadores_val': r2_predadores_val,
        'rmse_presas_test': rmse_presas_test,
        'rmse_predadores_test': rmse_predadores_test,
        'r2_presas_test': r2_presas_test,
        'r2_predadores_test': r2_predadores_test,
    }
    
    print("\n=== AVALIAÇÃO DO MODELO MLP - LOTKA-VOLTERRA ===")
    print(f"  RMSE Presas Validação: {rmse_presas_val:.6f}")
    print(f"  RMSE Predadores Validação: {rmse_predadores_val:.6f}")
    print(f"  R² Presas Validação: {r2_presas_val:.4f}")
    print(f"  R² Predadores Validação: {r2_predadores_val:.4f}")
    print(f"  RMSE Presas Teste: {rmse_presas_test:.6f}")
    print(f"  RMSE Predadores Teste: {rmse_predadores_test:.6f}")
    print(f"  R² Presas Teste: {r2_presas_test:.4f}")
    print(f"  R² Predadores Teste: {r2_predadores_test:.4f}")

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
        
        # intercala presas e predadores (x0, y0, x1, y1, ...)
        pred_presas = pred_traj[0::2]
        pred_predadores = pred_traj[1::2]
        true_presas = true_traj[0::2]
        true_predadores = true_traj[1::2]
        
        for i in range(len(pred_presas)):
            predictions_flat.append([pred_presas[i], pred_predadores[i]])
            y_true_flat.append([true_presas[i], true_predadores[i]])
    
    predictions_flat = np.array(predictions_flat)
    y_true_flat = np.array(y_true_flat)
    
    fig = cria_grafico_real_previsto_mlp(
        predictions=predictions_flat,
        y_true=y_true_flat,
        titulo="Real vs Previsto - Lotka-Volterra (Dados de Teste)"
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
    Node: Visualiza as previsões do modelo no espaço de fases (Presas vs Predadores).
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
    y_presas_true_list = []
    y_predadores_true_list = []
    y_presas_pred_list = []
    y_predadores_pred_list = []
    
    for idx in indices_vis:
        pred_traj = predictions[idx]
        true_traj = y_test_original[idx]
        
        pred_presas = pred_traj[0::2]
        pred_predadores = pred_traj[1::2]
        true_presas = true_traj[0::2]
        true_predadores = true_traj[1::2]
        
        y_presas_true_list.extend(true_presas)
        y_predadores_true_list.extend(true_predadores)
        y_presas_pred_list.extend(pred_presas)
        y_predadores_pred_list.extend(pred_predadores)
    
    y_presas_true = np.array(y_presas_true_list).reshape(-1, 1)
    y_predadores_true = np.array(y_predadores_true_list).reshape(-1, 1)
    y_presas_pred = np.array(y_presas_pred_list).reshape(-1, 1)
    y_predadores_pred = np.array(y_predadores_pred_list).reshape(-1, 1)
    
    # calcula métricas para exibição
    rmse_presas = np.sqrt(mean_squared_error(y_presas_true, y_presas_pred))
    rmse_predadores = np.sqrt(mean_squared_error(y_predadores_true, y_predadores_pred))
    r2_presas = r2_score(y_presas_true, y_presas_pred)
    r2_predadores = r2_score(y_predadores_true, y_predadores_pred)
    
    print(f"  RMSE Presas: {rmse_presas:.6f}")
    print(f"  RMSE Predadores: {rmse_predadores:.6f}")
    print(f"  R² Presas: {r2_presas:.4f}")
    print(f"  R² Predadores: {r2_predadores:.4f}")
    
    fig = cria_grafico_previsoes_espaco_fases(
        y_pos_true=y_presas_true,
        y_vel_true=y_predadores_true,
        y_pos_pred=y_presas_pred,
        y_vel_pred=y_predadores_pred,
        titulo="Previsões do Modelo no Espaço de Fases - Lotka-Volterra"
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
    Nota: Os parâmetros do sistema (a, b, c, d) são fixos para todos os casos.
    
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
    
    grafico_interpolacao_completa = f"{output_dir}/interpolacao_avulsa_presas_predadores_vs_t.html"
    grafico_interpolacao_espaco_fases = f"{output_dir}/interpolacao_avulsa_espaco_fases.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    intervals = parameters.get('intervals', {})
    
    # Parâmetros do sistema Lotka-Volterra
    a = intervals.get('taxa_crescimento', 2.0)
    b = intervals.get('taxa_predacao', 0.5)
    c = intervals.get('taxa_mortalidade', 1.0)
    d = intervals.get('taxa_eficiencia', 0.2)
    
    # Ponto de equilíbrio
    x_eq = c / d
    y_eq = a / b
    
    if tempos_referencia is None:
        # período aproximado para definir tempo de simulação
        osc_temp = OsciladorLotkaVolterra(
            taxas_crescimento=[a],
            taxas_mortalidade=[c],
            taxas_predacao=[b],
            taxas_eficiencia=[d]
        )
        T_aprox = osc_temp.periodos.cpu().numpy()[0]
        
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
            "x0": 2.0,
            "y0": 1.0,
            "cor": CORES_PALETA[0]
        },
        {
            "nome": "Caso 2",
            "x0": 3.0,
            "y0": 0.5,
            "cor": CORES_PALETA[1]
        },
        {
            "nome": "Caso 3",
            "x0": 1.0,
            "y0": 2.0,
            "cor": CORES_PALETA[2]
        },
        {
            "nome": "Caso 4",
            "x0": 4.0,
            "y0": 1.5,
            "cor": CORES_PALETA[3]
        },
        {
            "nome": "Caso 5",
            "x0": 0.5,
            "y0": 3.0,
            "cor": CORES_PALETA[4]
        },
    ]
    
    # gera os nomes das legendas dinamicamente
    for caso in casos_teste:
        caso["nome_legenda"] = (
            f"{caso['nome']}: x0={caso['x0']:.1f}, "
            f"y0={caso['y0']:.1f}"
        )
        # informações do sistema
        caso["a"] = a
        caso["b"] = b
        caso["c"] = c
        caso["d"] = d
        caso["x_eq"] = x_eq
        caso["y_eq"] = y_eq
    
    tempos_lista = []
    presas_lista = []
    predadores_lista = []
    
    for caso in casos_teste:
        # verifica se o número de timesteps é compatível com o modelo
        if len(tempos_referencia) != num_timesteps:
            print(f"  AVISO: {caso['nome']} - Ajustando tempos para {num_timesteps} pontos")
            tempos = np.linspace(tempos_referencia[0], tempos_referencia[-1], num_timesteps)
        else:
            tempos = tempos_referencia
        
        # entrada: [x0, y0] - presas e predadores iniciais
        X_caso = np.array([[caso["x0"], caso["y0"]]], dtype=np.float32)
        
        # normaliza a entrada
        X_caso_scaled = scaler_X.transform(X_caso)
        X_tensor = torch.tensor(X_caso_scaled, dtype=torch.float32).to(device)
        
        # previsão: trajetória completa
        with torch.no_grad():
            predictions_scaled = model(X_tensor).cpu().numpy()
        
        # desnormaliza a trajetória completa
        predictions = scaler_y.inverse_transform(predictions_scaled)
        
        # separa presas e predadores da trajetória completa
        presas = predictions[0, 0::2]  # presas (índices pares)
        predadores = predictions[0, 1::2]  # predadores (índices ímpares)
        
        # se os tempos não têm o mesmo tamanho, ajusta
        if len(presas) != len(tempos):
            print(f"  AVISO: {caso['nome']} - Ajustando tempos para {len(presas)} pontos")
            tempos = np.linspace(tempos[0], tempos[-1], len(presas))
        
        tempos_lista.append(tempos)
        presas_lista.append(presas)
        predadores_lista.append(predadores)
        
        caso["num_pontos"] = len(presas)
        caso["dt"] = tempos[1] - tempos[0] if len(tempos) > 1 else 0
    
    print("\n=== INTERPOLAÇÃO DE TRAJETÓRIAS AVULSAS ===")
    print(f"  Parâmetros do sistema: a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f}")
    print(f"  Ponto de equilíbrio: x*={x_eq:.2f}, y*={y_eq:.2f}")
    for caso in casos_teste:
        print(f"    {caso['nome']}: x0={caso['x0']:.1f}, y0={caso['y0']:.1f}")
    
    fig_completo = cria_grafico_interpolacao_completo(
        tempos_lista=tempos_lista,
        presas_lista=presas_lista,
        predadores_lista=predadores_lista,
        casos_info=casos_teste,
        titulo="Interpolação Avulsa: Presas e Predadores vs Tempo - Lotka-Volterra"
    )
    
    fig_fases = cria_grafico_interpolacao_espaco_fases(
        presas_lista=presas_lista,
        predadores_lista=predadores_lista,
        casos_info=casos_teste,
        titulo="Interpolação Avulsa: Espaço de Fases - Lotka-Volterra"
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
    Os parâmetros do sistema (a, b, c, d) são constantes.
    
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
    grafico_interpolacao_pontual_temporal = f"{output_dir}/interpolacao_pontual_presas_predadores_vs_t.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    intervals = parameters.get('intervals', {})
    seed = parameters.get('seed', 42)
    
    a = intervals.get('taxa_crescimento', 2.0)
    b = intervals.get('taxa_predacao', 0.5)
    c = intervals.get('taxa_mortalidade', 1.0)
    d = intervals.get('taxa_eficiencia', 0.2)
    
    # ponto de equilíbrio
    x_eq = c / d
    y_eq = a / b
    
    # fixa a semente para reprodutibilidade
    np.random.seed(seed)
    
    print("\n=== INTERPOLAÇÃO PONTUAL - LOTKA-VOLTERRA ===")
    print(f"  Parâmetros do sistema: a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f}")
    print(f"  Ponto de equilíbrio: x*={x_eq:.2f}, y*={y_eq:.2f}")
    print("\n  A interpolação é feita variando o tempo para uma mesma trajetória")
    
    # ============================================
    # GERAÇÃO DE DADOS ALEATÓRIOS
    # ============================================
    
    # limites dos intervalos dos parâmetros
    x0_min = intervals.get('x0_min', 0.5)
    x0_max = intervals.get('x0_max', 3.0)
    y0_min = intervals.get('y0_min', 0.3)
    y0_max = intervals.get('y0_max', 2.0)
    
    # número de trajetórias a serem geradas
    num_trajetorias = 2
    
    print(f"\n  Gerando {num_trajetorias} trajetórias aleatórias:")
    print(f"    x0 no intervalo [{x0_min:.3f}, {x0_max:.3f}]")
    print(f"    y0 no intervalo [{y0_min:.3f}, {y0_max:.3f}]")
    
    # gera condições iniciais aleatórias
    x0_values = np.random.uniform(x0_min, x0_max, num_trajetorias)
    y0_values = np.random.uniform(y0_min, y0_max, num_trajetorias)
    
    # número de pontos baseado nos tempos de referência ou padrão
    if tempos_referencia is not None:
        num_pontos_por_trajetoria = len(tempos_referencia)
        tempos_interpolados = tempos_referencia
        # tempo_maximo baseado nos tempos de referência
        tempo_maximo = tempos_interpolados[-1]
        dt_interpolacao = tempos_interpolados[1] - tempos_interpolados[0] if len(tempos_interpolados) > 1 else 0.01
        print(f"\n  Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria} pontos")
        print(f"  Tempo máximo: {tempo_maximo:.3f} s")
        print(f"  Passo temporal: {dt_interpolacao:.6f} s")
    else:
        osc_temp = OsciladorLotkaVolterra(
            taxas_crescimento=[a],
            taxas_mortalidade=[c],
            taxas_predacao=[b],
            taxas_eficiencia=[d]
        )
        T_aprox = osc_temp.periodos.cpu().numpy()[0]
        
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
    presas_previstas_lista = []
    predadores_previstos_lista = []
    presas_reais_lista = []
    predadores_reais_lista = []
    casos_info_lista = []
    dados_interpolados = []
    
    for idx in range(num_trajetorias):
        x0 = x0_values[idx]
        y0 = y0_values[idx]
        
        print(f"\n  Processando trajetória {idx}: x0={x0:.3f}, y0={y0:.3f}")
        
        # para o Lotka-Volterra não temos solução analítica simples,
        # então usamos a simulação RK4 para gerar a solução "real"
        osc = OsciladorLotkaVolterra(
            taxas_crescimento=[a],
            taxas_mortalidade=[c],
            taxas_predacao=[b],
            taxas_eficiencia=[d],
            device='cpu'
        )
        
        cond_curta = torch.tensor([[x0, y0]], dtype=torch.float32)
        solucao_curta = osc.resolve_multi_condicoes_sistemas(
            condicoes_iniciais=cond_curta,
            t_final=tempo_maximo,
            dt=dt_interpolacao
        )
        
        # valores reais da simulação
        presas_reais_interpolados = solucao_curta['presas'][:, 0, 0]
        predadores_reais_interpolados = solucao_curta['predadores'][:, 0, 0]
        tempos_reais = solucao_curta['tempo']
        
        if len(tempos_reais) != len(tempos_interpolados):
            interp_presas = interp1d(tempos_reais, presas_reais_interpolados, kind='linear', fill_value='extrapolate')
            interp_predadores = interp1d(tempos_reais, predadores_reais_interpolados, kind='linear', fill_value='extrapolate')
            presas_reais_interpolados = interp_presas(tempos_interpolados)
            predadores_reais_interpolados = interp_predadores(tempos_interpolados)
        
        X_interpolado = np.array([[x0, y0]], dtype=np.float32)
        
        X_interpolado_scaled = scaler_X.transform(X_interpolado)
        X_tensor = torch.tensor(X_interpolado_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()
        
        pred = scaler_y.inverse_transform(pred_scaled)
        
        # separa presas e predadores da trajetória completa
        # a saída está no formato: [x0, y0, x1, y1, ..., xN, yN]
        presas_previstas = pred[0, 0::2]  # Pega as presas (índices pares)
        predadores_previstos = pred[0, 1::2]  # Pega os predadores (índices ímpares)
        
        if len(presas_previstas) != len(tempos_interpolados):
            print(f"    AVISO: Ajustando tempos para {len(presas_previstas)} pontos")
            tempos_ajustados = np.linspace(tempos_interpolados[0], tempos_interpolados[-1], len(presas_previstas))
        else:
            tempos_ajustados = tempos_interpolados
        
        pred_pontos = np.column_stack([presas_previstas, predadores_previstos])
        real_pontos = np.column_stack([presas_reais_interpolados, predadores_reais_interpolados])
        
        if len(presas_previstas) != len(presas_reais_interpolados):
            interp_presas = interp1d(tempos_interpolados, presas_reais_interpolados, kind='linear', fill_value='extrapolate')
            interp_predadores = interp1d(tempos_interpolados, predadores_reais_interpolados, kind='linear', fill_value='extrapolate')
            presas_reais_ajustados = interp_presas(tempos_ajustados)
            predadores_reais_ajustados = interp_predadores(tempos_ajustados)
            real_pontos = np.column_stack([presas_reais_ajustados, predadores_reais_ajustados])
            tempos_para_grafico = tempos_ajustados
        else:
            tempos_para_grafico = tempos_interpolados
        
        todas_previsoes.append(pred_pontos)
        todos_reais_interpolados.append(real_pontos)
        
        tempos_lista.append(tempos_para_grafico)
        presas_previstas_lista.append(presas_previstas)
        predadores_previstos_lista.append(predadores_previstos)
        presas_reais_lista.append(presas_reais_interpolados[:len(tempos_para_grafico)])
        predadores_reais_lista.append(predadores_reais_interpolados[:len(tempos_para_grafico)])
        
        cor = CORES_PALETA[idx % len(CORES_PALETA)]
        
        casos_info_lista.append({
            'x0': x0,
            'y0': y0,
            'cor': cor
        })
        
        for k in range(len(tempos_para_grafico)):
            dados_interpolados.append({
                'id_trajetoria': f"x0_{x0:.3f}_y0_{y0:.3f}",
                'x0': x0,
                'y0': y0,
                'taxa_crescimento_a': a,
                'taxa_predacao_b': b,
                'taxa_mortalidade_c': c,
                'taxa_eficiencia_d': d,
                'presas_eq': x_eq,
                'predadores_eq': y_eq,
                'tempo_interpolado': tempos_para_grafico[k],
                'presas_real': real_pontos[k, 0],
                'predadores_real': real_pontos[k, 1],
                'presas_previsto_mlp': pred_pontos[k, 0],
                'predadores_previsto_mlp': pred_pontos[k, 1],
                'erro_presas': pred_pontos[k, 0] - real_pontos[k, 0],
                'erro_predadores': pred_pontos[k, 1] - real_pontos[k, 1],
                'erro_abs_presas': abs(pred_pontos[k, 0] - real_pontos[k, 0]),
                'erro_abs_predadores': abs(pred_pontos[k, 1] - real_pontos[k, 1]),
            })
    
    if len(todas_previsoes) == 0:
        print("  ERRO: Nenhuma trajetória válida encontrada para interpolação")
        return pd.DataFrame()
        
    predictions_all = np.vstack(todas_previsoes)
    y_true_all = np.vstack(todos_reais_interpolados)
    
    rmse_presas = float(np.sqrt(mean_squared_error(y_true_all[:, 0], predictions_all[:, 0])))
    rmse_predadores = float(np.sqrt(mean_squared_error(y_true_all[:, 1], predictions_all[:, 1])))
    r2_presas = float(r2_score(y_true_all[:, 0], predictions_all[:, 0]))
    r2_predadores = float(r2_score(y_true_all[:, 1], predictions_all[:, 1]))
    
    print(f"\n  Total de pontos interpolados: {len(predictions_all)}")
    print(f"  RMSE Presas (vs solução RK4): {rmse_presas:.6f}")
    print(f"  RMSE Predadores (vs solução RK4): {rmse_predadores:.6f}")
    print(f"  R² Presas (vs solução RK4): {r2_presas:.4f}")
    print(f"  R² Predadores (vs solução RK4): {r2_predadores:.4f}")
    
    # ============================================
    # GRÁFICO 1: Real vs Previsto
    # ============================================
    
    fig1 = cria_grafico_interpolacao_pontual_mlp(
        predictions=predictions_all,
        y_true=y_true_all,
        titulo="Interpolação Pontual: RK4 vs MLP - Dados Gerados Aleatoriamente - Lotka-Volterra"
    )
    
    fig1.write_html(grafico_interpolacao_pontual)
    
    # ============================================
    # GRÁFICO 2: Espaço de Fases
    # ============================================
    
    y_presas_true = y_true_all[:, 0].reshape(-1, 1)
    y_predadores_true = y_true_all[:, 1].reshape(-1, 1)
    y_presas_pred = predictions_all[:, 0].reshape(-1, 1)
    y_predadores_pred = predictions_all[:, 1].reshape(-1, 1)
    
    fig2 = cria_grafico_interpolacao_pontual_espaco_fases(
        y_pos_true=y_presas_true,
        y_vel_true=y_predadores_true,
        y_pos_pred=y_presas_pred,
        y_vel_pred=y_predadores_pred,
        titulo="Interpolação Pontual: MLP vs RK4 - Espaço de Fases - Lotka-Volterra"
    )
    
    fig2.write_html(grafico_interpolacao_pontual_espaco_fases)
    
    # ============================================
    # GRÁFICO 3: Presas e Predadores vs Tempo
    # ============================================
    
    fig3 = cria_grafico_interpolacao_pontual_completo(
        tempos_lista=tempos_lista,
        posicoes_previstas_lista=presas_previstas_lista,
        velocidades_previstas_lista=predadores_previstos_lista,
        posicoes_reais_lista=presas_reais_lista,
        velocidades_reais_lista=predadores_reais_lista,
        casos_info=casos_info_lista,
        titulo="Interpolação Pontual: MLP vs RK4 - Presas e Predadores vs Tempo - Lotka-Volterra"
    )
    
    fig3.write_html(grafico_interpolacao_pontual_temporal)
    
    fig1.show()
    fig2.show()
    fig3.show()
    
    df_interpolado = pd.DataFrame(dados_interpolados)
    
    df_interpolado.attrs['rmse_presas'] = rmse_presas
    df_interpolado.attrs['rmse_predadores'] = rmse_predadores
    df_interpolado.attrs['r2_presas'] = r2_presas
    df_interpolado.attrs['r2_predadores'] = r2_predadores
    df_interpolado.attrs['total_pontos'] = len(predictions_all)
    df_interpolado.attrs['num_trajetorias'] = num_trajetorias
    df_interpolado.attrs['pontos_por_trajetoria'] = num_pontos_por_trajetoria
    df_interpolado.attrs['taxa_crescimento_a'] = a
    df_interpolado.attrs['taxa_predacao_b'] = b
    df_interpolado.attrs['taxa_mortalidade_c'] = c
    df_interpolado.attrs['taxa_eficiencia_d'] = d
    df_interpolado.attrs['presas_eq'] = x_eq
    df_interpolado.attrs['predadores_eq'] = y_eq
    df_interpolado.attrs['tempo_maximo'] = tempo_maximo
    df_interpolado.attrs['dt_interpolacao'] = dt_interpolacao
    df_interpolado.attrs['x0_min'] = x0_min
    df_interpolado.attrs['x0_max'] = x0_max
    df_interpolado.attrs['y0_min'] = y0_min
    df_interpolado.attrs['y0_max'] = y0_max
    
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
    Para cada instante de tempo, interpola entre duas trajetórias diferentes (variando x0 e y0).
    Não mistura dados de treino/validação/teste pois usa dados gerados aleatoriamente.
    Agora prevê trajetórias completas a partir das condições iniciais.
    """
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_interpolacao_entre_trajetorias = f"{output_dir}/interpolacoes_entre_trajetorias_real_previsto_mlp.html"
    grafico_interpolacao_entre_trajetorias_espaco_fases = f"{output_dir}/interpolacao_entre_trajetorias_espaco_fases.html"
    grafico_interpolacao_entre_trajetorias_temporal = f"{output_dir}/interpolacao_entre_trajetorias_presas_predadores_vs_t.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    intervals = parameters.get('intervals', {})
    
    a = intervals.get('taxa_crescimento', 2.0)
    b = intervals.get('taxa_predacao', 0.5)
    c = intervals.get('taxa_mortalidade', 1.0)
    d = intervals.get('taxa_eficiencia', 0.2)
    
    # ponto de equilíbrio
    x_eq = c / d
    y_eq = a / b
    
    seed = parameters.get('seed', 42)
    np.random.seed(seed)
    
    print("\n=== INTERPOLAÇÃO ENTRE TRAJETÓRIAS - LOTKA-VOLTERRA ===")
    print(f"  Parâmetros do sistema: a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f}")
    print(f"  Ponto de equilíbrio: x*={x_eq:.2f}, y*={y_eq:.2f}")
    print("\n  Para cada instante de tempo, interpola entre duas trajetórias diferentes")
    
    # ============================================
    # GERAÇÃO DE DADOS ALEATÓRIOS
    # ============================================
    
    # limites dos intervalos dos parâmetros
    x0_min = intervals.get('x0_min', 0.5)
    x0_max = intervals.get('x0_max', 3.0)
    y0_min = intervals.get('y0_min', 0.3)
    y0_max = intervals.get('y0_max', 2.0)
    
    # define o número de pontos baseado nos tempos de referência ou padrão
    if tempos_referencia is not None:
        num_pontos_por_trajetoria = len(tempos_referencia)
        tempos_unicos = tempos_referencia
        tempo_maximo = tempos_unicos[-1]
        dt_interpolacao = tempos_unicos[1] - tempos_unicos[0] if len(tempos_unicos) > 1 else 0.01
        print(f"\n  Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria} pontos")
        print(f"  Tempo máximo: {tempo_maximo:.3f} s")
    else:
        osc_temp = OsciladorLotkaVolterra(
            taxas_crescimento=[a],
            taxas_mortalidade=[c],
            taxas_predacao=[b],
            taxas_eficiencia=[d]
        )
        T_aprox = osc_temp.periodos.cpu().numpy()[0]
        
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
    x0_candidates = np.random.uniform(x0_min, x0_max, 100)
    y0_candidates = np.random.uniform(y0_min, y0_max, 100)
    
    # calcula distância do equilíbrio
    distancias = np.sqrt((x0_candidates - x_eq)**2 + (y0_candidates - y_eq)**2)
    
    # trajetória de menor amplitude (mais próxima do equilíbrio)
    idx_pequena = np.argmin(distancias)
    x0_1 = x0_candidates[idx_pequena]
    y0_1 = y0_candidates[idx_pequena]
    
    # trajetória de maior amplitude (mais distante do equilíbrio)
    idx_grande = np.argmax(distancias)
    x0_2 = x0_candidates[idx_grande]
    y0_2 = y0_candidates[idx_grande]
    
    print(f"\n  Trajetória 1 (próxima ao equilíbrio): x0={x0_1:.3f}, y0={y0_1:.3f}")
    print(f"  Trajetória 2 (distante do equilíbrio): x0={x0_2:.3f}, y0={y0_2:.3f}")
    print(f"  Distâncias do equilíbrio: d1={distancias[idx_pequena]:.3f}, d2={distancias[idx_grande]:.3f}")
    
    # define os níveis de interpolação
    alphas = np.linspace(0, 1, 5)  # 5 níveis de interpolação
    
    todas_previsoes = []
    todos_reais_interpolados = []
    
    tempos_lista = []
    presas_previstas_lista = []
    predadores_previstos_lista = []
    presas_reais_lista = []
    predadores_reais_lista = []
    casos_info_lista = []
    dados_interpolados = []
    
    for alpha in alphas:
        x0_interp = (1 - alpha) * x0_1 + alpha * x0_2
        y0_interp = (1 - alpha) * y0_1 + alpha * y0_2
        
        # entrada para o modelo: x0, y0
        X_interpolado = np.array([[x0_interp, y0_interp]], dtype=np.float32)
        
        # normaliza e faz previsão
        X_interpolado_scaled = scaler_X.transform(X_interpolado)
        X_tensor = torch.tensor(X_interpolado_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()
        
        # desnormaliza a trajetória completa
        pred = scaler_y.inverse_transform(pred_scaled)
        
        # separa presas e predadores da trajetória completa
        # a saída está no formato: [x0, y0, x1, y1, ..., xN, yN]
        presas_previstas = pred[0, 0::2]  # presas (índices pares)
        predadores_previstos = pred[0, 1::2]  # predadores (índices ímpares)
        
        # verifica se o número de pontos coincide com os tempos
        if len(presas_previstas) != len(tempos_unicos):
            print(f"  AVISO: Ajustando tempos para {len(presas_previstas)} pontos")
            tempos_ajustados = np.linspace(tempos_unicos[0], tempos_unicos[-1], len(presas_previstas))
        else:
            tempos_ajustados = tempos_unicos
        
        osc = OsciladorLotkaVolterra(
            taxas_crescimento=[a],
            taxas_mortalidade=[c],
            taxas_predacao=[b],
            taxas_eficiencia=[d],
            device='cpu'
        )
        
        cond_curta = torch.tensor([[x0_interp, y0_interp]], dtype=torch.float32)
        solucao_curta = osc.resolve_multi_condicoes_sistemas(
            condicoes_iniciais=cond_curta,
            t_final=tempo_maximo,
            dt=dt_interpolacao
        )
        
        presas_reais = solucao_curta['presas'][:, 0, 0]
        predadores_reais = solucao_curta['predadores'][:, 0, 0]
        tempos_reais = solucao_curta['tempo']
        
        if len(tempos_reais) != len(tempos_ajustados):
            interp_presas = interp1d(tempos_reais, presas_reais, kind='linear', fill_value='extrapolate')
            interp_predadores = interp1d(tempos_reais, predadores_reais, kind='linear', fill_value='extrapolate')
            presas_reais_ajustados = interp_presas(tempos_ajustados)
            predadores_reais_ajustados = interp_predadores(tempos_ajustados)
        else:
            presas_reais_ajustados = presas_reais
            predadores_reais_ajustados = predadores_reais
        
        pred_pontos = np.column_stack([presas_previstas, predadores_previstos])
        real_pontos = np.column_stack([presas_reais_ajustados, predadores_reais_ajustados])
        
        todas_previsoes.append(pred_pontos)
        todos_reais_interpolados.append(real_pontos)
        
        tempos_lista.append(tempos_ajustados)
        presas_previstas_lista.append(presas_previstas)
        predadores_previstos_lista.append(predadores_previstos)
        presas_reais_lista.append(presas_reais_ajustados)
        predadores_reais_lista.append(predadores_reais_ajustados)
        
        cor_idx = int(alpha * (len(CORES_PALETA) - 1))
        cor = CORES_PALETA[cor_idx]
        
        casos_info_lista.append({
            'alpha': alpha,
            'x0': x0_interp,
            'y0': y0_interp,
            'cor': cor
        })
        
        for k in range(len(tempos_ajustados)):
            dados_interpolados.append({
                'alpha_interpolacao': alpha,
                'x0_original_1': x0_1,
                'y0_original_1': y0_1,
                'x0_original_2': x0_2,
                'y0_original_2': y0_2,
                'x0_interpolado': x0_interp,
                'y0_interpolado': y0_interp,
                'taxa_crescimento_a': a,
                'taxa_predacao_b': b,
                'taxa_mortalidade_c': c,
                'taxa_eficiencia_d': d,
                'presas_eq': x_eq,
                'predadores_eq': y_eq,
                'tempo': tempos_ajustados[k],
                'presas_real': presas_reais_ajustados[k],
                'predadores_real': predadores_reais_ajustados[k],
                'presas_previsto_mlp': pred_pontos[k, 0],
                'predadores_previsto_mlp': pred_pontos[k, 1],
                'erro_presas': pred_pontos[k, 0] - presas_reais_ajustados[k],
                'erro_predadores': pred_pontos[k, 1] - predadores_reais_ajustados[k],
                'erro_abs_presas': abs(pred_pontos[k, 0] - presas_reais_ajustados[k]),
                'erro_abs_predadores': abs(pred_pontos[k, 1] - predadores_reais_ajustados[k]),
                'erro_rel_presas_pct': (abs(pred_pontos[k, 0] - presas_reais_ajustados[k]) / (abs(presas_reais_ajustados[k]) + 1e-6)) * 100,
                'erro_rel_predadores_pct': (abs(pred_pontos[k, 1] - predadores_reais_ajustados[k]) / (abs(predadores_reais_ajustados[k]) + 1e-6)) * 100,
            })
    
    if len(todas_previsoes) == 0:
        print("  ERRO: Nenhuma interpolação realizada")
        return pd.DataFrame()
    
    predictions_all = np.vstack(todas_previsoes)
    y_true_all = np.vstack(todos_reais_interpolados)
    
    rmse_presas = float(np.sqrt(mean_squared_error(y_true_all[:, 0], predictions_all[:, 0])))
    rmse_predadores = float(np.sqrt(mean_squared_error(y_true_all[:, 1], predictions_all[:, 1])))
    r2_presas = float(r2_score(y_true_all[:, 0], predictions_all[:, 0]))
    r2_predadores = float(r2_score(y_true_all[:, 1], predictions_all[:, 1]))
    
    print(f"\n  Total de pontos interpolados: {len(predictions_all)}")
    print(f"  RMSE Presas (vs solução RK4): {rmse_presas:.6f}")
    print(f"  RMSE Predadores (vs solução RK4): {rmse_predadores:.6f}")
    print(f"  R² Presas (vs solução RK4): {r2_presas:.4f}")
    print(f"  R² Predadores (vs solução RK4): {r2_predadores:.4f}")
    
    # ============================================
    # GRÁFICO 1: Real vs Previsto
    # ============================================
    
    fig1 = cria_grafico_interpolacao_pontual_mlp(
        predictions=predictions_all,
        y_true=y_true_all,
        titulo="Interpolação entre Trajetórias: RK4 vs MLP - Lotka-Volterra"
    )
    
    fig1.write_html(grafico_interpolacao_entre_trajetorias)
    
    # ============================================
    # GRÁFICO 2: Espaço de Fases
    # ============================================
    
    y_presas_true = y_true_all[:, 0].reshape(-1, 1)
    y_predadores_true = y_true_all[:, 1].reshape(-1, 1)
    y_presas_pred = predictions_all[:, 0].reshape(-1, 1)
    y_predadores_pred = predictions_all[:, 1].reshape(-1, 1)
    
    fig2 = cria_grafico_interpolacao_pontual_espaco_fases(
        y_pos_true=y_presas_true,
        y_vel_true=y_predadores_true,
        y_pos_pred=y_presas_pred,
        y_vel_pred=y_predadores_pred,
        titulo="Interpolação entre Trajetórias: MLP vs RK4 - Espaço de Fases - Lotka-Volterra"
    )
    
    fig2.write_html(grafico_interpolacao_entre_trajetorias_espaco_fases)
    
    # ============================================
    # GRÁFICO 3: Presas e Predadores vs Tempo
    # ============================================
    
    fig3 = cria_grafico_interpolacao_pontual_completo(
        tempos_lista=tempos_lista,
        posicoes_previstas_lista=presas_previstas_lista,
        velocidades_previstas_lista=predadores_previstos_lista,
        posicoes_reais_lista=presas_reais_lista,
        velocidades_reais_lista=predadores_reais_lista,
        casos_info=casos_info_lista,
        titulo="Interpolação entre Trajetórias: MLP vs RK4 - Presas e Predadores vs Tempo - Lotka-Volterra"
    )
    
    fig3.write_html(grafico_interpolacao_entre_trajetorias_temporal)
    
    fig1.show()
    fig2.show()
    fig3.show()
    
    df_interpolado = pd.DataFrame(dados_interpolados)
    
    df_interpolado.attrs['rmse_presas'] = rmse_presas
    df_interpolado.attrs['rmse_predadores'] = rmse_predadores
    df_interpolado.attrs['r2_presas'] = r2_presas
    df_interpolado.attrs['r2_predadores'] = r2_predadores
    df_interpolado.attrs['total_pontos'] = len(predictions_all)
    df_interpolado.attrs['num_trajetorias'] = 2
    df_interpolado.attrs['num_alpha'] = len(alphas)
    df_interpolado.attrs['num_tempos'] = len(tempos_ajustados)
    df_interpolado.attrs['taxa_crescimento_a'] = a
    df_interpolado.attrs['taxa_predacao_b'] = b
    df_interpolado.attrs['taxa_mortalidade_c'] = c
    df_interpolado.attrs['taxa_eficiencia_d'] = d
    df_interpolado.attrs['presas_eq'] = x_eq
    df_interpolado.attrs['predadores_eq'] = y_eq
    df_interpolado.attrs['x0_min'] = x0_min
    df_interpolado.attrs['x0_max'] = x0_max
    df_interpolado.attrs['y0_min'] = y0_min
    df_interpolado.attrs['y0_max'] = y0_max
    
    # ========================================================================
    # GRÁFICO 4: Trajetórias Originais e Interpoladas no Espaço de Fases
    # ========================================================================
    
    osc1 = OsciladorLotkaVolterra(
        taxas_crescimento=[a],
        taxas_mortalidade=[c],
        taxas_predacao=[b],
        taxas_eficiencia=[d],
        device='cpu'
    )
    
    cond1 = torch.tensor([[x0_1, y0_1]], dtype=torch.float32)
    sol1 = osc1.resolve_multi_condicoes_sistemas(
        condicoes_iniciais=cond1,
        t_final=tempo_maximo,
        dt=dt_interpolacao
    )
    presas_traj1 = sol1['presas'][:, 0, 0]
    predadores_traj1 = sol1['predadores'][:, 0, 0]
    tempos_reais1 = sol1['tempo']
    
    interp_presas1 = interp1d(tempos_reais1, presas_traj1, kind='linear', fill_value='extrapolate')
    interp_predadores1 = interp1d(tempos_reais1, predadores_traj1, kind='linear', fill_value='extrapolate')
    presas_traj1 = interp_presas1(tempos_ajustados)
    predadores_traj1 = interp_predadores1(tempos_ajustados)
    
    cond2 = torch.tensor([[x0_2, y0_2]], dtype=torch.float32)
    sol2 = osc1.resolve_multi_condicoes_sistemas(
        condicoes_iniciais=cond2,
        t_final=tempo_maximo,
        dt=dt_interpolacao
    )
    presas_traj2 = sol2['presas'][:, 0, 0]
    predadores_traj2 = sol2['predadores'][:, 0, 0]
    tempos_reais2 = sol2['tempo']
    
    interp_presas2 = interp1d(tempos_reais2, presas_traj2, kind='linear', fill_value='extrapolate')
    interp_predadores2 = interp1d(tempos_reais2, predadores_traj2, kind='linear', fill_value='extrapolate')
    presas_traj2 = interp_presas2(tempos_ajustados)
    predadores_traj2 = interp_predadores2(tempos_ajustados)
    
    # interpolações
    interpolacoes_para_grafico = []
    alphas_unicos = np.sort(df_interpolado['alpha_interpolacao'].unique())
    
    for alpha in alphas_unicos:
        if alpha == 0 or alpha == 1:
            continue
        
        mask_alpha = df_interpolado['alpha_interpolacao'] == alpha
        dados_alpha = df_interpolado[mask_alpha].sort_values('tempo')
        
        x0_interp = dados_alpha['x0_interpolado'].iloc[0]
        y0_interp = dados_alpha['y0_interpolado'].iloc[0]
        
        interpolacoes_para_grafico.append({
            'alpha': alpha,
            'posicoes': dados_alpha['presas_previsto_mlp'].values,
            'velocidades': dados_alpha['predadores_previsto_mlp'].values,
            'x0_interp': x0_interp,
            'v0_interp': y0_interp
        })
    
    casos_info_grafico = [{
        'x0_1': x0_1,
        'v0_1': y0_1,
        'x0_2': x0_2,
        'v0_2': y0_2
    }]
    
    fig4 = cria_grafico_interpolacao_entre_trajetorias_espaco_fases(
        trajetoria1_pos=presas_traj1,
        trajetoria1_vel=predadores_traj1,
        trajetoria2_pos=presas_traj2,
        trajetoria2_vel=predadores_traj2,
        interpolacoes_lista=interpolacoes_para_grafico,
        casos_info=casos_info_grafico,
        titulo="Interpolação entre Trajetórias no Espaço de Fases - Lotka-Volterra"
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
    A partir de uma trajetória escolhida aleatoriamente, gera novas condições iniciais variando x0 e y0.
    Não mistura dados de treino/validação/teste pois usa dados gerados aleatoriamente.
    Prevê trajetórias completas a partir das condições iniciais.
    """
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_interpolacao_trajetorias = f"{output_dir}/interpolacoes_trajetorias_real_previsto_mlp.html"
    grafico_interpolacao_trajetorias_espaco_fases = f"{output_dir}/interpolacao_trajetorias_espaco_fases.html"
    grafico_interpolacao_trajetorias_temporal = f"{output_dir}/interpolacao_trajetorias_presas_predadores_vs_t.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    intervals = parameters.get('intervals', {})
    
    a = intervals.get('taxa_crescimento', 2.0)
    b = intervals.get('taxa_predacao', 0.5)
    c = intervals.get('taxa_mortalidade', 1.0)
    d = intervals.get('taxa_eficiencia', 0.2)
    
    x_eq = c / d
    y_eq = a / b
    
    x0_min = intervals.get('x0_min', 0.5)
    x0_max = intervals.get('x0_max', 3.0)
    y0_min = intervals.get('y0_min', 0.3)
    y0_max = intervals.get('y0_max', 2.0)
    
    seed = parameters.get('seed', 42)
    np.random.seed(seed)
    
    print("\n=== GERAÇÃO DE CONDIÇÕES INICIAIS A PARTIR DA TRAJETÓRIA BASE - LOTKA-VOLTERRA ===")
    print(f"  Parâmetros do sistema: a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f}")
    print(f"  Ponto de equilíbrio: x*={x_eq:.2f}, y*={y_eq:.2f}")
    print("  Gerando novas condições iniciais variando x0 e y0 dentro dos limites de treino do modelo")
    
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
        osc_temp = OsciladorLotkaVolterra(
            taxas_crescimento=[a],
            taxas_mortalidade=[c],
            taxas_predacao=[b],
            taxas_eficiencia=[d]
        )
        T_aprox = osc_temp.periodos.cpu().numpy()[0]
        
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
    x0_base = np.random.uniform(x0_min, x0_max)
    y0_base = np.random.uniform(y0_min, y0_max)
    
    print(f"\n  Trajetória Base Selecionada:")
    print(f"    x0 = {x0_base:.3f}")
    print(f"    y0 = {y0_base:.3f}")
    print(f"    Distância do equilíbrio: {np.sqrt((x0_base - x_eq)**2 + (y0_base - y_eq)**2):.3f}")
    print(f"    Tempo máximo: {tempo_maximo:.3f} s")
    
    num_variacoes = 5
    variacoes = []
    
    # geração aleatória dentro dos limites
    np.random.seed(seed)
    x0_variacoes = np.random.uniform(x0_min, x0_max, num_variacoes)
    y0_variacoes = np.random.uniform(y0_min, y0_max, num_variacoes)
        
    print(f"\n  Gerando {num_variacoes} novas condições iniciais:")
    for i in range(num_variacoes):
        distancia = np.sqrt((x0_variacoes[i] - x_eq)**2 + (y0_variacoes[i] - y_eq)**2)
        variacoes.append({
            'x0': x0_variacoes[i],
            'y0': y0_variacoes[i],
            'distancia_eq': distancia
        })
        print(f"    Caso {i+1}: x0={x0_variacoes[i]:.3f}, y0={y0_variacoes[i]:.3f}, distância={distancia:.3f}")
    
    todas_previsoes = []
    todos_reais_interpolados = []
    
    tempos_lista = []
    presas_previstas_lista = []
    predadores_previstos_lista = []
    presas_reais_lista = []
    predadores_reais_lista = []
    casos_info_lista = []
    dados_interpolados = []
    
    # para cada nova condição inicial, faz a previsão
    for var_idx, var in enumerate(variacoes):
        x0_novo = var['x0']
        y0_novo = var['y0']
        
        # entrada para o modelo: x0, y0
        X_novo = np.array([[x0_novo, y0_novo]], dtype=np.float32)
        
        # normaliza e faz previsão
        X_novo_scaled = scaler_X.transform(X_novo)
        X_tensor = torch.tensor(X_novo_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()
        
        # desnormaliza a trajetória completa
        pred = scaler_y.inverse_transform(pred_scaled)
        
        # separa presas e predadores da trajetória completa
        # saída: [x0, y0, x1, y1, ..., xN, yN]
        presas_previstas = pred[0, 0::2]  # presas (índices pares)
        predadores_previstos = pred[0, 1::2]  # predadores (índices ímpares)
        
        # verifica se o número de pontos coincide com os tempos
        if len(presas_previstas) != len(tempos_unicos):
            print(f"  AVISO: Ajustando tempos para {len(presas_previstas)} pontos")
            tempos_ajustados = np.linspace(tempos_unicos[0], tempos_unicos[-1], len(presas_previstas))
        else:
            tempos_ajustados = tempos_unicos
        
        osc = OsciladorLotkaVolterra(
            taxas_crescimento=[a],
            taxas_mortalidade=[c],
            taxas_predacao=[b],
            taxas_eficiencia=[d],
            device='cpu'
        )
        
        cond_curta = torch.tensor([[x0_novo, y0_novo]], dtype=torch.float32)
        solucao_curta = osc.resolve_multi_condicoes_sistemas(
            condicoes_iniciais=cond_curta,
            t_final=tempo_maximo,
            dt=dt_interpolacao
        )
        
        presas_reais = solucao_curta['presas'][:, 0, 0]
        predadores_reais = solucao_curta['predadores'][:, 0, 0]
        tempos_reais = solucao_curta['tempo']
        
        if len(tempos_reais) != len(tempos_ajustados):
            interp_presas = interp1d(tempos_reais, presas_reais, kind='linear', fill_value='extrapolate')
            interp_predadores = interp1d(tempos_reais, predadores_reais, kind='linear', fill_value='extrapolate')
            presas_reais_ajustados = interp_presas(tempos_ajustados)
            predadores_reais_ajustados = interp_predadores(tempos_ajustados)
        else:
            presas_reais_ajustados = presas_reais
            predadores_reais_ajustados = predadores_reais
        
        # métricas globais (ponto a ponto para compatibilidade)
        pred_pontos = np.column_stack([presas_previstas, predadores_previstos])
        real_pontos = np.column_stack([presas_reais_ajustados, predadores_reais_ajustados])
        
        todas_previsoes.append(pred_pontos)
        todos_reais_interpolados.append(real_pontos)
        
        tempos_lista.append(tempos_ajustados)
        presas_previstas_lista.append(presas_previstas)
        predadores_previstos_lista.append(predadores_previstos)
        presas_reais_lista.append(presas_reais_ajustados)
        predadores_reais_lista.append(predadores_reais_ajustados)
        
        cor = CORES_PALETA[var_idx % len(CORES_PALETA)]
        
        casos_info_lista.append({
            'x0': x0_novo,
            'y0': y0_novo,
            'cor': cor,
            'variation_id': var_idx
        })
        
        for k in range(len(tempos_ajustados)):
            dados_interpolados.append({
                'variacao_id': var_idx,
                'x0': x0_novo,
                'y0': y0_novo,
                'taxa_crescimento_a': a,
                'taxa_predacao_b': b,
                'taxa_mortalidade_c': c,
                'taxa_eficiencia_d': d,
                'presas_eq': x_eq,
                'predadores_eq': y_eq,
                'tempo': tempos_ajustados[k],
                'presas_real': presas_reais_ajustados[k],
                'predadores_real': predadores_reais_ajustados[k],
                'presas_previsto_mlp': pred_pontos[k, 0],
                'predadores_previsto_mlp': pred_pontos[k, 1],
                'erro_presas': pred_pontos[k, 0] - presas_reais_ajustados[k],
                'erro_predadores': pred_pontos[k, 1] - predadores_reais_ajustados[k],
                'erro_abs_presas': abs(pred_pontos[k, 0] - presas_reais_ajustados[k]),
                'erro_abs_predadores': abs(pred_pontos[k, 1] - predadores_reais_ajustados[k]),
                'erro_rel_presas_pct': (abs(pred_pontos[k, 0] - presas_reais_ajustados[k]) / (abs(presas_reais_ajustados[k]) + 1e-6)) * 100,
                'erro_rel_predadores_pct': (abs(pred_pontos[k, 1] - predadores_reais_ajustados[k]) / (abs(predadores_reais_ajustados[k]) + 1e-6)) * 100,
            })
    
    if len(todas_previsoes) == 0:
        print("  ERRO: Nenhuma previsão realizada")
        return pd.DataFrame()
    
    predictions_all = np.vstack(todas_previsoes)
    y_true_all = np.vstack(todos_reais_interpolados)
    
    rmse_presas = float(np.sqrt(mean_squared_error(y_true_all[:, 0], predictions_all[:, 0])))
    rmse_predadores = float(np.sqrt(mean_squared_error(y_true_all[:, 1], predictions_all[:, 1])))
    r2_presas = float(r2_score(y_true_all[:, 0], predictions_all[:, 0]))
    r2_predadores = float(r2_score(y_true_all[:, 1], predictions_all[:, 1]))
    
    print(f"\n  Total de pontos previstos: {len(predictions_all)}")
    print(f"  RMSE Presas (vs solução RK4): {rmse_presas:.6f}")
    print(f"  RMSE Predadores (vs solução RK4): {rmse_predadores:.6f}")
    print(f"  R² Presas (vs solução RK4): {r2_presas:.4f}")
    print(f"  R² Predadores (vs solução RK4): {r2_predadores:.4f}")
    
    # ============================================
    # GRÁFICO 1: Real vs Previsto
    # ============================================
    
    fig1 = cria_grafico_interpolacao_pontual_mlp(
        predictions=predictions_all,
        y_true=y_true_all,
        titulo="Novas Condições Iniciais: RK4 vs MLP - Lotka-Volterra"
    )
    
    fig1.write_html(grafico_interpolacao_trajetorias)
    
    # ============================================
    # GRÁFICO 2: Espaço de Fases
    # ============================================
    
    y_presas_true = y_true_all[:, 0].reshape(-1, 1)
    y_predadores_true = y_true_all[:, 1].reshape(-1, 1)
    y_presas_pred = predictions_all[:, 0].reshape(-1, 1)
    y_predadores_pred = predictions_all[:, 1].reshape(-1, 1)
    
    fig2 = cria_grafico_interpolacao_pontual_espaco_fases(
        y_pos_true=y_presas_true,
        y_vel_true=y_predadores_true,
        y_pos_pred=y_presas_pred,
        y_vel_pred=y_predadores_pred,
        titulo="Novas Condições Iniciais: MLP vs RK4 - Espaço de Fases - Lotka-Volterra"
    )
    
    fig2.write_html(grafico_interpolacao_trajetorias_espaco_fases)
    
    # ============================================
    # GRÁFICO 3: Presas e Predadores vs Tempo
    # ============================================
    
    fig3 = cria_grafico_interpolacao_pontual_completo(
        tempos_lista=tempos_lista,
        posicoes_previstas_lista=presas_previstas_lista,
        velocidades_previstas_lista=predadores_previstos_lista,
        posicoes_reais_lista=presas_reais_lista,
        velocidades_reais_lista=predadores_reais_lista,
        casos_info=casos_info_lista,
        titulo="Novas Condições Iniciais: MLP vs RK4 - Presas e Predadores vs Tempo - Lotka-Volterra"
    )
    
    fig3.write_html(grafico_interpolacao_trajetorias_temporal)
    
    fig1.show()
    fig2.show()
    fig3.show()
    
    df_interpolado = pd.DataFrame(dados_interpolados)
    
    df_interpolado.attrs['rmse_presas'] = rmse_presas
    df_interpolado.attrs['rmse_predadores'] = rmse_predadores
    df_interpolado.attrs['r2_presas'] = r2_presas
    df_interpolado.attrs['r2_predadores'] = r2_predadores
    df_interpolado.attrs['total_pontos'] = len(predictions_all)
    df_interpolado.attrs['num_variacoes'] = num_variacoes
    df_interpolado.attrs['num_tempos'] = len(tempos_ajustados)
    df_interpolado.attrs['taxa_crescimento_a'] = a
    df_interpolado.attrs['taxa_predacao_b'] = b
    df_interpolado.attrs['taxa_mortalidade_c'] = c
    df_interpolado.attrs['taxa_eficiencia_d'] = d
    df_interpolado.attrs['presas_eq'] = x_eq
    df_interpolado.attrs['predadores_eq'] = y_eq
    df_interpolado.attrs['x0_min'] = x0_min
    df_interpolado.attrs['x0_max'] = x0_max
    df_interpolado.attrs['y0_min'] = y0_min
    df_interpolado.attrs['y0_max'] = y0_max
    
    # ========================================================================
    # GRÁFICO 4: Trajetória Base e Novas Condições Iniciais no Espaço de Fases
    # ========================================================================
    
    # gera a trajetória base via RK4
    osc_base = OsciladorLotkaVolterra(
        taxas_crescimento=[a],
        taxas_mortalidade=[c],
        taxas_predacao=[b],
        taxas_eficiencia=[d],
        device='cpu'
    )
    
    cond_base = torch.tensor([[x0_base, y0_base]], dtype=torch.float32)
    sol_base = osc_base.resolve_multi_condicoes_sistemas(
        condicoes_iniciais=cond_base,
        t_final=tempo_maximo,
        dt=dt_interpolacao
    )
    
    presas_base = sol_base['presas'][:, 0, 0]
    predadores_base = sol_base['predadores'][:, 0, 0]
    tempos_base = sol_base['tempo']
    
    if len(tempos_base) != len(tempos_ajustados):
        interp_presas_base = interp1d(tempos_base, presas_base, kind='linear', fill_value='extrapolate')
        interp_predadores_base = interp1d(tempos_base, predadores_base, kind='linear', fill_value='extrapolate')
        presas_base = interp_presas_base(tempos_ajustados)
        predadores_base = interp_predadores_base(tempos_ajustados)
    
    novas_trajetorias_para_grafico = []
    
    for var_idx in range(num_variacoes):
        mask_var = df_interpolado['variacao_id'] == var_idx
        dados_var = df_interpolado[mask_var].sort_values('tempo')
        
        if len(dados_var) > 0:
            x0_var = dados_var['x0'].iloc[0]
            y0_var = dados_var['y0'].iloc[0]
            
            novas_trajetorias_para_grafico.append({
                'variacao_id': var_idx,
                'posicoes': dados_var['presas_previsto_mlp'].values,
                'velocidades': dados_var['predadores_previsto_mlp'].values,
                'x0': x0_var,
                'v0': y0_var
            })
    
    casos_info_grafico = {
        'x0_base': x0_base,
        'v0_base': y0_base
    }
    
    fig4 = cria_grafico_interpolacao_trajetorias_espaco_fases(
        trajetoria_base_pos=presas_base,
        trajetoria_base_vel=predadores_base,
        novas_trajetorias_lista=novas_trajetorias_para_grafico,
        casos_info=casos_info_grafico,
        titulo="Trajetória Base vs Novas Condições Iniciais no Espaço de Fases - Lotka-Volterra"
    )
    
    grafico_novas_trajetorias = f"{output_dir}/trajetoria_base_vs_novas_condicoes.html"
    fig4.write_html(grafico_novas_trajetorias)
    
    fig4.show()

    return df_interpolado