# nodes saída: trajetória completa; entrada: [x0, v0]

"""
Nodes do pipeline MLP para previsão de trajetórias completas.
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
from typing import Dict, Any, Tuple
from .model import MLP
from oscilador_harmonico.utils import (
    CORES_PALETA,
    cria_grafico_real_previsto_mlp,
    cria_grafico_distribuicao_amplitudes,
    cria_grafico_pesos_por_amplitude,
    cria_grafico_distribuicao_dados,
    cria_grafico_historico_treinamento,
    cria_grafico_previsoes_espaco_fases,
    cria_grafico_interpolacao_completo,
    cria_grafico_interpolacao_espaco_fases,
    cria_grafico_interpolacao_pontual_mlp,
    cria_grafico_interpolacao_pontual_espaco_fases,
    cria_grafico_interpolacao_pontual_completo,
    cria_grafico_interpolacao_entre_trajetorias_espaco_fases,
    cria_grafico_interpolacao_trajetorias_espaco_fases,
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
    
    Entrada: [x0, v0]
    Saída: Trajetória completa [x_0, v_0, x_1, v_1, ..., x_N, v_N]
    
    O tempo é usado apenas para organizar os pontos da trajetória,
    mas não é uma feature de entrada.
    
    A divisão dos dados é feita considerando apenas as trajetórias mais internas
    no espaço de fases (menor amplitude). As trajetórias externas são separadas
    para avaliação posterior.
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
    
    if base_oscilador[['x0', 'v0']].isnull().any().any():
        print("  AVISO: Valores NaN detectados nas colunas numéricas!")
        base_oscilador = base_oscilador.dropna(subset=['x0', 'v0'])
    
    frequencia_angular_unica = base_oscilador['frequencia_angular'].iloc[0] if len(base_oscilador) > 0 else 5.0
    omega = frequencia_angular_unica
    
    print(f"\n=== BASE DE DADOS ===")
    print(f"  Frequência angular do sistema: {frequencia_angular_unica} rad/s")
    print(f"  Total de linhas da base: {len(base_oscilador)}")
    
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
    
    # obtém o número de pontos por trajetória a partir dos dados
    # cada trajetória tem a mesma quantidade de pontos (definido pela simulação)
    primeiro_grupo = base_oscilador[base_oscilador['id_trajetoria'] == trajetorias_unicas[0]].sort_values('tempo')
    num_timesteps = len(primeiro_grupo)
    
    print(f"\n=== PREPARAÇÃO DOS DADOS ===")
    
    # ============================================
    # SELEÇÃO DAS TRAJETÓRIAS MAIS INTERNAS
    # ============================================
    
    # calcula amplitude para cada trajetória
    amplitudes = {}
    for traj_id in trajetorias_unicas:
        grupo = base_oscilador[base_oscilador['id_trajetoria'] == traj_id].iloc[0]
        x0 = grupo['x0']
        v0 = grupo['v0']
        amplitude = np.sqrt(x0**2 + (v0 / omega)**2)
        amplitudes[traj_id] = amplitude
    
    # ordena trajetórias por amplitude (internas primeiro)
    trajetorias_ordenadas = sorted(amplitudes.items(), key=lambda x: x[1])
    trajetorias_ids_ordenadas = [t[0] for t in trajetorias_ordenadas]
    amplitudes_ordenadas = [t[1] for t in trajetorias_ordenadas]
    
    # seleciona apenas as 70% mais internas para treino/validação/teste
    n_traj = len(trajetorias_ids_ordenadas)
    n_internas = int(0.7 * n_traj)
    trajetorias_internas = trajetorias_ids_ordenadas[:n_internas]
    trajetorias_externas = trajetorias_ids_ordenadas[n_internas:]
    
    amplitude_limite_internas = amplitudes_ordenadas[n_internas - 1] if n_internas > 0 else 0
    
    print(f"\n  Trajetórias internas: {len(trajetorias_internas)} trajetórias")
    print(f"    Amplitude ≤ {amplitude_limite_internas:.4f} m")
    print(f"  Trajetórias externas: {len(trajetorias_externas)} trajetórias")
    print(f"    Amplitude > {amplitude_limite_internas:.4f} m")
    
    # =========================================================
    # DIVISÃO DOS DADOS (APENAS TRAJETÓRIAS INTERNAS)
    # =========================================================
    
    X_list = []  # [x0, v0] para cada trajetória
    y_list = []  # trajetória completa intercalada para cada trajetória
    tempos_list = []  # tempos para referência
    trajetorias_internas_list = []  # lista para manter rastreamento
    
    for traj_id in trajetorias_internas:
        grupo = base_oscilador[base_oscilador['id_trajetoria'] == traj_id].sort_values('tempo')
        
        # verifica se todos os pontos estão presentes
        if len(grupo) != num_timesteps:
            print(f"  AVISO: Trajetória {traj_id} tem {len(grupo)} pontos, pulando...")
            continue
        
        # entrada: [x0, v0]
        x0 = grupo['x0'].iloc[0]
        v0 = grupo['v0'].iloc[0]
        X_list.append([x0, v0])
        
        # saída: trajetória completa intercalada [x0, v0, x1, v1, ..., xN, vN]
        posicoes = grupo['posicao'].values
        velocidades = grupo['velocidade'].values
        trajetoria = np.column_stack([posicoes, velocidades]).flatten()
        y_list.append(trajetoria)
        
        # tempos para referência
        tempos_list.append(grupo['tempo'].values)
        trajetorias_internas_list.append(traj_id)
    
    X_raw = np.array(X_list, dtype=np.float32)
    y_raw = np.array(y_list, dtype=np.float32)
    tempos_referencia = np.array(tempos_list[0]) if tempos_list else np.array([])
    
    print(f"\n  Trajetórias internas válidas: {len(X_raw)}")
    print(f"  Dimensão entrada: {X_raw.shape[1]} (x0, v0)")
    print(f"  Dimensão saída: {y_raw.shape[1]} (2N)")
    print(f"  Nós de saída do modelo por trajetória: {num_timesteps}")
    
    # treino, validação e teste (70-20-10) - apenas trajetórias internas
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
    
    trajetorias_train = np.array(trajetorias_internas_list)[train_indices]
    trajetorias_val = np.array(trajetorias_internas_list)[val_indices]
    trajetorias_test = np.array(trajetorias_internas_list)[test_indices]
    
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
    
    print(f"\n  Dimensão entrada: {input_dim} (x0, v0)")
    print(f"  Dimensão saída: {output_dim} (2N)")
    
    # tempos de referência e número de timesteps
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
    
    Agora considera apenas as trajetórias mais internas (70%) para a divisão,
    mantendo as externas separadas para visualização.
    
    Args:
        base_oscilador: DataFrame com a base consolidada
        parameters: Parâmetros do pipeline
    """
    
    data_version = parameters.get('data_version', 'default_v1')
    omega = parameters.get('intervals', {}).get('omega', 5.0)
    
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
    # CÁLCULO DAS AMPLITUDES
    # ============================================
    
    # calcula amplitude para cada trajetória
    amplitudes = {}
    for traj_id in trajetorias_unicas:
        grupo = base_oscilador[base_oscilador['id_trajetoria'] == traj_id].iloc[0]
        x0 = grupo['x0']
        v0 = grupo['v0']
        amplitude = np.sqrt(x0**2 + (v0 / omega)**2)
        amplitudes[traj_id] = amplitude
    
    # ordena de forma ascendente as trajetórias por amplitude
    trajetorias_ordenadas = sorted(amplitudes.items(), key=lambda x: x[1])
    trajetorias_ids_ordenadas = [t[0] for t in trajetorias_ordenadas]
    amplitudes_ordenadas = [t[1] for t in trajetorias_ordenadas]
    
    # seleciona apenas as 70% mais internas
    n_traj = len(trajetorias_ids_ordenadas)
    n_internas = int(0.7 * n_traj)
    trajetorias_internas = trajetorias_ids_ordenadas[:n_internas]
    trajetorias_externas = trajetorias_ids_ordenadas[n_internas:]
    
    amplitude_limite_internas = amplitudes_ordenadas[n_internas - 1] if n_internas > 0 else 0
    
    print(f"\n=== DISTRIBUIÇÃO DAS TRAJETÓRIAS POR AMPLITUDE ===")
    print(f"  Amplitude mínima: {amplitudes_ordenadas[0]:.4f} m")
    print(f"  Amplitude máxima: {amplitudes_ordenadas[-1]:.4f} m")
    print(f"  Amplitude mediana: {amplitudes_ordenadas[n_traj//2]:.4f} m")
    print(f"  Amplitude limite trajetórias internas: {amplitude_limite_internas:.4f} m")
    print(f"\n  Trajetórias internas: {len(trajetorias_internas)} trajetórias")
    print(f"  Trajetórias externas: {len(trajetorias_externas)} trajetórias")
    
    # ============================================
    # GRÁFICO: Distribuição das Amplitudes
    # ============================================
    
    fig_amp = cria_grafico_distribuicao_amplitudes(
        amplitudes=np.array(amplitudes_ordenadas),
        amplitude_limite_internas=amplitude_limite_internas,
        omega=omega,
        titulo="Distribuição das Amplitudes das Trajetórias"
    )
    
    fig_amp.write_html(grafico_distribuicao_amplitudes)
    fig_amp.show()
    
    # ========================================================
    # DIVISÃO DOS DADOS (APENAS TRAJETÓRIAS INTERNAS)
    # ========================================================
    
    # divide as trajetórias internas em treino, validação e teste (70-20-10)
    trajetorias_train, trajetorias_temp = train_test_split(
        trajetorias_internas, test_size=0.30, random_state=42
    )
    trajetorias_val, trajetorias_test = train_test_split(
        trajetorias_temp, test_size=0.3333, random_state=42
    )
    
    print(f"\n=== DIVISÃO DOS DADOS INTERNOS ===")
    print(f"  Trajetórias de treino: {len(trajetorias_train)}")
    print(f"  Trajetórias de validação: {len(trajetorias_val)}")
    print(f"  Trajetórias de teste: {len(trajetorias_test)}")
        
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
    
    # ============================================
    # GRÁFICO: Distribuição no Espaço de Fases
    # ============================================
    
    fig = cria_grafico_distribuicao_dados(
        y_pos_train=y_pos_train,
        y_vel_train=y_vel_train,
        y_pos_val=y_pos_val,
        y_vel_val=y_vel_val,
        y_pos_test=y_pos_test,
        y_vel_test=y_vel_test,
        titulo="Distribuição dos Dados no Espaço de Fases - Apenas Trajetórias Internas (70%)"
    )
    
    fig.write_html(grafico_distribuicao_dados) 
    fig.show()
    
    return None


def cria_modelo_mlp_node(input_dim: int, output_dim: int, parameters: Dict[str, Any]) -> nn.Module:
    """Cria o modelo MLP para previsão de trajetórias completas."""

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
    print(f"  Dimensão entrada: {input_dim} (x0, v0)")
    print(f"  Camadas ocultas: {hidden_dims}")
    print(f"  Dimensão saída: {output_dim}  (2N)")
    print(f"  Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"  Função de ativação: {activation.capitalize()}")
    
    return model


def treina_mlp_node(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    parameters: Dict[str, Any],
    base_oscilador: pd.DataFrame = None,
    trajetorias_train: np.ndarray = None
) -> Tuple[nn.Module, Dict]:
    """Treina o modelo MLP para prever trajetórias completas.
    
    Se base_oscilador e trajetorias_train forem fornecidos, aplica pesos
    inversos à amplitude para dar mais importância às trajetórias internas.
    Estratégias combinadas:
    1. Weighted Sampling: amostragem ponderada no DataLoader
    2. Weighted Loss: pesos na função de custo
    """

    mlp_config = parameters.get('mlp', {})
    
    batch_size = mlp_config.get('batch_size', 512)
    epochs = mlp_config.get('epochs', 500)
    learning_rate = mlp_config.get('learning_rate', 0.005)
    weight_decay = 0.0001
    
    exp_name = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    omega = parameters.get('intervals', {}).get('omega', 5.0)
    
    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    grafico_historico_loss = f"{output_dir}/historico_treinamento_loss.html"
    grafico_pesos = f"{output_dir}/distribuicao_pesos_treino.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    model = model.to(device)

    use_weighted_sampling = False
    use_weighted_loss = False
    sampler = None
    weights_por_amostra = None
    
    if base_oscilador is not None and trajetorias_train is not None:
        # calcula amplitude para cada trajetória de treino
        amplitudes = {}
        for traj_id in trajetorias_train:
            grupo = base_oscilador[base_oscilador['id_trajetoria'] == traj_id].iloc[0]
            x0 = grupo['x0']
            v0 = grupo['v0']
            amplitude = np.sqrt(x0**2 + (v0 / omega)**2)
            amplitudes[traj_id] = amplitude
        
        amplitudes_array = np.array(list(amplitudes.values()))

        amplitude_min = amplitudes_array.min()
        amplitude_max = amplitudes_array.max()
        amplitudes_norm = (amplitudes_array - amplitude_min) / (amplitude_max - amplitude_min + 1e-8)
        pesos_trajetorias = 1.0 / (amplitudes_norm + 0.01)
        
        print("\n=== DISTRIBUIÇÃO DOS PESOS POR AMPLITUDE ===")
        print(f"  Amplitude mínima: {amplitudes_array.min():.4f} m")
        print(f"  Amplitude máxima: {amplitudes_array.max():.4f} m")
        print(f"  Peso médio: {pesos_trajetorias.mean():.4f}")
        print(f"  Peso mínimo: {pesos_trajetorias.min():.4f}")
        print(f"  Peso máximo: {pesos_trajetorias.max():.4f}")
                
        fig_pesos = cria_grafico_pesos_por_amplitude(
            amplitudes=amplitudes_array,
            pesos=pesos_trajetorias,
            omega=omega,
            titulo="Distribuição dos Pesos por Amplitude"
        )
        
        fig_pesos.write_html(grafico_pesos)
        fig_pesos.show()
        
        # ============================================
        # ESTRATÉGIA 1: Weighted Sampling
        # ============================================
        use_weighted_sampling = True
        traj_peso_map = dict(zip(trajetorias_train, pesos_trajetorias))
        weights = np.array([traj_peso_map[traj_id] for traj_id in trajetorias_train], dtype=np.float64)
        
        # normaliza os pesos para somar 1
        weights = weights / weights.sum()
        
        print(f"\n  Weighted Sampling: {len(weights)} trajetórias com pesos")
        print(f"    Peso médio: {weights.mean():.6f}")
        print(f"    Peso mínimo: {weights.min():.6f}")
        print(f"    Peso máximo: {weights.max():.6f}")

        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        
        # ============================================
        # ESTRATÉGIA 2: Weighted Loss
        # ============================================
        use_weighted_loss = True
        # cria pesos para cada ponto da trajetória
        weights_por_amostra = []
        for traj_id in trajetorias_train:
            peso = traj_peso_map[traj_id]
            # obtém o número de pontos desta trajetória
            grupo = base_oscilador[base_oscilador['id_trajetoria'] == traj_id]
            n_pontos = len(grupo)
            weights_por_amostra.extend([peso] * n_pontos)
        
        weights_por_amostra = np.array(weights_por_amostra, dtype=np.float32)
        # normaliza para ter média 1 (não alterar a escala da loss)
        weights_por_amostra = weights_por_amostra / weights_por_amostra.mean()
        
        print(f"\n  Weighted Loss: {len(weights_por_amostra)} amostras com pesos")
        print(f"    Peso médio: {weights_por_amostra.mean():.6f}")
        print(f"    Peso mínimo: {weights_por_amostra.min():.6f}")
        print(f"    Peso máximo: {weights_por_amostra.max():.6f}")
        
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # configura o dataloader com ou sem amostragem ponderada
    if use_weighted_sampling and sampler is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        print(f"\n  Usando WeightedRandomSampler para balancear trajetórias internas")
        print(f"    Número de trajetórias: {len(weights)}")
        print(f"    Batch size: {batch_size}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss(reduction='none')  # reduction='none' para aplicar pesos
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    print("\n=== INICIANDO TREINAMENTO DO MLP ===")
    print(f"  Entrada: (x0, v0) -> Saída: (trajetória completa)")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Função loss: RMSE (Root Mean Squared Error)")
    
    if use_weighted_sampling:
        print(f"  Estratégia 1: Weighted Sampling (pesos inversos à amplitude)")
        print(f"    Trajetórias internas têm maior probabilidade de serem amostradas")
    
    if use_weighted_loss:
        print(f"  Estratégia 2: Weighted Loss (pesos inversos à amplitude)")
        print(f"    Erros em trajetórias internas são penalizados com maior peso")
    
    for epoch in range(epochs):
        # treino
        model.train()
        epoch_train_loss = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            # ============================================
            # APLICAÇÃO DA WEIGHTED LOSS
            # ============================================
            if use_weighted_loss and weights_por_amostra is not None:
                # obtém os índices das amostras no batch
                if use_weighted_sampling:
                    # os índices são amostrados aleatoriamente
                    # usamos os pesos correspondentes a cada índice
                    batch_indices = batch_idx * batch_size + np.arange(len(batch_X))
                    batch_indices = batch_indices % len(weights_por_amostra)
                    batch_weights = torch.tensor(
                        weights_por_amostra[batch_indices], 
                        dtype=torch.float32
                    ).to(device)
                else:
                    # sem sampler, usamos os pesos na ordem do dataset
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(weights_por_amostra))
                    batch_weights = torch.tensor(
                        weights_por_amostra[start_idx:end_idx], 
                        dtype=torch.float32
                    ).to(device)
                
                # loss ponderada
                loss_per_element = (predictions - batch_y) ** 2
                weighted_loss = (loss_per_element * batch_weights.reshape(-1, 1)).mean()
                loss = weighted_loss
            else:
                # loss sem pesos
                loss = nn.MSELoss()(predictions, batch_y)
            
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
                loss = nn.MSELoss()(predictions, batch_y)
                epoch_val_loss += loss.item()
        
        epoch_val_loss /= len(val_loader)
        
        history['train_loss'].append(float(epoch_train_loss))
        history['val_loss'].append(float(epoch_val_loss))
        
        scheduler.step(epoch_val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
        
    titulo_historico = "Evolução da Função de Custo durante o Treinamento do MLP"
    if use_weighted_sampling and use_weighted_loss:
        titulo_historico += " (Weighted Sampling + Weighted Loss)"
    elif use_weighted_sampling:
        titulo_historico += " (Weighted Sampling)"
    elif use_weighted_loss:
        titulo_historico += " (Weighted Loss)"
    
    fig = cria_grafico_historico_treinamento(
        history=history,
        titulo=titulo_historico
    )
    
    fig.write_html(grafico_historico_loss)
    
    print(f"\n=== TREINAMENTO CONCLUÍDO ===")
    print(f"  Loss final de treino: {history['train_loss'][-1]:.6f}")
    print(f"  Loss final de validação: {history['val_loss'][-1]:.6f}")
    
    if use_weighted_sampling and use_weighted_loss:
        print(f"\n  Estratégias utilizadas: Weighted Sampling + Weighted Loss")
        print(f"    - Amostragem ponderada para balancear o dataset")
        print(f"    - Função de custo ponderada para dar mais peso às trajetórias internas")
    elif use_weighted_sampling:
        print(f"\n  Estratégia utilizada: Weighted Sampling (trajetórias internas priorizadas)")
    elif use_weighted_loss:
        print(f"\n  Estratégia utilizada: Weighted Loss (trajetórias internas priorizadas)")
    
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
    Avalia o modelo MLP nos dados de validação e teste.
    
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
    
    # separa posições e velocidades das trajetórias
    pos_pred_val = predictions_val[:, 0::2]
    vel_pred_val = predictions_val[:, 1::2]
    pos_true_val = y_val_original[:, 0::2]
    vel_true_val = y_val_original[:, 1::2]
    
    pos_pred_test = predictions_test[:, 0::2]
    vel_pred_test = predictions_test[:, 1::2]
    pos_true_test = y_test_original[:, 0::2]
    vel_true_test = y_test_original[:, 1::2]
    
    # avalia ponto a ponto
    rmse_pos_val = float(np.sqrt(mean_squared_error(pos_true_val.flatten(), pos_pred_val.flatten())))
    rmse_vel_val = float(np.sqrt(mean_squared_error(vel_true_val.flatten(), vel_pred_val.flatten())))
    r2_pos_val = float(r2_score(pos_true_val.flatten(), pos_pred_val.flatten()))
    r2_vel_val = float(r2_score(vel_true_val.flatten(), vel_pred_val.flatten()))
    
    rmse_pos_test = float(np.sqrt(mean_squared_error(pos_true_test.flatten(), pos_pred_test.flatten())))
    rmse_vel_test = float(np.sqrt(mean_squared_error(vel_true_test.flatten(), vel_pred_test.flatten())))
    r2_pos_test = float(r2_score(pos_true_test.flatten(), pos_pred_test.flatten()))
    r2_vel_test = float(r2_score(vel_true_test.flatten(), vel_pred_test.flatten()))

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
    print(f"  RMSE Posição Validação: {rmse_pos_val:.6f} m")
    print(f"  RMSE Velocidade Validação: {rmse_vel_val:.6f} m/s")
    print(f"  R² Posição Validação: {r2_pos_val:.4f}")
    print(f"  R² Velocidade Validação: {r2_vel_val:.4f}")
    print(f"  RMSE Posição Teste: {rmse_pos_test:.6f} m")
    print(f"  RMSE Velocidade Teste: {rmse_vel_test:.6f} m/s")
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
        
        # intercala posições e velocidades
        pred_pos = pred_traj[0::2]
        pred_vel = pred_traj[1::2]
        true_pos = true_traj[0::2]
        true_vel = true_traj[1::2]
        
        for i in range(len(pred_pos)):
            predictions_flat.append([pred_pos[i], pred_vel[i]])
            y_true_flat.append([true_pos[i], true_vel[i]])
    
    predictions_flat = np.array(predictions_flat)
    y_true_flat = np.array(y_true_flat)
    
    fig = cria_grafico_real_previsto_mlp(
        predictions=predictions_flat,
        y_true=y_true_flat,
        titulo="Real vs Previsto (Dados de Teste)"
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
    Node: Visualiza as previsões do modelo no espaço de fases (Posição vs Velocidade).
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
    y_pos_true_list = []
    y_vel_true_list = []
    y_pos_pred_list = []
    y_vel_pred_list = []
    
    for idx in indices_vis:
        pred_traj = predictions[idx]
        true_traj = y_test_original[idx]
        
        pred_pos = pred_traj[0::2]
        pred_vel = pred_traj[1::2]
        true_pos = true_traj[0::2]
        true_vel = true_traj[1::2]
        
        y_pos_true_list.extend(true_pos)
        y_vel_true_list.extend(true_vel)
        y_pos_pred_list.extend(pred_pos)
        y_vel_pred_list.extend(pred_vel)
    
    y_pos_true = np.array(y_pos_true_list).reshape(-1, 1)
    y_vel_true = np.array(y_vel_true_list).reshape(-1, 1)
    y_pos_pred = np.array(y_pos_pred_list).reshape(-1, 1)
    y_vel_pred = np.array(y_vel_pred_list).reshape(-1, 1)
    
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
        titulo="Previsões do Modelo no Espaço de Fases"
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
    Nota: A frequência angular é fixa para todos os casos.
    
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
    
    grafico_interpolacao_completa = f"{output_dir}/interpolacao_avulsa_v_x_vs_t.html"
    grafico_interpolacao_espaco_fases = f"{output_dir}/interpolacao_avulsa_espaco_fases.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    intervals = parameters.get('intervals', {})
    omega_fixo = intervals.get('omega', 5.0)
    
    if tempos_referencia is None:
        sim_params = parameters.get('simulation', {})
        dt = sim_params.get('dt', 0.01)
        T = 2 * np.pi / omega_fixo
        tempos_referencia = np.arange(0, T + dt, dt)
        print(f"  Tempos de referência criados: {len(tempos_referencia)} pontos")
    else:
        print(f"\n  Nós de saída do modelo por trajetória: {len(tempos_referencia)} pontos")
    
    num_timesteps = len(tempos_referencia)
    
    # casos de teste para interpolação (apenas condições iniciais variam)
    casos_teste = [
        {
            "nome": "Caso 1",
            "x0": 0.3,
            "v0": 0.0,
            "omega": 5.1,
            "cor": CORES_PALETA[0]
        },
        {
            "nome": "Caso 2",
            "x0": -0.3,
            "v0": 1.0,
            "omega": omega_fixo,
            "cor": CORES_PALETA[1]
        },
        {
            "nome": "Caso 3",
            "x0": 0.5,
            "v0": -1.0,
            "omega": omega_fixo,
            "cor": CORES_PALETA[2]
        },
        {
            "nome": "Caso 4",
            "x0": -0.1,
            "v0": 0.5,
            "omega": omega_fixo,
            "cor": CORES_PALETA[3]
        },
    ]
    
    # gera os nomes das legendas dinamicamente
    for caso in casos_teste:
        T = 2 * np.pi / caso["omega"] if caso["omega"] > 0 else 2 * np.pi / omega_fixo
        caso["t_final"] = T
        caso["nome_legenda"] = (
            f"{caso['nome']}: x0={caso['x0']:.1f} m, "
            f"v0={caso['v0']:.1f} m/s, "
            f"ω={caso['omega']:.1f} rad/s, "
            f"T={T:.2f} s"
        )
    
    tempos_lista = []
    posicoes_lista = []
    velocidades_lista = []
    
    for caso in casos_teste:
        # verifica se o número de timesteps é compatível com o modelo
        if len(tempos_referencia) != num_timesteps:
            print(f"  AVISO: {caso['nome']} - Ajustando tempos para {num_timesteps} pontos")
            tempos = np.linspace(tempos_referencia[0], tempos_referencia[-1], num_timesteps)
        else:
            tempos = tempos_referencia
        
        # entrada: [x0, v0]
        X_caso = np.array([[caso["x0"], caso["v0"]]], dtype=np.float32)
        
        # normaliza a entrada
        X_caso_scaled = scaler_X.transform(X_caso)
        X_tensor = torch.tensor(X_caso_scaled, dtype=torch.float32).to(device)
        
        # previsão: trajetória completa
        with torch.no_grad():
            predictions_scaled = model(X_tensor).cpu().numpy()
        
        # desnormaliza a trajetória completa
        predictions = scaler_y.inverse_transform(predictions_scaled)
        
        # separa posições e velocidades da trajetória completa
        posicoes = predictions[0, 0::2]  # posições (índices pares)
        velocidades = predictions[0, 1::2]  # velocidades (índices ímpares)
        
        # se os tempos não têm o mesmo tamanho, ajusta
        if len(posicoes) != len(tempos):
            print(f"  AVISO: {caso['nome']} - Ajustando tempos para {len(posicoes)} pontos")
            tempos = np.linspace(tempos[0], tempos[-1], len(posicoes))
        
        tempos_lista.append(tempos)
        posicoes_lista.append(posicoes)
        velocidades_lista.append(velocidades)
        
        caso["num_pontos"] = len(posicoes)
        caso["dt"] = tempos[1] - tempos[0] if len(tempos) > 1 else 0
    
    print("\n=== INTERPOLAÇÃO DE TRAJETÓRIAS AVULSAS ===")
    for caso in casos_teste:
        print(f"    {caso['nome']}: x0={caso['x0']:.1f} m, v0={caso['v0']:.1f} m/s, ω={caso['omega']:.1f} rad/s")
    
    fig_completo = cria_grafico_interpolacao_completo(
        tempos_lista=tempos_lista,
        posicoes_lista=posicoes_lista,
        velocidades_lista=velocidades_lista,
        casos_info=casos_teste,
        titulo="Interpolação Avulsa: Posição e Velocidade vs Tempo"
    )
    
    fig_fases = cria_grafico_interpolacao_espaco_fases(
        posicoes_lista=posicoes_lista,
        velocidades_lista=velocidades_lista,
        casos_info=casos_teste,
        titulo="Interpolação Avulsa: Espaço de Fases"
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
    A frequência angular é constante.
    
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
    grafico_interpolacao_pontual_temporal = f"{output_dir}/interpolacao_pontual_v_x_vs_t.html"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    intervals = parameters.get('intervals', {})
    omega_fixo = intervals.get('omega', 5.0)
    seed = parameters.get('seed', 42)
    
    # fixa a semente para reprodutibilidade
    np.random.seed(seed)
    
    print("\n=== INTERPOLAÇÃO PONTUAL ===")
    print("\n  A interpolação é feita variando o tempo para uma mesma trajetória")
    
    # ============================================
    # GERAÇÃO DE DADOS ALEATÓRIOS
    # ============================================
    
    # limites dos intervalos dos parâmetros
    x0_min = intervals.get('x0_min', -0.5)
    x0_max = intervals.get('x0_max', 0.5)
    v0_min = intervals.get('v0_min', -1.0)
    v0_max = intervals.get('v0_max', 1.0)
    
    # número de trajetórias a serem geradas
    num_trajetorias = 2
    
    print(f"\n  Gerando {num_trajetorias} trajetórias aleatórias:")
    print(f"    x0 no intervalo [{x0_min:.3f}, {x0_max:.3f}]")
    print(f"    v0 no intervalo [{v0_min:.3f}, {v0_max:.3f}]")
    
    # gera condições iniciais aleatórias
    x0_values = np.random.uniform(x0_min, x0_max, num_trajetorias)
    v0_values = np.random.uniform(v0_min, v0_max, num_trajetorias)
    
    # Define o número de pontos baseado nos tempos de referência ou padrão
    if tempos_referencia is not None:
        num_pontos_por_trajetoria = len(tempos_referencia)
        tempos_interpolados = tempos_referencia
        print(f"\n  Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria} pontos")
    else:
        # calcula o período do sistema
        T = 2 * np.pi / omega_fixo
        tempo_maximo = T  # 1 período completo
        num_pontos_por_trajetoria = 1000
        dt_interpolacao = tempo_maximo / (num_pontos_por_trajetoria - 1)
        tempos_interpolados = np.linspace(0, tempo_maximo, num_pontos_por_trajetoria)
        print(f"\n  Configuração da interpolação:")
        print(f"    Frequência angular: {omega_fixo:.3f} rad/s")
        print(f"    Período: {T:.3f} s")
        print(f"    Passo temporal: {dt_interpolacao:.6f} s")
        print(f"    Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria}")
    
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
    
    for idx in range(num_trajetorias):
        x0 = x0_values[idx]
        v0 = v0_values[idx]
        
        print(f"\n  Processando trajetória {idx}: x0={x0:.3f}, v0={v0:.3f}")
        
        # solução analítica para validação
        pos_reais_interpolados = x0 * np.cos(omega_fixo * tempos_interpolados) + \
                                 (v0 / omega_fixo) * np.sin(omega_fixo * tempos_interpolados)
        vel_reais_interpolados = -x0 * omega_fixo * np.sin(omega_fixo * tempos_interpolados) + \
                                 v0 * np.cos(omega_fixo * tempos_interpolados)
        
        # Prepara entrada para o modelo: apenas [x0, v0] (sem tempo)
        # O modelo prevê a trajetória completa de uma só vez
        X_interpolado = np.array([[x0, v0]], dtype=np.float32)
        
        # Normaliza e faz previsão
        X_interpolado_scaled = scaler_X.transform(X_interpolado)
        X_tensor = torch.tensor(X_interpolado_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()
        
        # Desnormaliza a trajetória completa
        pred = scaler_y.inverse_transform(pred_scaled)
        
        # Separa posições e velocidades da trajetória completa
        # A saída está no formato: [x0, v0, x1, v1, ..., xN, vN]
        posicoes_previstas = pred[0, 0::2]  # Pega as posições (índices pares)
        velocidades_previstas = pred[0, 1::2]  # Pega as velocidades (índices ímpares)
        
        # Verifica se o número de pontos coincide
        if len(posicoes_previstas) != len(tempos_interpolados):
            print(f"    AVISO: Ajustando tempos para {len(posicoes_previstas)} pontos")
            tempos_ajustados = np.linspace(tempos_interpolados[0], tempos_interpolados[-1], len(posicoes_previstas))
        else:
            tempos_ajustados = tempos_interpolados
        
        # Armazena para métricas globais (ponto a ponto para compatibilidade)
        pred_pontos = np.column_stack([posicoes_previstas, velocidades_previstas])
        real_pontos = np.column_stack([pos_reais_interpolados, vel_reais_interpolados])
        
        # Se houver diferença no número de pontos, ajusta
        if len(posicoes_previstas) != len(pos_reais_interpolados):
            # Interpola os valores reais para o mesmo número de pontos
            from scipy.interpolate import interp1d
            interp_pos = interp1d(tempos_interpolados, pos_reais_interpolados, kind='linear', fill_value='extrapolate')
            interp_vel = interp1d(tempos_interpolados, vel_reais_interpolados, kind='linear', fill_value='extrapolate')
            pos_reais_ajustados = interp_pos(tempos_ajustados)
            vel_reais_ajustados = interp_vel(tempos_ajustados)
            real_pontos = np.column_stack([pos_reais_ajustados, vel_reais_ajustados])
            tempos_para_grafico = tempos_ajustados
        else:
            tempos_para_grafico = tempos_interpolados
        
        todas_previsoes.append(pred_pontos)
        todos_reais_interpolados.append(real_pontos)
        
        # Armazena informações para gráfico temporal
        tempos_lista.append(tempos_para_grafico)
        posicoes_previstas_lista.append(posicoes_previstas)
        velocidades_previstas_lista.append(velocidades_previstas)
        posicoes_reais_lista.append(pos_reais_interpolados[:len(tempos_para_grafico)])
        velocidades_reais_lista.append(vel_reais_interpolados[:len(tempos_para_grafico)])
        
        cor = CORES_PALETA[idx % len(CORES_PALETA)]
        
        casos_info_lista.append({
            'x0': x0,
            'v0': v0,
            'omega': omega_fixo,
            'cor': cor
        })
        
        # Registros para o DataFrame
        for k in range(len(tempos_para_grafico)):
            dados_interpolados.append({
                'id_trajetoria': f"x0_{x0:.3f}_v0_{v0:.3f}",
                'x0': x0,
                'v0': v0,
                'omega': omega_fixo,
                'tempo_interpolado': tempos_para_grafico[k],
                'posicao_analitica': real_pontos[k, 0],
                'velocidade_analitica': real_pontos[k, 1],
                'posicao_prevista_mlp': pred_pontos[k, 0],
                'velocidade_prevista_mlp': pred_pontos[k, 1],
                'erro_posicao': pred_pontos[k, 0] - real_pontos[k, 0],
                'erro_velocidade': pred_pontos[k, 1] - real_pontos[k, 1],
                'erro_abs_posicao': abs(pred_pontos[k, 0] - real_pontos[k, 0]),
                'erro_abs_velocidade': abs(pred_pontos[k, 1] - real_pontos[k, 1]),
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
        titulo="Interpolação Pontual: Solução Analítica vs MLP - Dados Gerados Aleatoriamente"
    )
    
    fig1.write_html(grafico_interpolacao_pontual)
    
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
        titulo="Interpolação Pontual: MLP vs Solução Analítica - Espaço de Fases"
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
        titulo="Interpolação Pontual: MLP vs Solução Analítica - Posição e Velocidade vs Tempo"
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
    df_interpolado.attrs['omega_fixo'] = omega_fixo
    df_interpolado.attrs['tempo_maximo'] = tempo_maximo if tempos_referencia is None else tempos_referencia[-1]
    df_interpolado.attrs['dt_interpolacao'] = dt_interpolacao if tempos_referencia is None else tempos_referencia[1] - tempos_referencia[0]
    df_interpolado.attrs['x0_min'] = x0_min
    df_interpolado.attrs['x0_max'] = x0_max
    df_interpolado.attrs['v0_min'] = v0_min
    df_interpolado.attrs['v0_max'] = v0_max
    
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
    Para cada instante de tempo, interpola entre duas trajetórias diferentes (variando x0 e v0).
    Não mistura dados de treino/validação/teste pois usa dados gerados aleatoriamente.
    Agora prevê trajetórias completas a partir das condições iniciais.
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
    print("\n  Para cada instante de tempo, interpola entre duas trajetórias diferentes")
    
    # ============================================
    # GERAÇÃO DE DADOS ALEATÓRIOS
    # ============================================
    
    # limites dos intervalos dos parâmetros
    x0_min = intervals.get('x0_min', -0.5)
    x0_max = intervals.get('x0_max', 0.5)
    v0_min = intervals.get('v0_min', -1.0)
    v0_max = intervals.get('v0_max', 1.0)
    
    # define o número de pontos baseado nos tempos de referência ou padrão
    if tempos_referencia is not None:
        num_pontos_por_trajetoria = len(tempos_referencia)
        tempos_unicos = tempos_referencia
        print(f"\n  Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria} pontos")
    else:
        # calcula o período do sistema
        T = 2 * np.pi / omega_fixo
        tempo_maximo = T  # 1 período completo
        num_pontos_por_trajetoria = 1000
        dt_interpolacao = tempo_maximo / (num_pontos_por_trajetoria - 1)
        tempos_unicos = np.linspace(0, tempo_maximo, num_pontos_por_trajetoria)
        print(f"\n  Configuração da interpolação:")
        print(f"    Frequência angular: {omega_fixo:.3f} rad/s")
        print(f"    Período: {T:.3f} s")
        print(f"    Passo temporal: {dt_interpolacao:.6f} s")
        print(f"    Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria}")
    
    # gera uma trajetória com amplitude pequena e outra com amplitude grande
    # para garantir que sejam diferentes, geramos várias e selecionamos as extremas
    x0_candidates = np.random.uniform(x0_min, x0_max, 100)
    v0_candidates = np.random.uniform(v0_min, v0_max, 100)
    
    amplitudes = np.sqrt(x0_candidates**2 + (v0_candidates / omega_fixo)**2)
    
    # trajetória de menor amplitude
    idx_pequena = np.argmin(amplitudes)
    x0_1 = x0_candidates[idx_pequena]
    v0_1 = v0_candidates[idx_pequena]
    
    # trajetória de maior amplitude
    idx_grande = np.argmax(amplitudes)
    x0_2 = x0_candidates[idx_grande]
    v0_2 = v0_candidates[idx_grande]
    
    print(f"\n  Trajetória 1: x0={x0_1:.3f} m, v0={v0_1:.3f} m/s")
    print(f"  Trajetória 2: x0={x0_2:.3f} m, v0={v0_2:.3f} m/s")
    
    # define os níveis de interpolação
    alphas = np.linspace(0, 1, 3)
    
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
        
        # entrada para o modelo: x0, v0
        X_interpolado = np.array([[x0_interp, v0_interp]], dtype=np.float32)
        
        # normaliza e faz previsão
        X_interpolado_scaled = scaler_X.transform(X_interpolado)
        X_tensor = torch.tensor(X_interpolado_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()
        
        # desnormaliza a trajetória completa
        pred = scaler_y.inverse_transform(pred_scaled)
        
        # separa posições e velocidades da trajetória completa
        # a saída está no formato: [x0, v0, x1, v1, ..., xN, vN]
        posicoes_previstas = pred[0, 0::2]  # posições (índices pares)
        velocidades_previstas = pred[0, 1::2]  # velocidades (índices ímpares)
        
        # verifica se o número de pontos coincide com os tempos
        if len(posicoes_previstas) != len(tempos_unicos):
            print(f"  AVISO: Ajustando tempos para {len(posicoes_previstas)} pontos")
            tempos_ajustados = np.linspace(tempos_unicos[0], tempos_unicos[-1], len(posicoes_previstas))
        else:
            tempos_ajustados = tempos_unicos
        
        # solução analítica para validação
        pos_analitico = x0_interp * np.cos(omega_fixo * tempos_ajustados) + \
                        (v0_interp / omega_fixo) * np.sin(omega_fixo * tempos_ajustados)
        vel_analitico = -x0_interp * omega_fixo * np.sin(omega_fixo * tempos_ajustados) + \
                        v0_interp * np.cos(omega_fixo * tempos_ajustados)
        
        pred_pontos = np.column_stack([posicoes_previstas, velocidades_previstas])
        real_pontos = np.column_stack([pos_analitico, vel_analitico])
        
        todas_previsoes.append(pred_pontos)
        todos_reais_interpolados.append(real_pontos)
        
        tempos_lista.append(tempos_ajustados)
        posicoes_previstas_lista.append(posicoes_previstas)
        velocidades_previstas_lista.append(velocidades_previstas)
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
        
        for k in range(len(tempos_ajustados)):
            dados_interpolados.append({
                'alpha_interpolacao': alpha,
                'x0_original_1': x0_1,
                'v0_original_1': v0_1,
                'x0_original_2': x0_2,
                'v0_original_2': v0_2,
                'x0_interpolado': x0_interp,
                'v0_interpolado': v0_interp,
                'omega': omega_fixo,
                'tempo': tempos_ajustados[k],
                'posicao_analitica': pos_analitico[k],
                'velocidade_analitica': vel_analitico[k],
                'posicao_prevista_mlp': pred_pontos[k, 0],
                'velocidade_prevista_mlp': pred_pontos[k, 1],
                'erro_posicao': pred_pontos[k, 0] - pos_analitico[k],
                'erro_velocidade': pred_pontos[k, 1] - vel_analitico[k],
                'erro_abs_posicao': abs(pred_pontos[k, 0] - pos_analitico[k]),
                'erro_abs_velocidade': abs(pred_pontos[k, 1] - vel_analitico[k]),
                'erro_rel_posicao_pct': (abs(pred_pontos[k, 0] - pos_analitico[k]) / (abs(pos_analitico[k]) + 1e-6)) * 100,
                'erro_rel_velocidade_pct': (abs(pred_pontos[k, 1] - vel_analitico[k]) / (abs(vel_analitico[k]) + 1e-6)) * 100,
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
    df_interpolado.attrs['num_tempos'] = len(tempos_ajustados)
    df_interpolado.attrs['omega_fixo'] = omega_fixo
    df_interpolado.attrs['x0_min'] = x0_min
    df_interpolado.attrs['x0_max'] = x0_max
    df_interpolado.attrs['v0_min'] = v0_min
    df_interpolado.attrs['v0_max'] = v0_max
    
    # ========================================================================
    # GRÁFICO 4: Trajetórias Originais e Interpoladas no Espaço de Fases
    # ========================================================================
    
    # gera trajetórias originais analiticamente para o gráfico
    pos_traj1 = x0_1 * np.cos(omega_fixo * tempos_ajustados) + \
                (v0_1 / omega_fixo) * np.sin(omega_fixo * tempos_ajustados)
    vel_traj1 = -x0_1 * omega_fixo * np.sin(omega_fixo * tempos_ajustados) + \
                v0_1 * np.cos(omega_fixo * tempos_ajustados)
    
    pos_traj2 = x0_2 * np.cos(omega_fixo * tempos_ajustados) + \
                (v0_2 / omega_fixo) * np.sin(omega_fixo * tempos_ajustados)
    vel_traj2 = -x0_2 * omega_fixo * np.sin(omega_fixo * tempos_ajustados) + \
                v0_2 * np.cos(omega_fixo * tempos_ajustados)
    
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
        trajetoria1_pos=pos_traj1,
        trajetoria1_vel=vel_traj1,
        trajetoria2_pos=pos_traj2,
        trajetoria2_vel=vel_traj2,
        interpolacoes_lista=interpolacoes_para_grafico,
        casos_info=casos_info_grafico,
        titulo="Interpolação entre Trajetórias no Espaço de Fases"
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
    A partir de uma trajetória escolhida aleatoriamente, gera novas condições iniciais variando x0 e v0.
    Não mistura dados de treino/validação/teste pois usa dados gerados aleatoriamente.
    Agora prevê trajetórias completas a partir das condições iniciais.
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
    omega_fixo = intervals.get('omega', 5.0)
    
    x0_min = intervals.get('x0_min', -0.5)
    x0_max = intervals.get('x0_max', 0.5)
    v0_min = intervals.get('v0_min', -1.0)
    v0_max = intervals.get('v0_max', 1.0)
    
    seed = parameters.get('seed', 42)
    np.random.seed(seed)
    
    print("\n=== GERAÇÃO DE CONDIÇÕES INICIAIS A PARTIR DA TRAJETÓRIA BASE ===")
    print(f"  Frequência: {omega_fixo} rad/s")
    print("  Gerando novas condições iniciais variando x0 e v0 dentro dos limites de treino do modelo")
    
    # ============================================
    # GERAÇÃO DE DADOS ALEATÓRIOS
    # ============================================
    
    # define o número de pontos baseado nos tempos de referência ou padrão
    if tempos_referencia is not None:
        num_pontos_por_trajetoria = len(tempos_referencia)
        tempos_unicos = tempos_referencia
        print(f"\n  Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria} pontos")
    else:
        # calcula o período do sistema
        T = 2 * np.pi / omega_fixo
        tempo_maximo = T  # 1 período completo
        num_pontos_por_trajetoria = 1000
        dt_interpolacao = tempo_maximo / (num_pontos_por_trajetoria - 1)
        tempos_unicos = np.linspace(0, tempo_maximo, num_pontos_por_trajetoria)
        print(f"\n  Configuração da interpolação:")
        print(f"    Frequência angular: {omega_fixo:.3f} rad/s")
        print(f"    Período: {T:.3f} s")
        print(f"    Passo temporal: {dt_interpolacao:.6f} s")
        print(f"    Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria}")
    
    # gera uma trajetória base aleatória
    x0_base = np.random.uniform(x0_min, x0_max)
    v0_base = np.random.uniform(v0_min, v0_max)
    
    print(f"\n  Trajetória Base Selecionada:")
    print(f"    x0 = {x0_base:.3f} m")
    print(f"    v0 = {v0_base:.3f} m/s")
    print(f"    Período: {tempos_unicos.max():.3f} s")
    
    num_variacoes = 5
    variacoes = []
    
    # opção 1: geração aleatória dentro dos limites
    np.random.seed(seed)
    x0_variacoes = np.random.uniform(x0_min, x0_max, num_variacoes)
    v0_variacoes = np.random.uniform(v0_min, v0_max, num_variacoes)
        
    print(f"\n  Gerando {num_variacoes} novas condições iniciais:")
    for i in range(num_variacoes):
        variacoes.append({
            'x0': x0_variacoes[i],
            'v0': v0_variacoes[i],
            'amplitude': np.sqrt(x0_variacoes[i]**2 + (v0_variacoes[i] / omega_fixo)**2)
        })
        print(f"    Caso {i+1}: x0={x0_variacoes[i]:.3f} m, v0={v0_variacoes[i]:.3f} m/s")
    
    todas_previsoes = []
    todos_reais_interpolados = []
    
    tempos_lista = []
    posicoes_previstas_lista = []
    velocidades_previstas_lista = []
    posicoes_reais_lista = []
    velocidades_reais_lista = []
    casos_info_lista = []
    dados_interpolados = []
    
    # para cada nova condição inicial, faz a previsão
    for var_idx, var in enumerate(variacoes):
        x0_novo = var['x0']
        v0_novo = var['v0']
        
        # entrada para o modelo: x0, v0
        X_novo = np.array([[x0_novo, v0_novo]], dtype=np.float32)
        
        # normaliza e faz previsão
        X_novo_scaled = scaler_X.transform(X_novo)
        X_tensor = torch.tensor(X_novo_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()
        
        # desnormaliza a trajetória completa
        pred = scaler_y.inverse_transform(pred_scaled)
        
        # separa posições e velocidades da trajetória completa
        # saída: [x0, v0, x1, v1, ..., xN, vN]
        posicoes_previstas = pred[0, 0::2]  # posições (índices pares)
        velocidades_previstas = pred[0, 1::2]  # velocidades (índices ímpares)
        
        # verifica se o número de pontos coincide com os tempos
        if len(posicoes_previstas) != len(tempos_unicos):
            print(f"  AVISO: Ajustando tempos para {len(posicoes_previstas)} pontos")
            tempos_ajustados = np.linspace(tempos_unicos[0], tempos_unicos[-1], len(posicoes_previstas))
        else:
            tempos_ajustados = tempos_unicos
        
        # solução analítica
        pos_analitico = x0_novo * np.cos(omega_fixo * tempos_ajustados) + \
                        (v0_novo / omega_fixo) * np.sin(omega_fixo * tempos_ajustados)
        vel_analitico = -x0_novo * omega_fixo * np.sin(omega_fixo * tempos_ajustados) + \
                        v0_novo * np.cos(omega_fixo * tempos_ajustados)
        
        # métricas globais (ponto a ponto para compatibilidade)
        pred_pontos = np.column_stack([posicoes_previstas, velocidades_previstas])
        real_pontos = np.column_stack([pos_analitico, vel_analitico])
        
        todas_previsoes.append(pred_pontos)
        todos_reais_interpolados.append(real_pontos)
        
        tempos_lista.append(tempos_ajustados)
        posicoes_previstas_lista.append(posicoes_previstas)
        velocidades_previstas_lista.append(velocidades_previstas)
        posicoes_reais_lista.append(pos_analitico)
        velocidades_reais_lista.append(vel_analitico)
        
        cor = CORES_PALETA[var_idx % len(CORES_PALETA)]
        
        casos_info_lista.append({
            'x0': x0_novo,
            'v0': v0_novo,
            'omega': omega_fixo,
            'cor': cor,
            'variation_id': var_idx
        })
        
        for k in range(len(tempos_ajustados)):
            dados_interpolados.append({
                'variacao_id': var_idx,
                'x0': x0_novo,
                'v0': v0_novo,
                'omega': omega_fixo,
                'tempo': tempos_ajustados[k],
                'posicao_analitica': pos_analitico[k],
                'velocidade_analitica': vel_analitico[k],
                'posicao_prevista_mlp': pred_pontos[k, 0],
                'velocidade_prevista_mlp': pred_pontos[k, 1],
                'erro_posicao': pred_pontos[k, 0] - pos_analitico[k],
                'erro_velocidade': pred_pontos[k, 1] - vel_analitico[k],
                'erro_abs_posicao': abs(pred_pontos[k, 0] - pos_analitico[k]),
                'erro_abs_velocidade': abs(pred_pontos[k, 1] - vel_analitico[k]),
                'erro_rel_posicao_pct': (abs(pred_pontos[k, 0] - pos_analitico[k]) / (abs(pos_analitico[k]) + 1e-6)) * 100,
                'erro_rel_velocidade_pct': (abs(pred_pontos[k, 1] - vel_analitico[k]) / (abs(vel_analitico[k]) + 1e-6)) * 100,
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
    
    print(f"\n  Total de pontos previstos: {len(predictions_all)}")
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
        titulo="Novas Condições Iniciais: Solução Analítica vs MLP"
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
        titulo="Novas Condições Iniciais: MLP vs Solução Analítica - Espaço de Fases"
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
        titulo="Novas Condições Iniciais: MLP vs Solução Analítica - Posição e Velocidade vs Tempo"
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
    df_interpolado.attrs['num_variacoes'] = num_variacoes
    df_interpolado.attrs['num_tempos'] = len(tempos_ajustados)
    df_interpolado.attrs['omega_fixo'] = omega_fixo
    df_interpolado.attrs['x0_min'] = x0_min
    df_interpolado.attrs['x0_max'] = x0_max
    df_interpolado.attrs['v0_min'] = v0_min
    df_interpolado.attrs['v0_max'] = v0_max
    
    # ========================================================================
    # GRÁFICO 4: Trajetória Base e Novas Condições Iniciais no Espaço de Fases
    # ========================================================================
    
    # gera a trajetória base analiticamente
    pos_base = x0_base * np.cos(omega_fixo * tempos_ajustados) + \
               (v0_base / omega_fixo) * np.sin(omega_fixo * tempos_ajustados)
    vel_base = -x0_base * omega_fixo * np.sin(omega_fixo * tempos_ajustados) + \
               v0_base * np.cos(omega_fixo * tempos_ajustados)
    
    novas_trajetorias_para_grafico = []
    
    for var_idx in range(num_variacoes):
        mask_var = df_interpolado['variacao_id'] == var_idx
        dados_var = df_interpolado[mask_var].sort_values('tempo')
        
        if len(dados_var) > 0:
            x0_var = dados_var['x0'].iloc[0]
            v0_var = dados_var['v0'].iloc[0]
            
            novas_trajetorias_para_grafico.append({
                'variacao_id': var_idx,
                'posicoes': dados_var['posicao_prevista_mlp'].values,
                'velocidades': dados_var['velocidade_prevista_mlp'].values,
                'x0': x0_var,
                'v0': v0_var
            })
    
    casos_info_grafico = {
        'x0_base': x0_base,
        'v0_base': v0_base
    }
    
    fig4 = cria_grafico_interpolacao_trajetorias_espaco_fases(
        trajetoria_base_pos=pos_base,
        trajetoria_base_vel=vel_base,
        novas_trajetorias_lista=novas_trajetorias_para_grafico,
        casos_info=casos_info_grafico,
        titulo="Trajetória Base vs Novas Condições Iniciais no Espaço de Fases"
    )
    
    grafico_novas_trajetorias = f"{output_dir}/trajetoria_base_vs_novas_condicoes.html"
    fig4.write_html(grafico_novas_trajetorias)
    
    fig4.show()

    return df_interpolado