"""
Utilitários para o pipeline do oscilador de Van der Pol.
"""

import numpy as np
from typing import Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score

CORES_PALETA = [
    '#C41E3A','#2E7D32','#1565C0','#E65100','#6A1B9A',
    '#00695C','#AD1457','#F57F17','#37474F','#D84315',
    '#1B5E20','#0D47A1','#4A148C','#BF360C','#1A237E',
    '#004D40','#880E4F','#4E342E','#263238','#B71C1C'
]


def formatar_numero_pt_br(numero):
    """Formata número no padrão brasileiro com 3 casas decimais."""
    try:
        valor = float(numero)
        return f"{valor:.3f}".replace('.', ',')
    except (ValueError, TypeError):
        return str(numero)


def cria_grafico_2d(solucao, sistemas_descricao):
    """
    Cria um único gráfico 2D com todas as trajetórias de todos os sistemas no espaço de fases
    para o oscilador de Van der Pol.
    
    Espaço de fases: (posição, velocidade)
    Os pontos das condições iniciais são destacados em preto.
    O ciclo limite teórico (círculo de raio 2) é mostrado como referência.
    """
    fig = go.Figure()
    
    n_sistemas = solucao['n_sistemas']
    n_condicoes = solucao['n_condicoes']
    
    for i_sistema in range(n_sistemas):
        cor = CORES_PALETA[i_sistema % len(CORES_PALETA)]
        mu = solucao['parametros_mu'][i_sistema]
        x_eq = solucao['posicao_eq'][i_sistema]
        y_eq = solucao['velocidade_eq'][i_sistema]
        
        amp_ciclo_teorica = solucao.get('amplitude_ciclo_limite_teorica', 2.0)[i_sistema]
        
        x0_list = []
        y0_list = []
        
        for i_cond in range(n_condicoes):
            x0, y0 = solucao['condicoes_iniciais'][i_cond]
            amp_x = solucao['amplitude_posicao'][i_cond, i_sistema] if 'amplitude_posicao' in solucao else 0.0
            x0_list.append(x0)
            y0_list.append(y0)
            
            if i_cond == 0:
                nome = f"Sistema {i_sistema}"
                show_legend = True
            else:
                nome = f"Traj_S{i_sistema}_C{i_cond}"
                show_legend = False
            
            fig.add_trace(go.Scatter(
                x=solucao['posicao'][:, i_cond, i_sistema],
                y=solucao['velocidade'][:, i_cond, i_sistema],
                mode='lines',
                line=dict(color=cor, width=1.5),
                name=nome,
                legendgroup=f'sistema_{i_sistema}',
                showlegend=show_legend,
                opacity=0.8,
                hovertemplate=(
                    f"<b>{sistemas_descricao[i_sistema]}</b><br>" +
                    f"mu = {mu:.3f}<br>" +
                    f"x* = {x_eq:.3f}, y* = {y_eq:.3f}<br>" +
                    f"x₀ = {x0:.3f}, y₀ = {y0:.3f}<br>" +
                    f"A_x = {amp_x:.3f}<br>" +
                    f"Posição: %{{x:.3f}}<br>" +
                    f"Velocidade: %{{y:.3f}}<br>" +
                    f"<extra></extra>"
                )
            ))
        
        fig.add_trace(go.Scatter(
            x=[x_eq],
            y=[y_eq],
            mode='markers',
            marker=dict(
                color='orange',
                size=10,
                symbol='x',
                line=dict(color='orange', width=2)
            ),
            name=f'Ponto de Equilíbrio',
            legendgroup=f'equilibrio_{i_sistema}',
            showlegend=True,
            hovertemplate=(
                f"<b>Ponto de Equilíbrio - {sistemas_descricao[i_sistema]}</b><br>" +
                f"x* = {x_eq:.3f}<br>" +
                f"y* = {y_eq:.3f}<br>" +
                f"<extra></extra>"
            )
        ))

        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=amp_ciclo_teorica * np.cos(theta),
            y=amp_ciclo_teorica * np.sin(theta),
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            name=f'Ciclo Limite',
            legendgroup=f'ciclo_limite_{i_sistema}',
            showlegend=True,
            hovertemplate=(
                f"<b>Ciclo Limite Teórico - Sistema {i_sistema}</b><br>" +
                f"Amplitude = {amp_ciclo_teorica:.3f}<br>" +
                f"<extra></extra>"
            )
        ))
        
        fig.add_trace(go.Scatter(
            x=x0_list,
            y=y0_list,
            mode='markers',
            marker=dict(
                color='blue',
                size=6,
                symbol='circle',
                line=dict(color='blue', width=1)
            ),
            name=f'Condição Inicial',
            legendgroup=f'cond_iniciais_{i_sistema}',
            showlegend=True,
            hovertemplate=(
                f"<b>Condição Inicial - {sistemas_descricao[i_sistema]}</b><br>" +
                f"x₀ = %{{x:.3f}}<br>" +
                f"y₀ = %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>Espaço de Fases 2D - Oscilador de Van der Pol</span><br><br>" +
                 f"<span style='font-size:18px; color:#555555;'>" +
                 f"Nro. de sistemas: {n_sistemas} | " +
                 f"Nro. de condições iniciais por sistema: {n_condicoes} | " +
                 f"Total de {n_sistemas * n_condicoes} trajetórias</span>",
            x=0.50,
            y=0.95,
            font=dict(size=20)
        ),
        xaxis_title="Posição (x)",
        yaxis_title="Velocidade (y)",
        width=1400,
        height=1000,
        legend=dict(
            title="Legenda",
            x=1.02,
            y=0.85,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=14),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            tracegroupgap=5
        ),
        hovermode='closest',
        plot_bgcolor='white',
        margin=dict(t=150, r=250),
        xaxis=dict(
            showgrid=False,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16),
            range=[-3.5, 3.5]
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16),
            range=[-3.5, 3.5]
        )
    )
    
    return fig


def cria_grafico_distribuicao_amplitudes(
    amplitudes: np.ndarray,
    amplitude_limite_internas: float = None,
    titulo: str = "Distribuição das Amplitudes das Trajetórias - Oscilador de Van der Pol"
) -> go.Figure:
    """
    Cria gráfico da distribuição das amplitudes das trajetórias no espaço de fases.
    
    Args:
        amplitudes: Array com as amplitudes calculadas para cada trajetória
        amplitude_limite_internas: Valor limite que separa trajetórias internas e externas (opcional)
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    
    fig = go.Figure()
    
    # histograma das amplitudes
    fig.add_trace(go.Histogram(
        x=amplitudes,
        nbinsx=30,
        name='Distribuição das Amplitudes',
        marker=dict(color='#1565C0', opacity=0.7),
        hovertemplate='Amplitude: %{x:.4f} <br>Frequência: %{y}<extra></extra>'
    ))
    
    if amplitude_limite_internas is not None:
        # linha vertical para o limite das trajetórias internas
        fig.add_vline(
            x=amplitude_limite_internas, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Limite Trajetórias Internas/Externas",
            annotation_position="top"
        )
        
        # áreas sombreadas
        fig.add_vrect(
            x0=amplitudes.min(),
            x1=amplitude_limite_internas,
            fillcolor="green",
            opacity=0.1,
            layer="below",
            line_width=0,
        )
        fig.add_vrect(
            x0=amplitude_limite_internas,
            x1=amplitudes.max(),
            fillcolor="red",
            opacity=0.1,
            layer="below",
            line_width=0,
        )
    
    amplitude_min = amplitudes.min()
    amplitude_max = amplitudes.max()
    amplitude_mediana = np.median(amplitudes)
    
    # contagem de trajetórias internas e externas
    if amplitude_limite_internas is not None:
        n_internas = np.sum(amplitudes <= amplitude_limite_internas)
        n_externas = np.sum(amplitudes > amplitude_limite_internas)
        texto_estatisticas = (
            f"<sup>Total: {len(amplitudes)} trajetórias | "
            f"Internas: {n_internas} | "
            f"Externas: {n_externas}"
        )
    else:
        texto_estatisticas = (
            f"<sup>Total: {len(amplitudes)} trajetórias | "
            f"Amplitude Mín: {amplitude_min:.4f} | "
            f"Amplitude Máx: {amplitude_max:.4f} | "
            f"Mediana: {amplitude_mediana:.4f}"
        )
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:20px; color:#555555;'>" +
                 texto_estatisticas,
            x=0.5,
            y=0.95,
            font=dict(size=16)
        ),
        xaxis_title="Amplitude",
        yaxis_title="Frequência",
        width=1200,
        height=700,
        legend=dict(
            title="Legenda",
            x=0.85,
            y=0.95,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=14)
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=150, b=80, l=80, r=80),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=14)
        )
    )
    
    return fig


def cria_grafico_distribuicao_dados(
    y_pos_train, y_vel_train,
    y_pos_val, y_vel_val,
    y_pos_test, y_vel_test,
    titulo="Distribuição dos Dados no Espaço de Fases - Oscilador de Van der Pol"
):
    """
    Cria gráfico 2D mostrando a distribuição dos dados de treino, validação e teste no espaço de fases.
    Para trajetórias completas, os dados são achatados para visualização pontual.
    
    Args:
        y_pos_train: Posições de treino
        y_vel_train: Velocidades de treino
        y_pos_val: Posições de validação
        y_vel_val: Velocidades de validação
        y_pos_test: Posições de teste
        y_vel_test: Velocidades de teste
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    # achata as trajetórias para visualização
    y_pos_train_flat = y_pos_train.flatten() if y_pos_train.ndim > 1 else y_pos_train
    y_vel_train_flat = y_vel_train.flatten() if y_vel_train.ndim > 1 else y_vel_train
    y_pos_val_flat = y_pos_val.flatten() if y_pos_val.ndim > 1 else y_pos_val
    y_vel_val_flat = y_vel_val.flatten() if y_vel_val.ndim > 1 else y_vel_val
    y_pos_test_flat = y_pos_test.flatten() if y_pos_test.ndim > 1 else y_pos_test
    y_vel_test_flat = y_vel_test.flatten() if y_vel_test.ndim > 1 else y_vel_test
    
    # treino
    fig.add_trace(go.Scatter(
        x=y_pos_train_flat,
        y=y_vel_train_flat,
        mode='markers',
        name='Dados de Treino (70%)',
        marker=dict(
            color='#2E7D32',
            size=3,
            opacity=0.5,
            symbol='circle'
        ),
        hovertemplate=(
            f"<b>Dados de Treino</b><br>" +
            f"Posição: %{{x:.3f}}<br>" +
            f"Velocidade: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))
    
    # validação
    fig.add_trace(go.Scatter(
        x=y_pos_val_flat,
        y=y_vel_val_flat,
        mode='markers',
        name='Dados de Validação (20%)',
        marker=dict(
            color='#4A148C',
            size=3,
            opacity=0.5,
            symbol='square'
        ),
        hovertemplate=(
            f"<b>Dados de Validação</b><br>" +
            f"Posição: %{{x:.3f}}<br>" +
            f"Velocidade: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))
    
    # teste
    fig.add_trace(go.Scatter(
        x=y_pos_test_flat,
        y=y_vel_test_flat,
        mode='markers',
        name='Dados de Teste (10%)',
        marker=dict(
            color='#B71C1C',
            size=3,
            opacity=0.5,
            symbol='diamond'
        ),
        hovertemplate=(
            f"<b>Dados de Teste</b><br>" +
            f"Posição: %{{x:.3f}}<br>" +
            f"Velocidade: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))
    
    n_train = len(y_pos_train_flat)
    n_val = len(y_pos_val_flat)
    n_test = len(y_pos_test_flat)
    total = n_train + n_val + n_test
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:20px; color:#555555;'>" +
                 f"<sup>Treino: {n_train} ({n_train/total*100:.1f}%) | " +
                 f"Validação: {n_val} ({n_val/total*100:.1f}%) | " +
                 f"Teste: {n_test} ({n_test/total*100:.1f}%)</sup>",
            x=0.50,
            y=0.95,
            font=dict(size=20)
        ),
        xaxis_title="Posição",
        yaxis_title="Velocidade",
        width=1400,
        height=1000,
        legend=dict(
            title="Legenda",
            x=0.85,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=16),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        hovermode='closest',
        plot_bgcolor='white',
        margin=dict(t=150),
        xaxis=dict(
            showgrid=False,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        )
    )
    
    return fig


def cria_grafico_historico_treinamento(
    history: Dict,
    titulo: str = "Evolução da Função de Custo durante o Treinamento - Oscilador de Van der Pol"
) -> go.Figure:
    """
    Cria gráfico da evolução das funções de custo de treino e validação ao longo das épocas.
    
    Args:
        history: Dicionário contendo 'train_loss' e 'val_loss'
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig = go.Figure()
    
    # função de custo de treino
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['train_loss'],
        mode='lines',
        name='Loss de Treino',
        line=dict(color='#2E7D32', width=2),
        hovertemplate=(
            f"<b>Loss de Treino</b><br>" +
            f"Época: %{{x}}<br>" +
            f"Loss: %{{y:.6f}}<br>" +
            f"<extra></extra>"
        )
    ))
    
    # loss de validação
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['val_loss'],
        mode='lines',
        name='Loss de Validação',
        line=dict(color='#B71C1C', width=2),
        hovertemplate=(
            f"<b>Loss de Validação</b><br>" +
            f"Época: %{{x}}<br>" +
            f"Loss: %{{y:.6f}}<br>" +
            f"<extra></extra>"
        )
    ))
        
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>",
            x=0.5,
            y=0.95,
            font=dict(size=16)
        ),
        xaxis_title="Época",
        yaxis_title="Função de Custo (RMSE)",
        width=1400,
        height=800,
        legend=dict(
            title="Legenda",
            x=0.85,
            y=0.95,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=14),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=150, b=80, l=80, r=80),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=14),
            type='log'
        )
    )
    
    return fig


def cria_grafico_real_previsto_mlp(predictions, y_true, titulo="Previsões do Modelo MLP - Oscilador de Van der Pol"):
    """
    Cria gráficos de dispersão para visualizar as previsões do modelo MLP.
    Para trajetórias completas, os dados são achatados para visualização pontual.
    
    Args:
        predictions: array com as previsões (n_trajetorias, 2*n_timesteps) - trajetórias completas
        y_true: array com os valores reais (n_trajetorias, 2*n_timesteps) - trajetórias completas
        titulo: título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Posição', 'Velocidade'),
        horizontal_spacing=0.15
    )
    
    # achata as trajetórias para visualização pontual
    if predictions.ndim > 1:
        # separa posição e velocidade das trajetórias
        posicao_pred = predictions[:, 0::2].flatten()
        velocidade_pred = predictions[:, 1::2].flatten()
        posicao_true = y_true[:, 0::2].flatten()
        velocidade_true = y_true[:, 1::2].flatten()
    else:
        posicao_pred = predictions[:, 0]
        velocidade_pred = predictions[:, 1]
        posicao_true = y_true[:, 0]
        velocidade_true = y_true[:, 1]
    
    cores = ['blue', 'green']
    nomes = ['Posição', 'Velocidade']
    
    # posição
    fig.add_trace(
        go.Scatter(
            x=posicao_true,
            y=posicao_pred,
            mode='markers',
            name='Posição',
            marker=dict(
                color=cores[0],
                size=3,
                opacity=0.5
            ),
            hovertemplate=(
                f"<b>Posição</b><br>" +
                f"Valor Real: %{{x:.3f}}<br>" +
                f"Valor Previsto: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ),
        row=1, col=1
    )
    
    # velocidade
    fig.add_trace(
        go.Scatter(
            x=velocidade_true,
            y=velocidade_pred,
            mode='markers',
            name='Velocidade',
            marker=dict(
                color=cores[1],
                size=3,
                opacity=0.5
            ),
            hovertemplate=(
                f"<b>Velocidade</b><br>" +
                f"Valor Real: %{{x:.3f}}<br>" +
                f"Valor Previsto: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ),
        row=1, col=2
    )
    
    # linha y=x (referência) para ambos os gráficos
    for i in range(2):
        if i == 0:
            min_val = min(posicao_true.min(), posicao_pred.min())
            max_val = max(posicao_true.max(), posicao_pred.max())
        else:
            min_val = min(velocidade_true.min(), velocidade_pred.min())
            max_val = max(velocidade_true.max(), velocidade_pred.max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Referência (y=x)',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=(i == 0),
                hovertemplate='Referência: %{x:.3f}<extra></extra>'
            ),
            row=1, col=i+1
        )
        
        fig.update_xaxes(
            title_text=f'Valor Real {nomes[i]}',
            row=1, col=i+1,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        )
        
        fig.update_yaxes(
            title_text=f'Valor Previsto {nomes[i]}',
            row=1, col=i+1,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        )
    
    rmse_posicao = np.sqrt(mean_squared_error(posicao_true, posicao_pred))
    rmse_velocidade = np.sqrt(mean_squared_error(velocidade_true, velocidade_pred))
    r2_posicao = r2_score(posicao_true, posicao_pred)
    r2_velocidade = r2_score(velocidade_true, velocidade_pred)
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:20px; color:#555555;'>" +
                 f"<sup>RMSE Posição: {rmse_posicao:.4f} | RMSE Velocidade: {rmse_velocidade:.4f}</sup><br>" +
                 f"<sup>R² Posição: {r2_posicao:.4f} | R² Velocidade: {r2_velocidade:.4f}</sup>",
            x=0.45,
            y=0.97,
            font=dict(size=16)
        ),
        width=1400,
        height=700,
        showlegend=True,
        legend=dict(
            title="Legenda",
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=16),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        hovermode='closest',
        plot_bgcolor='white',
        margin=dict(t=150)
    )
    
    return fig

def cria_grafico_previsoes_espaco_fases(
    y_pos_true, y_vel_true,
    y_pos_pred, y_vel_pred,
    titulo="Previsões do Modelo no Espaço de Fases - Oscilador de Van der Pol"
):
    """
    Cria gráfico 2D mostrando as previsões do modelo no espaço de fases.
    Para trajetórias completas, os dados são achatados para visualização pontual.
    
    Args:
        y_pos_true: Posições reais
        y_vel_true: Velocidades reais
        y_pos_pred: Posições previstas
        y_vel_pred: Velocidades previstas
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    # achata as trajetórias para visualização pontual
    if y_pos_true.ndim > 1:
        y_pos_true_flat = y_pos_true.flatten()
        y_vel_true_flat = y_vel_true.flatten()
        y_pos_pred_flat = y_pos_pred.flatten()
        y_vel_pred_flat = y_vel_pred.flatten()
    else:
        y_pos_true_flat = y_pos_true
        y_vel_true_flat = y_vel_true
        y_pos_pred_flat = y_pos_pred
        y_vel_pred_flat = y_vel_pred
    
    # previsões do modelo
    fig.add_trace(go.Scatter(
        x=y_pos_pred_flat,
        y=y_vel_pred_flat,
        mode='markers',
        name='MLP',
        marker=dict(
            color='#BF360C',
            size=3,
            opacity=0.6,
            symbol='diamond'
        ),
        hovertemplate=(
            f"<b>MLP</b><br>" +
            f"Posição: %{{x:.3f}}<br>" +
            f"Velocidade: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))

    # dados reais
    fig.add_trace(go.Scatter(
        x=y_pos_true_flat,
        y=y_vel_true_flat,
        mode='markers',
        name='Dados de Teste',
        marker=dict(
            color='#1A237E',
            size=3,
            opacity=0.6,
            symbol='circle'
        ),
        hovertemplate=(
            f"<b>Dados de Teste</b><br>" +
            f"Posição: %{{x:.3f}}<br>" +
            f"Velocidade: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))

    rmse_posicao = np.sqrt(mean_squared_error(y_pos_true_flat, y_pos_pred_flat))
    rmse_velocidade = np.sqrt(mean_squared_error(y_vel_true_flat, y_vel_pred_flat))
    r2_posicao = r2_score(y_pos_true_flat, y_pos_pred_flat)
    r2_velocidade = r2_score(y_vel_true_flat, y_vel_pred_flat)
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:20px; color:#555555;'>" +
                 f"<sup>RMSE Posição: {rmse_posicao:.4f} | RMSE Velocidade: {rmse_velocidade:.4f}</sup><br>" +
                 f"<sup>R² Posição: {r2_posicao:.4f} | R² Velocidade: {r2_velocidade:.4f}</sup>",
            x=0.5,
            y=0.95,
            font=dict(size=16)
        ),
        xaxis_title="Posição",
        yaxis_title="Velocidade",
        width=1400,
        height=1000,
        legend=dict(
            title="Legenda",
            x=0.95,
            y=0.95,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=14),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=150),
        xaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='darkgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='darkgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        )
    )
    
    return fig


def cria_grafico_trajetorias_completas(
    posicoes_true, velocidades_true,
    posicoes_pred, velocidades_pred,
    tempos,
    num_trajetorias=5,
    titulo="Trajetórias Completas: Real vs Previsto"
):
    """
    Cria gráfico mostrando trajetórias completas no espaço de fases.
    
    Args:
        posicoes_true: Posições reais (n_trajetorias, n_timesteps)
        velocidades_true: Velocidades reais (n_trajetorias, n_timesteps)
        posicoes_pred: Posições previstas (n_trajetorias, n_timesteps)
        velocidades_pred: Velocidades previstas (n_trajetorias, n_timesteps)
        tempos: Array com os tempos
        num_trajetorias: Número de trajetórias a serem exibidas
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Espaço de Fases', 'Posição e Velocidade vs Tempo'),
        horizontal_spacing=0.15
    )
    
    n_trajetorias = min(num_trajetorias, len(posicoes_true))
    indices = np.random.choice(len(posicoes_true), n_trajetorias, replace=False)
    
    for idx, i in enumerate(indices):
        cor = CORES_PALETA[idx % len(CORES_PALETA)]
        
        # Espaço de fases
        # Real
        fig.add_trace(
            go.Scatter(
                x=posicoes_true[i],
                y=velocidades_true[i],
                mode='lines',
                name=f'Real {i}',
                line=dict(color=cor, width=2, dash='solid'),
                legendgroup=f'traj_{i}',
                showlegend=True,
                hovertemplate=(
                    f"<b>Trajetória Real {i}</b><br>" +
                    f"Posição: %{{x:.3f}} m<br>" +
                    f"Velocidade: %{{y:.3f}} m/s<br>" +
                    f"<extra></extra>"
                )
            ),
            row=1, col=1
        )
        
        # Previsto
        fig.add_trace(
            go.Scatter(
                x=posicoes_pred[i],
                y=velocidades_pred[i],
                mode='lines',
                name=f'Previsto {i}',
                line=dict(color=cor, width=2, dash='dash'),
                legendgroup=f'traj_{i}',
                showlegend=False,
                hovertemplate=(
                    f"<b>Trajetória Prevista {i}</b><br>" +
                    f"Posição: %{{x:.3f}} m<br>" +
                    f"Velocidade: %{{y:.3f}} m/s<br>" +
                    f"<extra></extra>"
                )
            ),
            row=1, col=1
        )
        
        # Posição vs Tempo
        fig.add_trace(
            go.Scatter(
                x=tempos,
                y=posicoes_true[i],
                mode='lines',
                name=f'Posição Real {i}',
                line=dict(color=cor, width=2, dash='solid'),
                legendgroup=f'traj_{i}',
                showlegend=False,
                hovertemplate=(
                    f"<b>Posição Real {i}</b><br>" +
                    f"Tempo: %{{x:.3f}} s<br>" +
                    f"Posição: %{{y:.3f}} m<br>" +
                    f"<extra></extra>"
                )
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=tempos,
                y=posicoes_pred[i],
                mode='lines',
                name=f'Posição Prevista {i}',
                line=dict(color=cor, width=2, dash='dash'),
                legendgroup=f'traj_{i}',
                showlegend=False,
                hovertemplate=(
                    f"<b>Posição Prevista {i}</b><br>" +
                    f"Tempo: %{{x:.3f}} s<br>" +
                    f"Posição: %{{y:.3f}} m<br>" +
                    f"<extra></extra>"
                )
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Posição (m)", row=1, col=1)
    fig.update_yaxes(title_text="Velocidade (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Tempo (s)", row=1, col=2)
    fig.update_yaxes(title_text="Posição (m)", row=1, col=2)
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span>",
            x=0.5,
            y=0.95,
            font=dict(size=20)
        ),
        width=1600,
        height=800,
        showlegend=True,
        legend=dict(
            title="Legenda",
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=12)
        ),
        hovermode='closest',
        plot_bgcolor='white',
        margin=dict(t=100)
    )
    
    return fig


def cria_grafico_interpolacao_completo(
    tempos_lista,
    posicao_lista,
    velocidade_lista,
    casos_info,
    titulo="Interpolação do Modelo: Posição e Velocidade vs Tempo - Oscilador de Van der Pol"
):
    """
    Cria gráfico 2D combinando posição e velocidade no tempo.
    Trabalha com trajetórias completas previstas a partir das condições iniciais.
    
    Args:
        tempos_lista: Lista de arrays com os tempos para cada caso
        posicao_lista: Lista de arrays com as posições previstas para cada caso
        velocidade_lista: Lista de arrays com as velocidades previstas para cada caso
        casos_info: Lista de dicionários com informações dos casos (nome, x0, y0, cor, nome_legenda)
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    def clarear_cor(cor, fator=0.5):
        """Clareia uma cor hexadecimal."""
        cor = cor.lstrip('#')
        r, g, b = int(cor[0:2], 16), int(cor[2:4], 16), int(cor[4:6], 16)
        r = min(255, int(r + (255 - r) * fator))
        g = min(255, int(g + (255 - g) * fator))
        b = min(255, int(b + (255 - b) * fator))
        return f'#{r:02x}{g:02x}{b:02x}'
    
    for i, (tempos, posicao, velocidade) in enumerate(zip(tempos_lista, posicao_lista, velocidade_lista)):
        caso = casos_info[i]
        
        nome_legenda = caso.get('nome_legenda', caso['nome'])
        
        cor_velocidade = clarear_cor(caso['cor'], fator=0.6)
        
        # posição
        fig.add_trace(go.Scatter(
            x=tempos,
            y=posicao,
            mode='lines',
            name=f"{nome_legenda} - Posição",
            line=dict(color=caso['cor'], width=2, dash='solid'),
            legendgroup=f"posicao_{i}",
            hovertemplate=(
                f"<b>{nome_legenda} - Posição</b><br>" +
                f"x₀ = {caso['x0']:.3f}<br>" +
                f"y₀ = {caso['y0']:.3f}<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Posição: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
        
        # velocidade
        fig.add_trace(go.Scatter(
            x=tempos,
            y=velocidade,
            mode='lines',
            name=f"{nome_legenda} - Velocidade",
            line=dict(color=cor_velocidade, width=2, dash='solid'),
            legendgroup=f"velocidade_{i}",
            hovertemplate=(
                f"<b>{nome_legenda} - Velocidade</b><br>" +
                f"x₀ = {caso['x0']:.3f}<br>" +
                f"y₀ = {caso['y0']:.3f}<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Velocidade: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>",
            x=0.35,
            font=dict(size=16)
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="Estado",
        width=1400,
        height=900,
        legend=dict(
            title="Legenda",
            x=1.05,
            y=0.75,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=14),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='darkgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='darkgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        )
    )
    
    return fig


def cria_grafico_interpolacao_espaco_fases(
    posicao_lista,
    velocidade_lista,
    casos_info,
    titulo="Interpolação do Modelo no Espaço de Fases - Oscilador de Van der Pol"
):
    """
    Cria gráfico 2D mostrando as trajetórias previstas no espaço de fases.
    Trabalha com trajetórias completas previstas a partir das condições iniciais.
    
    Args:
        posicao_lista: Lista de arrays com as posições previstas para cada caso
        velocidade_lista: Lista de arrays com as velocidades previstas para cada caso
        casos_info: Lista de dicionários com informações dos casos (nome, x0, y0, cor, nome_legenda)
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    for i, (posicao, velocidade) in enumerate(zip(posicao_lista, velocidade_lista)):
        caso = casos_info[i]
        
        nome_legenda = caso.get('nome_legenda', caso['nome'])
        
        # trajetória completa no espaço de fases
        fig.add_trace(go.Scatter(
            x=posicao,
            y=velocidade,
            mode='lines',
            name=nome_legenda,
            line=dict(color=caso['cor'], width=2),
            hovertemplate=(
                f"<b>{nome_legenda}</b><br>" +
                f"x₀ = {caso['x0']:.3f}<br>" +
                f"y₀ = {caso['y0']:.3f}<br>" +
                f"Posição: %{{x:.3f}}<br>" +
                f"Velocidade: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
        
        # ponto inicial (condição inicial)
        fig.add_trace(go.Scatter(
            x=[posicao[0]],
            y=[velocidade[0]],
            mode='markers',
            marker=dict(
                color=caso['cor'],
                size=12,
                symbol='circle',
                line=dict(color='white', width=1.5)
            ),
            name=f"Início - {nome_legenda}",
            showlegend=False,
            hovertemplate=(
                f"<b>Condição Inicial - {nome_legenda}</b><br>" +
                f"x₀ = {caso['x0']:.3f}<br>" +
                f"y₀ = {caso['y0']:.3f}<br>" +
                f"<extra></extra>"
            )
        ))
        
        # ponto final (fim da trajetória)
        fig.add_trace(go.Scatter(
            x=[posicao[-1]],
            y=[velocidade[-1]],
            mode='markers',
            marker=dict(
                color=caso['cor'],
                size=10,
                symbol='x',
                line=dict(color='white', width=1)
            ),
            name=f"Fim - {nome_legenda}",
            showlegend=False,
            hovertemplate=(
                f"<b>Fim da Trajetória - {nome_legenda}</b><br>" +
                f"Posição: %{{x:.3f}}<br>" +
                f"Velocidade: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:16px; color:#555555;'>",
            x=0.35,
            font=dict(size=16)
        ),
        xaxis_title="Posição",
        yaxis_title="Velocidade",
        width=1400,
        height=1000,
        legend=dict(
            title="Legenda",
            x=1.05,
            y=0.75,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=14),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='darkgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='darkgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        )
    )
    
    return fig


def cria_grafico_interpolacao_pontual_mlp(
    predictions,
    y_true,
    titulo="Interpolação Pontual de Posição e Velocidade - Oscilador de Van der Pol"
):
    """
    Cria gráficos de dispersão para visualizar a interpolação pontual do modelo MLP.
    Agora trabalha com pontos achatados de trajetórias completas.
    
    Args:
        predictions: array com as previsões (n_samples, 2) - [x, y] (pontos achatados)
        y_true: array com os valores reais (n_samples, 2) - [x, y] (pontos achatados)
        titulo: título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Posição', 'Velocidade'),
        horizontal_spacing=0.15
    )
    
    cores = ['blue', 'green']
    nomes = ['Posição', 'Velocidade']
    
    for i in range(2):
        fig.add_trace(
            go.Scatter(
                x=y_true[:, i],
                y=predictions[:, i],
                mode='markers',
                name=f'{nomes[i]}',
                marker=dict(
                    color=cores[i],
                    size=3,
                    opacity=0.5
                ),
                hovertemplate=(
                    f"<b>{nomes[i]}</b><br>" +
                    f"Valor Real: %{{x:.3f}}<br>" +
                    f"Valor Previsto: %{{y:.3f}}<br>" +
                    f"<extra></extra>"
                )
            ),
            row=1, col=i+1
        )
        
        # linha y=x (referência)
        min_val = min(y_true[:, i].min(), predictions[:, i].min())
        max_val = max(y_true[:, i].max(), predictions[:, i].max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Referência (y=x)',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=(i == 0),  # mostra apenas na primeira coluna
                hovertemplate='Referência: %{x:.3f}<extra></extra>'
            ),
            row=1, col=i+1
        )
        
        fig.update_xaxes(
            title_text=f'Valor Real {nomes[i]}',
            row=1, col=i+1,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        )
        
        fig.update_yaxes(
            title_text=f'Valor Previsto {nomes[i]}',
            row=1, col=i+1,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        )
    
    rmse_posicao = np.sqrt(mean_squared_error(y_true[:, 0], predictions[:, 0]))
    rmse_velocidade = np.sqrt(mean_squared_error(y_true[:, 1], predictions[:, 1]))
    r2_posicao = r2_score(y_true[:, 0], predictions[:, 0])
    r2_velocidade = r2_score(y_true[:, 1], predictions[:, 1])
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:20px; color:#555555;'>" +
                 f"<sup>RMSE Posição: {rmse_posicao:.4f} | RMSE Velocidade: {rmse_velocidade:.4f}</sup><br>" +
                 f"<sup>R² Posição: {r2_posicao:.4f} | R² Velocidade: {r2_velocidade:.4f}</sup><br>",
            x=0.45,
            y=0.92,
            font=dict(size=16)
        ),
        width=1400,
        height=700,
        showlegend=True,
        legend=dict(
            title="Legenda",
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=14),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        margin=dict(t=180)
    )
    
    return fig


def cria_grafico_interpolacao_pontual_espaco_fases(
    y_pos_true, y_vel_true,
    y_pos_pred, y_vel_pred,
    titulo="Interpolação Pontual no Espaço de Fases - Oscilador de Van der Pol"
):
    """
    Cria gráfico 2D mostrando a interpolação pontual do modelo no espaço de fases.
    Agora mostra trajetórias completas previstas a partir das condições iniciais.
    
    Args:
        y_pos_true: Posições reais (pontos achatados)
        y_vel_true: Velocidades reais (pontos achatados)
        y_pos_pred: Posições previstas (pontos achatados)
        y_vel_pred: Velocidades previstas (pontos achatados)
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    # previsões do modelo
    fig.add_trace(go.Scatter(
        x=y_pos_pred.flatten(),
        y=y_vel_pred.flatten(),
        mode='markers',
        name='MLP',
        marker=dict(
            color='#BF360C',
            size=3,
            opacity=0.6,
            symbol='diamond'
        ),
        hovertemplate=(
            f"<b>MLP</b><br>" +
            f"Posição: %{{x:.3f}}<br>" +
            f"Velocidade: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))

    # dados reais (solução RK4)
    fig.add_trace(go.Scatter(
        x=y_pos_true.flatten(),
        y=y_vel_true.flatten(),
        mode='markers',
        name='Solução RK4',
        marker=dict(
            color='#1A237E',
            size=3,
            opacity=0.6,
            symbol='circle'
        ),
        hovertemplate=(
            f"<b>Solução RK4</b><br>" +
            f"Posição: %{{x:.3f}}<br>" +
            f"Velocidade: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))

    rmse_posicao = np.sqrt(mean_squared_error(y_pos_true.flatten(), y_pos_pred.flatten()))
    rmse_velocidade = np.sqrt(mean_squared_error(y_vel_true.flatten(), y_vel_pred.flatten()))
    r2_posicao = r2_score(y_pos_true.flatten(), y_pos_pred.flatten())
    r2_velocidade = r2_score(y_vel_true.flatten(), y_vel_pred.flatten())
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:20px; color:#555555;'>" +
                 f"<sup>RMSE Posição: {rmse_posicao:.4f} | RMSE Velocidade: {rmse_velocidade:.4f}</sup><br>" +
                 f"<sup>R² Posição: {r2_posicao:.4f} | R² Velocidade: {r2_velocidade:.4f}</sup><br>",
            x=0.5,
            y=0.92,
            font=dict(size=16)
        ),
        xaxis_title="Posição",
        yaxis_title="Velocidade",
        width=1400,
        height=1000,
        legend=dict(
            title="Legenda",
            x=0.95,
            y=0.85,
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=14),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=180),
        xaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='darkgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='darkgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        )
    )
    
    return fig


def cria_grafico_interpolacao_pontual_completo(
    tempos_lista,
    posicoes_previstas_lista,
    velocidades_previstas_lista,
    posicoes_reais_lista,
    velocidades_reais_lista,
    casos_info,
    titulo="Interpolação Pontual: Posição e Velocidade vs Tempo - Oscilador de Van der Pol"
):
    """
    Cria gráfico 2D combinando posição e velocidade no tempo para interpolação pontual.
    Mostra trajetórias completas previstas a partir das condições iniciais.
    
    Args:
        tempos_lista: Lista de arrays com os tempos para cada caso
        posicoes_previstas_lista: Lista de arrays com as posições previstas pelo MLP
        velocidades_previstas_lista: Lista de arrays com as velocidades previstas pelo MLP
        posicoes_reais_lista: Lista de arrays com as posições reais (solução RK4)
        velocidades_reais_lista: Lista de arrays com as velocidades reais (solução RK4)
        casos_info: Lista de dicionários com informações dos casos (x0, y0, cor)
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    def clarear_cor(cor, fator=0.5):
        """Clareia uma cor hexadecimal."""
        cor = cor.lstrip('#')
        r, g, b = int(cor[0:2], 16), int(cor[2:4], 16), int(cor[4:6], 16)
        r = min(255, int(r + (255 - r) * fator))
        g = min(255, int(g + (255 - g) * fator))
        b = min(255, int(b + (255 - b) * fator))
        return f'#{r:02x}{g:02x}{b:02x}'
    
    for i, (tempos, pos_prev, vel_prev, pos_real, vel_real) in enumerate(zip(
        tempos_lista, posicoes_previstas_lista, velocidades_previstas_lista,
        posicoes_reais_lista, velocidades_reais_lista
    )):
        caso = casos_info[i]
        
        nome_sistema = f"x₀={caso['x0']:.2f}, y₀={caso['y0']:.2f}"
        
        cor_velocidade = clarear_cor(caso['cor'], fator=0.6)
        
        # posição prevista pelo MLP
        fig.add_trace(go.Scatter(
            x=tempos,
            y=pos_prev,
            mode='lines',
            name=f"{nome_sistema} - Posição (MLP)",
            line=dict(color=caso['cor'], width=2, dash='solid'),
            legendgroup=f"posicao_mlp_{i}",
            hovertemplate=(
                f"<b>Posição (MLP)</b><br>" +
                f"x₀ = {caso['x0']:.3f}<br>" +
                f"y₀ = {caso['y0']:.3f}<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Posição Prevista: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
        
        # posição real (solução RK4)
        fig.add_trace(go.Scatter(
            x=tempos,
            y=pos_real,
            mode='lines',
            name=f"{nome_sistema} - Posição (Real)",
            line=dict(color=caso['cor'], width=1.5, dash='dot'),
            legendgroup=f"posicao_real_{i}",
            hovertemplate=(
                f"<b>Posição (Real)</b><br>" +
                f"x₀ = {caso['x0']:.3f}<br>" +
                f"y₀ = {caso['y0']:.3f}<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Posição Real: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
        
        # velocidade prevista pelo MLP
        fig.add_trace(go.Scatter(
            x=tempos,
            y=vel_prev,
            mode='lines',
            name=f"{nome_sistema} - Velocidade (MLP)",
            line=dict(color=cor_velocidade, width=2, dash='solid'),
            legendgroup=f"velocidade_mlp_{i}",
            hovertemplate=(
                f"<b>Velocidade (MLP)</b><br>" +
                f"x₀ = {caso['x0']:.3f}<br>" +
                f"y₀ = {caso['y0']:.3f}<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Velocidade Prevista: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
        
        # velocidade real (solução RK4)
        fig.add_trace(go.Scatter(
            x=tempos,
            y=vel_real,
            mode='lines',
            name=f"{nome_sistema} - Velocidade (Real)",
            line=dict(color=cor_velocidade, width=1.5, dash='dot'),
            legendgroup=f"velocidade_real_{i}",
            hovertemplate=(
                f"<b>Velocidade (Real)</b><br>" +
                f"x₀ = {caso['x0']:.3f}<br>" +
                f"y₀ = {caso['y0']:.3f}<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Velocidade Real: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:16px; color:#555555;'>",
            x=0.5,
            y=0.95,
            font=dict(size=16)
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="Estado",
        width=1400,
        height=900,
        legend=dict(
            title="Legenda",
            x=1.02,
            y=0.75,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=14),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=150),
        xaxis=dict(
            showgrid=True,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='darkgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='darkgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        )
    )
    
    return fig


def cria_grafico_interpolacao_entre_trajetorias_espaco_fases(
    trajetoria1_pos,
    trajetoria1_vel,
    trajetoria2_pos,
    trajetoria2_vel,
    interpolacoes_lista,
    casos_info,
    titulo="Interpolação entre Trajetórias no Espaço de Fases - Oscilador de Van der Pol",
    cores_paleta=CORES_PALETA
):
    """
    Cria gráfico 2D mostrando as duas trajetórias originais e as trajetórias interpoladas no espaço de fases.
    Trabalha com trajetórias completas previstas a partir das condições iniciais.
    Mostra os pontos inicial e final de cada trajetória.
    
    Args:
        trajetoria1_pos: Array com as posições da primeira trajetória
        trajetoria1_vel: Array com as velocidades da primeira trajetória
        trajetoria2_pos: Array com as posições da segunda trajetória
        trajetoria2_vel: Array com as velocidades da segunda trajetória
        interpolacoes_lista: Lista de dicionários contendo alpha, posicoes, velocidades, x0_interp, v0_interp
        casos_info: Lista de dicionários com informações dos casos interpolados
        titulo: Título do gráfico
        cores_paleta: Lista de cores para as trajetórias interpoladas
        
    Returns:
        Figura Plotly
    """
    
    fig = go.Figure()
    
    # trajetória 1 (alpha = 0)
    caso_info = casos_info[0] if casos_info else {}
    fig.add_trace(go.Scatter(
        x=trajetoria1_pos,
        y=trajetoria1_vel,
        mode='lines',
        name=f"Trajetória 1: x₀={caso_info.get('x0_1', 0):.3f}, y₀={caso_info.get('v0_1', 0):.3f}",
        line=dict(color='blue', width=3, dash='dash'),
        hovertemplate=(
            f"<b>Trajetória 1</b><br>" +
            f"x₀ = {caso_info.get('x0_1', 0):.3f}<br>" +
            f"y₀ = {caso_info.get('v0_1', 0):.3f}<br>" +
            f"Posição: %{{x:.3f}}<br>" +
            f"Velocidade: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))
    
    # trajetória 2 (alpha = 1)
    fig.add_trace(go.Scatter(
        x=trajetoria2_pos,
        y=trajetoria2_vel,
        mode='lines',
        name=f"Trajetória 2: x₀={caso_info.get('x0_2', 0):.3f}, y₀={caso_info.get('v0_2', 0):.3f}",
        line=dict(color='red', width=3, dash='dash'),
        hovertemplate=(
            f"<b>Trajetória 2</b><br>" +
            f"x₀ = {caso_info.get('x0_2', 0):.3f}<br>" +
            f"y₀ = {caso_info.get('v0_2', 0):.3f}<br>" +
            f"Posição: %{{x:.3f}}<br>" +
            f"Velocidade: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))
    
    # trajetórias interpoladas (0 < alpha < 1) - trajetórias completas
    n_interpolacoes = len(interpolacoes_lista)
    if n_interpolacoes > 0:
        indices_cores = np.linspace(0, len(cores_paleta) - 1, n_interpolacoes, dtype=int)
        
        for i, interpolacao in enumerate(interpolacoes_lista):
            alpha = interpolacao['alpha']
            posicoes = interpolacao['posicoes']
            velocidades = interpolacao['velocidades']
            x0_interp = interpolacao['x0_interp']
            v0_interp = interpolacao['v0_interp']
            
            cor = cores_paleta[indices_cores[i] % len(cores_paleta)]
            
            # linha da trajetória interpolada
            fig.add_trace(go.Scatter(
                x=posicoes,
                y=velocidades,
                mode='lines',
                name=f"Trajetória Interpolada: x₀={x0_interp:.3f}, y₀={v0_interp:.3f}",
                line=dict(color=cor, width=3, dash='solid'),
                opacity=0.7,
                hovertemplate=(
                    f"<b>Trajetória Interpolada</b><br>" +
                    f"x₀_interp = {x0_interp:.3f}<br>" +
                    f"y₀_interp = {v0_interp:.3f}<br>" +
                    f"Posição: %{{x:.3f}}<br>" +
                    f"Velocidade: %{{y:.3f}}<br>" +
                    f"<extra></extra>"
                )
            ))
            
            # ponto inicial da trajetória interpolada
            fig.add_trace(go.Scatter(
                x=[posicoes[0]],
                y=[velocidades[0]],
                mode='markers',
                marker=dict(
                    color=cor,
                    size=8,
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                name=f"Início Trajetória Interpolada",
                showlegend=False,
                hovertemplate=(
                    f"<b>Início Trajetória Interpolada</b><br>" +
                    f"x₀ = {x0_interp:.3f}<br>" +
                    f"y₀ = {v0_interp:.3f}<br>" +
                    f"<extra></extra>"
                )
            ))
            
            # ponto final da trajetória interpolada
            fig.add_trace(go.Scatter(
                x=[posicoes[-1]],
                y=[velocidades[-1]],
                mode='markers',
                marker=dict(
                    color=cor,
                    size=8,
                    symbol='x',
                    line=dict(color='white', width=1)
                ),
                name=f"Fim Trajetória Interpolada",
                showlegend=False,
                hovertemplate=(
                    f"<b>Fim Trajetória Interpolada</b><br>" +
                    f"x_final = {posicoes[-1]:.3f}<br>" +
                    f"y_final = {velocidades[-1]:.3f}<br>" +
                    f"<extra></extra>"
                )
            ))
    
    # ponto inicial da trajetória 1
    fig.add_trace(go.Scatter(
        x=[trajetoria1_pos[0]],
        y=[trajetoria1_vel[0]],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='circle', line=dict(color='white', width=2)),
        name="Início Trajetória 1",
        showlegend=False,
        hovertemplate=f"<b>Início Trajetória 1</b><br>x₀ = {caso_info.get('x0_1', 0):.3f}<br>y₀ = {caso_info.get('v0_1', 0):.3f}<br><extra></extra>"
    ))
    
    # ponto final da trajetória 1
    fig.add_trace(go.Scatter(
        x=[trajetoria1_pos[-1]],
        y=[trajetoria1_vel[-1]],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='x', line=dict(color='white', width=2)),
        name="Fim Trajetória 1",
        showlegend=False,
        hovertemplate=f"<b>Fim Trajetória 1</b><br>x_final = {trajetoria1_pos[-1]:.3f}<br>y_final = {trajetoria1_vel[-1]:.3f}<br><extra></extra>"
    ))
    
    # ponto inicial da trajetória 2
    fig.add_trace(go.Scatter(
        x=[trajetoria2_pos[0]],
        y=[trajetoria2_vel[0]],
        mode='markers',
        marker=dict(color='red', size=12, symbol='circle', line=dict(color='white', width=2)),
        name="Início Trajetória 2",
        showlegend=False,
        hovertemplate=f"<b>Início Trajetória 2</b><br>x₀ = {caso_info.get('x0_2', 0):.3f}<br>y₀ = {caso_info.get('v0_2', 0):.3f}<br><extra></extra>"
    ))
    
    # ponto final da trajetória 2
    fig.add_trace(go.Scatter(
        x=[trajetoria2_pos[-1]],
        y=[trajetoria2_vel[-1]],
        mode='markers',
        marker=dict(color='red', size=12, symbol='x', line=dict(color='white', width=2)),
        name="Fim Trajetória 2",
        showlegend=False,
        hovertemplate=f"<b>Fim Trajetória 2</b><br>x_final = {trajetoria2_pos[-1]:.3f}<br>y_final = {trajetoria2_vel[-1]:.3f}<br><extra></extra>"
    ))
     
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:16px; color:#555555;'>",
            x=0.45,
            y=0.92,
            font=dict(size=16)
        ),
        xaxis_title="Posição",
        yaxis_title="Velocidade",
        width=1400,
        height=1000,
        legend=dict(
            title="Legenda",
            x=1.00,
            y=0.75,
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=14),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=150),
        xaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='darkgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='darkgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        )
    )
    
    return fig


def cria_grafico_interpolacao_trajetorias_espaco_fases(
    trajetoria_base_pos,
    trajetoria_base_vel,
    novas_trajetorias_lista,
    casos_info,
    titulo="Trajetória Base vs Novas Condições Iniciais no Espaço de Fases - Oscilador de Van der Pol",
    cores_paleta=CORES_PALETA
):
    """
    Cria gráfico 2D mostrando a trajetória base e as novas trajetórias geradas a partir de diferentes condições iniciais no espaço de fases.
    Trabalha com trajetórias completas previstas a partir das condições iniciais.
    Mostra os pontos inicial e final de cada trajetória.
    
    Args:
        trajetoria_base_pos: Array com as posições da trajetória base
        trajetoria_base_vel: Array com as velocidades da trajetória base
        novas_trajetorias_lista: Lista de dicionários contendo posicoes, velocidades, x0, v0, variacao_id
        casos_info: Dicionário com informações da trajetória base
        titulo: Título do gráfico
        cores_paleta: Lista de cores para as novas trajetórias
        
    Returns:
        Figura Plotly
    """
    
    fig = go.Figure()
    
    # trajetória base
    caso_info = casos_info if casos_info else {}
    fig.add_trace(go.Scatter(
        x=trajetoria_base_pos,
        y=trajetoria_base_vel,
        mode='lines',
        name=f"Trajetória Base: x₀={caso_info.get('x0_base', 0):.3f}, y₀={caso_info.get('v0_base', 0):.3f}",
        line=dict(color='blue', width=3, dash='dash'),
        hovertemplate=(
            f"<b>Trajetória Base</b><br>" +
            f"x₀ = {caso_info.get('x0_base', 0):.3f}<br>" +
            f"y₀ = {caso_info.get('v0_base', 0):.3f}<br>" +
            f"Posição: %{{x:.3f}}<br>" +
            f"Velocidade: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))
    
    # novas trajetórias - trajetórias completas
    n_novas = len(novas_trajetorias_lista)
    if n_novas > 0:
        indices_cores = np.linspace(0, len(cores_paleta) - 1, n_novas, dtype=int)
        
        for i, trajetoria in enumerate(novas_trajetorias_lista):
            posicoes = trajetoria['posicoes']
            velocidades = trajetoria['velocidades']
            x0_novo = trajetoria['x0']
            v0_novo = trajetoria['v0']
            variacao_id = trajetoria.get('variacao_id', i)
            
            cor = cores_paleta[indices_cores[i] % len(cores_paleta)]
            
            # linha da nova trajetória
            fig.add_trace(go.Scatter(
                x=posicoes,
                y=velocidades,
                mode='lines',
                name=f"Trajetória Interpolada: x₀={x0_novo:.3f}, y₀={v0_novo:.3f}",
                line=dict(color=cor, width=2, dash='solid'),
                opacity=0.7,
                hovertemplate=(
                    f"<b>Trajetória Interpolada</b><br>" +
                    f"x₀ = {x0_novo:.3f}<br>" +
                    f"y₀ = {v0_novo:.3f}<br>" +
                    f"Posição: %{{x:.3f}}<br>" +
                    f"Velocidade: %{{y:.3f}}<br>" +
                    f"<extra></extra>"
                )
            ))
            
            # ponto inicial da nova trajetória
            fig.add_trace(go.Scatter(
                x=[posicoes[0]],
                y=[velocidades[0]],
                mode='markers',
                marker=dict(
                    color=cor,
                    size=8,
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                name=f"Início Trajetória Interpolada",
                showlegend=False,
                hovertemplate=(
                    f"<b>Início Trajetória Interpolada</b><br>" +
                    f"x₀ = {x0_novo:.3f}<br>" +
                    f"y₀ = {v0_novo:.3f}<br>" +
                    f"<extra></extra>"
                )
            ))
            
            # ponto final da nova trajetória
            fig.add_trace(go.Scatter(
                x=[posicoes[-1]],
                y=[velocidades[-1]],
                mode='markers',
                marker=dict(
                    color=cor,
                    size=8,
                    symbol='x',
                    line=dict(color='white', width=1)
                ),
                name=f"Fim Trajetória Interpolada",
                showlegend=False,
                hovertemplate=(
                    f"<b>Fim Trajetória Interpolada</b><br>" +
                    f"x_final = {posicoes[-1]:.3f}<br>" +
                    f"y_final = {velocidades[-1]:.3f}<br>" +
                    f"<extra></extra>"
                )
            ))
    
    # ponto inicial da trajetória base
    fig.add_trace(go.Scatter(
        x=[trajetoria_base_pos[0]],
        y=[trajetoria_base_vel[0]],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='circle', line=dict(color='white', width=2)),
        name="Início Trajetória Base",
        showlegend=False,
        hovertemplate=f"<b>Início Trajetória Base</b><br>x₀ = {caso_info.get('x0_base', 0):.3f}<br>y₀ = {caso_info.get('v0_base', 0):.3f}<br><extra></extra>"
    ))
    
    # ponto final da trajetória base
    fig.add_trace(go.Scatter(
        x=[trajetoria_base_pos[-1]],
        y=[trajetoria_base_vel[-1]],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='x', line=dict(color='white', width=2)),
        name="Fim Trajetória Base",
        showlegend=False,
        hovertemplate=f"<b>Fim Trajetória Base</b><br>x_final = {trajetoria_base_pos[-1]:.3f}<br>y_final = {trajetoria_base_vel[-1]:.3f}<br><extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:16px; color:#555555;'>",
            x=0.45,
            y=0.92,
            font=dict(size=16)
        ),
        xaxis_title="Posição",
        yaxis_title="Velocidade",
        width=1400,
        height=1000,
        legend=dict(
            title="Legenda",
            x=1.00,
            y=0.75,
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=14),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=150),
        xaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='darkgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='darkgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=16)
        )
    )
    
    return fig