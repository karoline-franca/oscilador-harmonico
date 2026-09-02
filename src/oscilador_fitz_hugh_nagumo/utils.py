"""
Utilitários para o pipeline do oscilador de Fitz Hugh–Nagumo.
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
    para o oscilador de FitzHugh-Nagumo.
    
    Espaço de fases: (potencial, recuperação)
    Os pontos das condições iniciais são destacados em preto.
    """
    fig = go.Figure()
    
    n_sistemas = solucao['n_sistemas']
    n_condicoes = solucao['n_condicoes']
    
    for i_sistema in range(n_sistemas):
        cor = CORES_PALETA[i_sistema % len(CORES_PALETA)]
        epsilon = solucao['parametros_epsilon'][i_sistema]
        v_eq = solucao['potencial_eq'][i_sistema]
        w_eq = solucao['recuperacao_eq'][i_sistema]
        
        amp_ciclo_teorica_v = solucao.get('amplitude_ciclo_limite_teorica_v', 2.0)[i_sistema]

        v0_list = []
        w0_list = []
        
        for i_cond in range(n_condicoes):
            v0, w0 = solucao['condicoes_iniciais'][i_cond]
            amp_v = solucao['amplitude_potencial'][i_cond, i_sistema] if 'amplitude_potencial' in solucao else 0.0
            amp_w = solucao['amplitude_recuperacao'][i_cond, i_sistema] if 'amplitude_recuperacao' in solucao else 0.0
            v0_list.append(v0)
            w0_list.append(w0)
            
            if i_cond == 0:
                nome = f"Sistema {i_sistema}"
                show_legend = True
            else:
                nome = f"Traj_S{i_sistema}_C{i_cond}"
                show_legend = False
            
            fig.add_trace(go.Scatter(
                x=solucao['potencial'][:, i_cond, i_sistema],
                y=solucao['recuperacao'][:, i_cond, i_sistema],
                mode='lines',
                line=dict(color=cor, width=1.5),
                name=nome,
                legendgroup=f'sistema_{i_sistema}',
                showlegend=show_legend,
                opacity=0.8,
                hovertemplate=(
                    f"<b>{sistemas_descricao[i_sistema]}</b><br>" +
                    f"epsilon = {epsilon:.4f}<br>" +
                    f"v* = {v_eq:.3f}, w* = {w_eq:.3f}<br>" +
                    f"v₀ = {v0:.3f}, w₀ = {w0:.3f}<br>" +
                    f"A_v = {amp_v:.3f}<br>" +
                    f"A_w = {amp_w:.3f}<br>" +
                    f"Potencial: %{{x:.3f}}<br>" +
                    f"Recuperação: %{{y:.3f}}<br>" +
                    f"<extra></extra>"
                )
            ))

        # ciclo limite teórico para FitzHugh-Nagumo
        # nuliclina como referência
        v_range = np.linspace(-amp_ciclo_teorica_v, amp_ciclo_teorica_v, 100)
        
        # nuliclina de v: w = v - v^3/3 + RI
        # nuliclina de w: w = (v + a)/b
        I = solucao.get('I', 0.5)
        R = solucao.get('R', 0.1)
        a = solucao.get('a', 0.7)
        b = solucao.get('b', 0.8)
        
        # nuliclina de v para referência
        w_nuliclina_v = v_range - v_range**3/3 + R * I
        
        # nuliclina de w para referência
        w_nuliclina_w = (v_range + a) / b
        
        fig.add_trace(go.Scatter(
            x=v_range,
            y=w_nuliclina_v,
            mode='lines',
            line=dict(color='green', width=1, dash='dot'),
            name=f'Nuliclina v (w = v - v³/3 + RI)',
            legendgroup=f'nuliclina_v_{i_sistema}',
            showlegend=(i_sistema == 0),
            hovertemplate=(
                f"<b>Nuliclina de v - Sistema {i_sistema}</b><br>" +
                f"w = v - v³/3 + RI<br>" +
                f"v: %{{x:.3f}}<br>" +
                f"w: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
        
        fig.add_trace(go.Scatter(
            x=v_range,
            y=w_nuliclina_w,
            mode='lines',
            line=dict(color='red', width=1, dash='dot'),
            name=f'Nuliclina w (w = (v + a)/b)',
            legendgroup=f'nuliclina_w_{i_sistema}',
            showlegend=(i_sistema == 0),
            hovertemplate=(
                f"<b>Nuliclina de w - Sistema {i_sistema}</b><br>" +
                f"w = (v + a)/b<br>" +
                f"v: %{{x:.3f}}<br>" +
                f"w: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
        
        fig.add_trace(go.Scatter(
            x=v0_list,
            y=w0_list,
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
                f"v₀ = %{{x:.3f}}<br>" +
                f"w₀ = %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
    
    a = solucao.get('a', 0.7)
    b = solucao.get('b', 0.8)
    I = solucao.get('I', 0.5)
    R = solucao.get('R', 0.1)
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>Espaço de Fases 2D - Oscilador de FitzHugh-Nagumo</span><br><br>" +
                 f"<span style='font-size:18px; color:#555555;'>" +
                 f"Parâmetros: a={a:.2f}, b={b:.2f}, I={I:.2f}, R={R:.2f} | " +
                 f"Nro. de sistemas: {n_sistemas} | " +
                 f"Nro. de condições iniciais por sistema: {n_condicoes} | " +
                 f"Total de {n_sistemas * n_condicoes} trajetórias</span>",
            x=0.50,
            y=0.95,
            font=dict(size=20)
        ),
        xaxis_title="Potencial (v)",
        yaxis_title="Recuperação (w)",
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
    titulo: str = "Distribuição das Amplitudes das Trajetórias - Oscilador de FitzHugh-Nagumo"
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
    y_potencial_train, y_recuperacao_train,
    y_potencial_val, y_recuperacao_val,
    y_potencial_test, y_recuperacao_test,
    titulo="Distribuição dos Dados no Espaço de Fases - Oscilador de FitzHugh-Nagumo"
):
    """
    Cria gráfico 2D mostrando a distribuição dos dados de treino, validação e teste no espaço de fases.
    Para trajetórias completas, os dados são achatados para visualização pontual.
    
    Args:
        y_potencial_train: Potenciais de treino
        y_recuperacao_train: Recuperações de treino
        y_potencial_val: Potenciais de validação
        y_recuperacao_val: Recuperações de validação
        y_potencial_test: Potenciais de teste
        y_recuperacao_test: Recuperações de teste
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    # achata as trajetórias para visualização
    y_potencial_train_flat = y_potencial_train.flatten() if y_potencial_train.ndim > 1 else y_potencial_train
    y_recuperacao_train_flat = y_recuperacao_train.flatten() if y_recuperacao_train.ndim > 1 else y_recuperacao_train
    y_potencial_val_flat = y_potencial_val.flatten() if y_potencial_val.ndim > 1 else y_potencial_val
    y_recuperacao_val_flat = y_recuperacao_val.flatten() if y_recuperacao_val.ndim > 1 else y_recuperacao_val
    y_potencial_test_flat = y_potencial_test.flatten() if y_potencial_test.ndim > 1 else y_potencial_test
    y_recuperacao_test_flat = y_recuperacao_test.flatten() if y_recuperacao_test.ndim > 1 else y_recuperacao_test
    
    # treino
    fig.add_trace(go.Scatter(
        x=y_potencial_train_flat,
        y=y_recuperacao_train_flat,
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
            f"Potencial: %{{x:.3f}}<br>" +
            f"Recuperação: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))
    
    # validação
    fig.add_trace(go.Scatter(
        x=y_potencial_val_flat,
        y=y_recuperacao_val_flat,
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
            f"Potencial: %{{x:.3f}}<br>" +
            f"Recuperação: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))
    
    # teste
    fig.add_trace(go.Scatter(
        x=y_potencial_test_flat,
        y=y_recuperacao_test_flat,
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
            f"Potencial: %{{x:.3f}}<br>" +
            f"Recuperação: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))
    
    n_train = len(y_potencial_train_flat)
    n_val = len(y_potencial_val_flat)
    n_test = len(y_potencial_test_flat)
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
        xaxis_title="Potencial (v)",
        yaxis_title="Recuperação (w)",
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
    titulo: str = "Evolução da Função de Custo durante o Treinamento - Oscilador de FitzHugh-Nagumo"
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

def cria_grafico_real_previsto_mlp(predictions, y_true, titulo="Previsões do Modelo MLP - Oscilador de FitzHugh-Nagumo"):
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
        subplot_titles=('Potencial', 'Recuperação'),
        horizontal_spacing=0.15
    )
    
    # achata as trajetórias para visualização pontual
    if predictions.ndim > 1:
        # separa potencial e recuperação das trajetórias
        potencial_pred = predictions[:, 0::2].flatten()
        recuperacao_pred = predictions[:, 1::2].flatten()
        potencial_true = y_true[:, 0::2].flatten()
        recuperacao_true = y_true[:, 1::2].flatten()
    else:
        potencial_pred = predictions[:, 0]
        recuperacao_pred = predictions[:, 1]
        potencial_true = y_true[:, 0]
        recuperacao_true = y_true[:, 1]
    
    cores = ['blue', 'green']
    nomes = ['Potencial', 'Recuperação']
    
    # potencial
    fig.add_trace(
        go.Scatter(
            x=potencial_true,
            y=potencial_pred,
            mode='markers',
            name='Potencial',
            marker=dict(
                color=cores[0],
                size=3,
                opacity=0.5
            ),
            hovertemplate=(
                f"<b>Potencial</b><br>" +
                f"Valor Real: %{{x:.3f}}<br>" +
                f"Valor Previsto: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ),
        row=1, col=1
    )
    
    # recuperação
    fig.add_trace(
        go.Scatter(
            x=recuperacao_true,
            y=recuperacao_pred,
            mode='markers',
            name='Recuperação',
            marker=dict(
                color=cores[1],
                size=3,
                opacity=0.5
            ),
            hovertemplate=(
                f"<b>Recuperação</b><br>" +
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
            min_val = min(potencial_true.min(), potencial_pred.min())
            max_val = max(potencial_true.max(), potencial_pred.max())
        else:
            min_val = min(recuperacao_true.min(), recuperacao_pred.min())
            max_val = max(recuperacao_true.max(), recuperacao_pred.max())
        
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
    
    rmse_potencial = np.sqrt(mean_squared_error(potencial_true, potencial_pred))
    rmse_recuperacao = np.sqrt(mean_squared_error(recuperacao_true, recuperacao_pred))
    r2_potencial = r2_score(potencial_true, potencial_pred)
    r2_recuperacao = r2_score(recuperacao_true, recuperacao_pred)
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:20px; color:#555555;'>" +
                 f"<sup>RMSE Potencial: {rmse_potencial:.4f} | RMSE Recuperação: {rmse_recuperacao:.4f}</sup><br>" +
                 f"<sup>R² Potencial: {r2_potencial:.4f} | R² Recuperação: {r2_recuperacao:.4f}</sup>",
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
    y_pot_true, y_rec_true,
    y_pot_pred, y_rec_pred,
    titulo="Previsões do Modelo no Espaço de Fases - Oscilador de FitzHugh-Nagumo"
):
    """
    Cria gráfico 2D mostrando as previsões do modelo no espaço de fases.
    Para trajetórias completas, os dados são achatados para visualização pontual.
    
    Args:
        y_pot_true: Potenciais reais
        y_rec_true: Recuperações reais
        y_pot_pred: Potenciais previstos
        y_rec_pred: Recuperações previstas
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    # achata as trajetórias para visualização pontual
    if y_pot_true.ndim > 1:
        y_pot_true_flat = y_pot_true.flatten()
        y_rec_true_flat = y_rec_true.flatten()
        y_pot_pred_flat = y_pot_pred.flatten()
        y_rec_pred_flat = y_rec_pred.flatten()
    else:
        y_pot_true_flat = y_pot_true
        y_rec_true_flat = y_rec_true
        y_pot_pred_flat = y_pot_pred
        y_rec_pred_flat = y_rec_pred
    
    # previsões do modelo
    fig.add_trace(go.Scatter(
        x=y_pot_pred_flat,
        y=y_rec_pred_flat,
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
            f"Potencial: %{{x:.3f}}<br>" +
            f"Recuperação: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))

    # dados reais
    fig.add_trace(go.Scatter(
        x=y_pot_true_flat,
        y=y_rec_true_flat,
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
            f"Potencial: %{{x:.3f}}<br>" +
            f"Recuperação: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))

    rmse_potencial = np.sqrt(mean_squared_error(y_pot_true_flat, y_pot_pred_flat))
    rmse_recuperacao = np.sqrt(mean_squared_error(y_rec_true_flat, y_rec_pred_flat))
    r2_potencial = r2_score(y_pot_true_flat, y_pot_pred_flat)
    r2_recuperacao = r2_score(y_rec_true_flat, y_rec_pred_flat)
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:20px; color:#555555;'>" +
                 f"<sup>RMSE Potencial: {rmse_potencial:.4f} | RMSE Recuperação: {rmse_recuperacao:.4f}</sup><br>" +
                 f"<sup>R² Potencial: {r2_potencial:.4f} | R² Recuperação: {r2_recuperacao:.4f}</sup>",
            x=0.5,
            y=0.95,
            font=dict(size=16)
        ),
        xaxis_title="Potencial (v)",
        yaxis_title="Recuperação (w)",
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
    potenciais_true, recuperacoes_true,
    potenciais_pred, recuperacoes_pred,
    tempos,
    num_trajetorias=5,
    titulo="Trajetórias Completas: Real vs Previsto"
):
    """
    Cria gráfico mostrando trajetórias completas no espaço de fases.
    
    Args:
        potenciais_true: Potenciais reais (n_trajetorias, n_timesteps)
        recuperacoes_true: Recuperações reais (n_trajetorias, n_timesteps)
        potenciais_pred: Potenciais previstos (n_trajetorias, n_timesteps)
        recuperacoes_pred: Recuperações previstas (n_trajetorias, n_timesteps)
        tempos: Array com os tempos
        num_trajetorias: Número de trajetórias a serem exibidas
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Espaço de Fases (v, w)', 'Potencial e Recuperação vs Tempo'),
        horizontal_spacing=0.15
    )
    
    n_trajetorias = min(num_trajetorias, len(potenciais_true))
    indices = np.random.choice(len(potenciais_true), n_trajetorias, replace=False)
    
    for idx, i in enumerate(indices):
        cor = CORES_PALETA[idx % len(CORES_PALETA)]
        
        # Espaço de fases
        # Real
        fig.add_trace(
            go.Scatter(
                x=potenciais_true[i],
                y=recuperacoes_true[i],
                mode='lines',
                name=f'Real {i}',
                line=dict(color=cor, width=2, dash='solid'),
                legendgroup=f'traj_{i}',
                showlegend=True,
                hovertemplate=(
                    f"<b>Trajetória Real {i}</b><br>" +
                    f"Potencial: %{{x:.3f}}<br>" +
                    f"Recuperação: %{{y:.3f}}<br>" +
                    f"<extra></extra>"
                )
            ),
            row=1, col=1
        )
        
        # Previsto
        fig.add_trace(
            go.Scatter(
                x=potenciais_pred[i],
                y=recuperacoes_pred[i],
                mode='lines',
                name=f'Previsto {i}',
                line=dict(color=cor, width=2, dash='dash'),
                legendgroup=f'traj_{i}',
                showlegend=False,
                hovertemplate=(
                    f"<b>Trajetória Prevista {i}</b><br>" +
                    f"Potencial: %{{x:.3f}}<br>" +
                    f"Recuperação: %{{y:.3f}}<br>" +
                    f"<extra></extra>"
                )
            ),
            row=1, col=1
        )
        
        # Potencial vs Tempo
        fig.add_trace(
            go.Scatter(
                x=tempos,
                y=potenciais_true[i],
                mode='lines',
                name=f'Potencial Real {i}',
                line=dict(color=cor, width=2, dash='solid'),
                legendgroup=f'traj_{i}',
                showlegend=False,
                hovertemplate=(
                    f"<b>Potencial Real {i}</b><br>" +
                    f"Tempo: %{{x:.3f}} s<br>" +
                    f"Potencial: %{{y:.3f}}<br>" +
                    f"<extra></extra>"
                )
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=tempos,
                y=potenciais_pred[i],
                mode='lines',
                name=f'Potencial Previsto {i}',
                line=dict(color=cor, width=2, dash='dash'),
                legendgroup=f'traj_{i}',
                showlegend=False,
                hovertemplate=(
                    f"<b>Potencial Previsto {i}</b><br>" +
                    f"Tempo: %{{x:.3f}} s<br>" +
                    f"Potencial: %{{y:.3f}}<br>" +
                    f"<extra></extra>"
                )
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Potencial (v)", row=1, col=1)
    fig.update_yaxes(title_text="Recuperação (w)", row=1, col=1)
    fig.update_xaxes(title_text="Tempo (s)", row=1, col=2)
    fig.update_yaxes(title_text="Potencial (v)", row=1, col=2)
    
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
    potencial_lista,
    recuperacao_lista,
    casos_info,
    titulo="Interpolação do Modelo: Potencial e Recuperação vs Tempo - Oscilador de FitzHugh-Nagumo"
):
    """
    Cria gráfico 2D combinando potencial e recuperação no tempo.
    Trabalha com trajetórias completas previstas a partir das condições iniciais.
    
    Args:
        tempos_lista: Lista de arrays com os tempos para cada caso
        potencial_lista: Lista de arrays com os potenciais previstos para cada caso
        recuperacao_lista: Lista de arrays com as recuperações previstas para cada caso
        casos_info: Lista de dicionários com informações dos casos (nome, v0, w0, cor, nome_legenda)
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
    
    for i, (tempos, potencial, recuperacao) in enumerate(zip(tempos_lista, potencial_lista, recuperacao_lista)):
        caso = casos_info[i]
        
        nome_legenda = caso.get('nome_legenda', caso['nome'])
        
        cor_recuperacao = clarear_cor(caso['cor'], fator=0.6)
        
        # potencial
        fig.add_trace(go.Scatter(
            x=tempos,
            y=potencial,
            mode='lines',
            name=f"{nome_legenda} - Potencial",
            line=dict(color=caso['cor'], width=2, dash='solid'),
            legendgroup=f"potencial_{i}",
            hovertemplate=(
                f"<b>{nome_legenda} - Potencial</b><br>" +
                f"v₀ = {caso['v0']:.3f}<br>" +
                f"w₀ = {caso['w0']:.3f}<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Potencial: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
        
        # recuperação
        fig.add_trace(go.Scatter(
            x=tempos,
            y=recuperacao,
            mode='lines',
            name=f"{nome_legenda} - Recuperação",
            line=dict(color=cor_recuperacao, width=2, dash='solid'),
            legendgroup=f"recuperacao_{i}",
            hovertemplate=(
                f"<b>{nome_legenda} - Recuperação</b><br>" +
                f"v₀ = {caso['v0']:.3f}<br>" +
                f"w₀ = {caso['w0']:.3f}<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Recuperação: %{{y:.3f}}<br>" +
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
    potencial_lista,
    recuperacao_lista,
    casos_info,
    titulo="Interpolação do Modelo no Espaço de Fases - Oscilador de FitzHugh-Nagumo"
):
    """
    Cria gráfico 2D mostrando as trajetórias previstas no espaço de fases.
    Trabalha com trajetórias completas previstas a partir das condições iniciais.
    
    Args:
        potencial_lista: Lista de arrays com os potenciais previstos para cada caso
        recuperacao_lista: Lista de arrays com as recuperações previstas para cada caso
        casos_info: Lista de dicionários com informações dos casos (nome, v0, w0, cor, nome_legenda)
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    for i, (potencial, recuperacao) in enumerate(zip(potencial_lista, recuperacao_lista)):
        caso = casos_info[i]
        
        nome_legenda = caso.get('nome_legenda', caso['nome'])
        
        # trajetória completa no espaço de fases
        fig.add_trace(go.Scatter(
            x=potencial,
            y=recuperacao,
            mode='lines',
            name=nome_legenda,
            line=dict(color=caso['cor'], width=2),
            hovertemplate=(
                f"<b>{nome_legenda}</b><br>" +
                f"v₀ = {caso['v0']:.3f}<br>" +
                f"w₀ = {caso['w0']:.3f}<br>" +
                f"Potencial: %{{x:.3f}}<br>" +
                f"Recuperação: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
        
        # ponto inicial (condição inicial)
        fig.add_trace(go.Scatter(
            x=[potencial[0]],
            y=[recuperacao[0]],
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
                f"v₀ = {caso['v0']:.3f}<br>" +
                f"w₀ = {caso['w0']:.3f}<br>" +
                f"<extra></extra>"
            )
        ))
        
        # ponto final (fim da trajetória)
        fig.add_trace(go.Scatter(
            x=[potencial[-1]],
            y=[recuperacao[-1]],
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
                f"Potencial: %{{x:.3f}}<br>" +
                f"Recuperação: %{{y:.3f}}<br>" +
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
        xaxis_title="Potencial (v)",
        yaxis_title="Recuperação (w)",
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
    titulo="Interpolação Pontual de Potencial e Recuperação - Oscilador de FitzHugh-Nagumo"
):
    """
    Cria gráficos de dispersão para visualizar a interpolação pontual do modelo MLP.
    Agora trabalha com pontos achatados de trajetórias completas.
    
    Args:
        predictions: array com as previsões (n_samples, 2) - [v, w] (pontos achatados)
        y_true: array com os valores reais (n_samples, 2) - [v, w] (pontos achatados)
        titulo: título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Potencial', 'Recuperação'),
        horizontal_spacing=0.15
    )
    
    cores = ['blue', 'green']
    nomes = ['Potencial', 'Recuperação']
    
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
    
    rmse_potencial = np.sqrt(mean_squared_error(y_true[:, 0], predictions[:, 0]))
    rmse_recuperacao = np.sqrt(mean_squared_error(y_true[:, 1], predictions[:, 1]))
    r2_potencial = r2_score(y_true[:, 0], predictions[:, 0])
    r2_recuperacao = r2_score(y_true[:, 1], predictions[:, 1])
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:20px; color:#555555;'>" +
                 f"<sup>RMSE Potencial: {rmse_potencial:.4f} | RMSE Recuperação: {rmse_recuperacao:.4f}</sup><br>" +
                 f"<sup>R² Potencial: {r2_potencial:.4f} | R² Recuperação: {r2_recuperacao:.4f}</sup><br>",
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
    y_pot_true, y_rec_true,
    y_pot_pred, y_rec_pred,
    titulo="Interpolação Pontual no Espaço de Fases - Oscilador de FitzHugh-Nagumo"
):
    """
    Cria gráfico 2D mostrando a interpolação pontual do modelo no espaço de fases.
    Agora mostra trajetórias completas previstas a partir das condições iniciais.
    
    Args:
        y_pot_true: Potenciais reais (pontos achatados)
        y_rec_true: Recuperações reais (pontos achatados)
        y_pot_pred: Potenciais previstos (pontos achatados)
        y_rec_pred: Recuperações previstas (pontos achatados)
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    # previsões do modelo
    fig.add_trace(go.Scatter(
        x=y_pot_pred.flatten(),
        y=y_rec_pred.flatten(),
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
            f"Potencial: %{{x:.3f}}<br>" +
            f"Recuperação: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))

    # dados reais (solução RK4)
    fig.add_trace(go.Scatter(
        x=y_pot_true.flatten(),
        y=y_rec_true.flatten(),
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
            f"Potencial: %{{x:.3f}}<br>" +
            f"Recuperação: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))

    rmse_potencial = np.sqrt(mean_squared_error(y_pot_true.flatten(), y_pot_pred.flatten()))
    rmse_recuperacao = np.sqrt(mean_squared_error(y_rec_true.flatten(), y_rec_pred.flatten()))
    r2_potencial = r2_score(y_pot_true.flatten(), y_pot_pred.flatten())
    r2_recuperacao = r2_score(y_rec_true.flatten(), y_rec_pred.flatten())
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:20px; color:#555555;'>" +
                 f"<sup>RMSE Potencial: {rmse_potencial:.4f} | RMSE Recuperação: {rmse_recuperacao:.4f}</sup><br>" +
                 f"<sup>R² Potencial: {r2_potencial:.4f} | R² Recuperação: {r2_recuperacao:.4f}</sup><br>",
            x=0.5,
            y=0.92,
            font=dict(size=16)
        ),
        xaxis_title="Potencial (v)",
        yaxis_title="Recuperação (w)",
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
    potenciais_previstos_lista,
    recuperacoes_previstos_lista,
    potenciais_reais_lista,
    recuperacoes_reais_lista,
    casos_info,
    titulo="Interpolação Pontual: Potencial e Recuperação vs Tempo - Oscilador de FitzHugh-Nagumo"
):
    """
    Cria gráfico 2D combinando potencial e recuperação no tempo para interpolação pontual.
    Mostra trajetórias completas previstas a partir das condições iniciais.
    
    Args:
        tempos_lista: Lista de arrays com os tempos para cada caso
        potenciais_previstos_lista: Lista de arrays com os potenciais previstos pelo MLP
        recuperacoes_previstos_lista: Lista de arrays com as recuperações previstas pelo MLP
        potenciais_reais_lista: Lista de arrays com os potenciais reais (solução RK4)
        recuperacoes_reais_lista: Lista de arrays com as recuperações reais (solução RK4)
        casos_info: Lista de dicionários com informações dos casos (v0, w0, cor)
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
    
    for i, (tempos, pot_prev, rec_prev, pot_real, rec_real) in enumerate(zip(
        tempos_lista, potenciais_previstos_lista, recuperacoes_previstos_lista,
        potenciais_reais_lista, recuperacoes_reais_lista
    )):
        caso = casos_info[i]
        
        nome_sistema = f"v₀={caso['v0']:.2f}, w₀={caso['w0']:.2f}"
        
        cor_recuperacao = clarear_cor(caso['cor'], fator=0.6)
        
        # potencial previsto pelo MLP
        fig.add_trace(go.Scatter(
            x=tempos,
            y=pot_prev,
            mode='lines',
            name=f"{nome_sistema} - Potencial (MLP)",
            line=dict(color=caso['cor'], width=2, dash='solid'),
            legendgroup=f"potencial_mlp_{i}",
            hovertemplate=(
                f"<b>Potencial (MLP)</b><br>" +
                f"v₀ = {caso['v0']:.3f}<br>" +
                f"w₀ = {caso['w0']:.3f}<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Potencial Previsto: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
        
        # potencial real (solução RK4)
        fig.add_trace(go.Scatter(
            x=tempos,
            y=pot_real,
            mode='lines',
            name=f"{nome_sistema} - Potencial (Real)",
            line=dict(color=caso['cor'], width=1.5, dash='dot'),
            legendgroup=f"potencial_real_{i}",
            hovertemplate=(
                f"<b>Potencial (Real)</b><br>" +
                f"v₀ = {caso['v0']:.3f}<br>" +
                f"w₀ = {caso['w0']:.3f}<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Potencial Real: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
        
        # recuperação prevista pelo MLP
        fig.add_trace(go.Scatter(
            x=tempos,
            y=rec_prev,
            mode='lines',
            name=f"{nome_sistema} - Recuperação (MLP)",
            line=dict(color=cor_recuperacao, width=2, dash='solid'),
            legendgroup=f"recuperacao_mlp_{i}",
            hovertemplate=(
                f"<b>Recuperação (MLP)</b><br>" +
                f"v₀ = {caso['v0']:.3f}<br>" +
                f"w₀ = {caso['w0']:.3f}<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Recuperação Prevista: %{{y:.3f}}<br>" +
                f"<extra></extra>"
            )
        ))
        
        # recuperação real (solução RK4)
        fig.add_trace(go.Scatter(
            x=tempos,
            y=rec_real,
            mode='lines',
            name=f"{nome_sistema} - Recuperação (Real)",
            line=dict(color=cor_recuperacao, width=1.5, dash='dot'),
            legendgroup=f"recuperacao_real_{i}",
            hovertemplate=(
                f"<b>Recuperação (Real)</b><br>" +
                f"v₀ = {caso['v0']:.3f}<br>" +
                f"w₀ = {caso['w0']:.3f}<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Recuperação Real: %{{y:.3f}}<br>" +
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
    trajetoria1_pot,
    trajetoria1_rec,
    trajetoria2_pot,
    trajetoria2_rec,
    interpolacoes_lista,
    casos_info,
    titulo="Interpolação entre Trajetórias no Espaço de Fases - Oscilador de FitzHugh-Nagumo",
    cores_paleta=CORES_PALETA
):
    """
    Cria gráfico 2D mostrando as duas trajetórias originais e as trajetórias interpoladas no espaço de fases.
    Trabalha com trajetórias completas previstas a partir das condições iniciais.
    Mostra os pontos inicial e final de cada trajetória.
    
    Args:
        trajetoria1_pot: Array com os potenciais da primeira trajetória
        trajetoria1_rec: Array com as recuperações da primeira trajetória
        trajetoria2_pot: Array com os potenciais da segunda trajetória
        trajetoria2_rec: Array com as recuperações da segunda trajetória
        interpolacoes_lista: Lista de dicionários contendo alpha, potenciais, recuperacoes, v0_interp, w0_interp
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
        x=trajetoria1_pot,
        y=trajetoria1_rec,
        mode='lines',
        name=f"Trajetória 1: v₀={caso_info.get('v0_1', 0):.3f}, w₀={caso_info.get('w0_1', 0):.3f}",
        line=dict(color='blue', width=3, dash='dash'),
        hovertemplate=(
            f"<b>Trajetória 1</b><br>" +
            f"v₀ = {caso_info.get('v0_1', 0):.3f}<br>" +
            f"w₀ = {caso_info.get('w0_1', 0):.3f}<br>" +
            f"Potencial: %{{x:.3f}}<br>" +
            f"Recuperação: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))
    
    # trajetória 2 (alpha = 1)
    fig.add_trace(go.Scatter(
        x=trajetoria2_pot,
        y=trajetoria2_rec,
        mode='lines',
        name=f"Trajetória 2: v₀={caso_info.get('v0_2', 0):.3f}, w₀={caso_info.get('w0_2', 0):.3f}",
        line=dict(color='red', width=3, dash='dash'),
        hovertemplate=(
            f"<b>Trajetória 2</b><br>" +
            f"v₀ = {caso_info.get('v0_2', 0):.3f}<br>" +
            f"w₀ = {caso_info.get('w0_2', 0):.3f}<br>" +
            f"Potencial: %{{x:.3f}}<br>" +
            f"Recuperação: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))
    
    # trajetórias interpoladas (0 < alpha < 1) - trajetórias completas
    n_interpolacoes = len(interpolacoes_lista)
    if n_interpolacoes > 0:
        indices_cores = np.linspace(0, len(cores_paleta) - 1, n_interpolacoes, dtype=int)
        
        for i, interpolacao in enumerate(interpolacoes_lista):
            alpha = interpolacao['alpha']
            potenciais = interpolacao['potenciais']
            recuperacoes = interpolacao['recuperacoes']
            v0_interp = interpolacao['v0_interp']
            w0_interp = interpolacao['w0_interp']
            
            cor = cores_paleta[indices_cores[i] % len(cores_paleta)]
            
            # linha da trajetória interpolada
            fig.add_trace(go.Scatter(
                x=potenciais,
                y=recuperacoes,
                mode='lines',
                name=f"Trajetória Interpolada: v₀={v0_interp:.3f}, w₀={w0_interp:.3f}",
                line=dict(color=cor, width=3, dash='solid'),
                opacity=0.7,
                hovertemplate=(
                    f"<b>Trajetória Interpolada</b><br>" +
                    f"v₀_interp = {v0_interp:.3f}<br>" +
                    f"w₀_interp = {w0_interp:.3f}<br>" +
                    f"Potencial: %{{x:.3f}}<br>" +
                    f"Recuperação: %{{y:.3f}}<br>" +
                    f"<extra></extra>"
                )
            ))
            
            # ponto inicial da trajetória interpolada
            fig.add_trace(go.Scatter(
                x=[potenciais[0]],
                y=[recuperacoes[0]],
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
                    f"v₀ = {v0_interp:.3f}<br>" +
                    f"w₀ = {w0_interp:.3f}<br>" +
                    f"<extra></extra>"
                )
            ))
            
            # ponto final da trajetória interpolada
            fig.add_trace(go.Scatter(
                x=[potenciais[-1]],
                y=[recuperacoes[-1]],
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
                    f"v_final = {potenciais[-1]:.3f}<br>" +
                    f"w_final = {recuperacoes[-1]:.3f}<br>" +
                    f"<extra></extra>"
                )
            ))
    
    # ponto inicial da trajetória 1
    fig.add_trace(go.Scatter(
        x=[trajetoria1_pot[0]],
        y=[trajetoria1_rec[0]],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='circle', line=dict(color='white', width=2)),
        name="Início Trajetória 1",
        showlegend=False,
        hovertemplate=f"<b>Início Trajetória 1</b><br>v₀ = {caso_info.get('v0_1', 0):.3f}<br>w₀ = {caso_info.get('w0_1', 0):.3f}<br><extra></extra>"
    ))
    
    # ponto final da trajetória 1
    fig.add_trace(go.Scatter(
        x=[trajetoria1_pot[-1]],
        y=[trajetoria1_rec[-1]],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='x', line=dict(color='white', width=2)),
        name="Fim Trajetória 1",
        showlegend=False,
        hovertemplate=f"<b>Fim Trajetória 1</b><br>v_final = {trajetoria1_pot[-1]:.3f}<br>w_final = {trajetoria1_rec[-1]:.3f}<br><extra></extra>"
    ))
    
    # ponto inicial da trajetória 2
    fig.add_trace(go.Scatter(
        x=[trajetoria2_pot[0]],
        y=[trajetoria2_rec[0]],
        mode='markers',
        marker=dict(color='red', size=12, symbol='circle', line=dict(color='white', width=2)),
        name="Início Trajetória 2",
        showlegend=False,
        hovertemplate=f"<b>Início Trajetória 2</b><br>v₀ = {caso_info.get('v0_2', 0):.3f}<br>w₀ = {caso_info.get('w0_2', 0):.3f}<br><extra></extra>"
    ))
    
    # ponto final da trajetória 2
    fig.add_trace(go.Scatter(
        x=[trajetoria2_pot[-1]],
        y=[trajetoria2_rec[-1]],
        mode='markers',
        marker=dict(color='red', size=12, symbol='x', line=dict(color='white', width=2)),
        name="Fim Trajetória 2",
        showlegend=False,
        hovertemplate=f"<b>Fim Trajetória 2</b><br>v_final = {trajetoria2_pot[-1]:.3f}<br>w_final = {trajetoria2_rec[-1]:.3f}<br><extra></extra>"
    ))
     
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:16px; color:#555555;'>",
            x=0.45,
            y=0.92,
            font=dict(size=16)
        ),
        xaxis_title="Potencial (v)",
        yaxis_title="Recuperação (w)",
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
    trajetoria_base_pot,
    trajetoria_base_rec,
    novas_trajetorias_lista,
    casos_info,
    titulo="Trajetória Base vs Novas Condições Iniciais no Espaço de Fases - Oscilador de FitzHugh-Nagumo",
    cores_paleta=CORES_PALETA
):
    """
    Cria gráfico 2D mostrando a trajetória base e as novas trajetórias geradas a partir de diferentes condições iniciais no espaço de fases.
    Trabalha com trajetórias completas previstas a partir das condições iniciais.
    Mostra os pontos inicial e final de cada trajetória.
    
    Args:
        trajetoria_base_pot: Array com os potenciais da trajetória base
        trajetoria_base_rec: Array com as recuperações da trajetória base
        novas_trajetorias_lista: Lista de dicionários contendo potenciais, recuperacoes, v0, w0, variacao_id
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
        x=trajetoria_base_pot,
        y=trajetoria_base_rec,
        mode='lines',
        name=f"Trajetória Base: v₀={caso_info.get('v0_base', 0):.3f}, w₀={caso_info.get('w0_base', 0):.3f}",
        line=dict(color='blue', width=3, dash='dash'),
        hovertemplate=(
            f"<b>Trajetória Base</b><br>" +
            f"v₀ = {caso_info.get('v0_base', 0):.3f}<br>" +
            f"w₀ = {caso_info.get('w0_base', 0):.3f}<br>" +
            f"Potencial: %{{x:.3f}}<br>" +
            f"Recuperação: %{{y:.3f}}<br>" +
            f"<extra></extra>"
        )
    ))
    
    # novas trajetórias - trajetórias completas
    n_novas = len(novas_trajetorias_lista)
    if n_novas > 0:
        indices_cores = np.linspace(0, len(cores_paleta) - 1, n_novas, dtype=int)
        
        for i, trajetoria in enumerate(novas_trajetorias_lista):
            potenciais = trajetoria['potenciais']
            recuperacoes = trajetoria['recuperacoes']
            v0_novo = trajetoria['v0']
            w0_novo = trajetoria['w0']
            variacao_id = trajetoria.get('variacao_id', i)
            
            cor = cores_paleta[indices_cores[i] % len(cores_paleta)]
            
            # linha da nova trajetória
            fig.add_trace(go.Scatter(
                x=potenciais,
                y=recuperacoes,
                mode='lines',
                name=f"Trajetória Interpolada: v₀={v0_novo:.3f}, w₀={w0_novo:.3f}",
                line=dict(color=cor, width=2, dash='solid'),
                opacity=0.7,
                hovertemplate=(
                    f"<b>Trajetória Interpolada</b><br>" +
                    f"v₀ = {v0_novo:.3f}<br>" +
                    f"w₀ = {w0_novo:.3f}<br>" +
                    f"Potencial: %{{x:.3f}}<br>" +
                    f"Recuperação: %{{y:.3f}}<br>" +
                    f"<extra></extra>"
                )
            ))
            
            # ponto inicial da nova trajetória
            fig.add_trace(go.Scatter(
                x=[potenciais[0]],
                y=[recuperacoes[0]],
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
                    f"v₀ = {v0_novo:.3f}<br>" +
                    f"w₀ = {w0_novo:.3f}<br>" +
                    f"<extra></extra>"
                )
            ))
            
            # ponto final da nova trajetória
            fig.add_trace(go.Scatter(
                x=[potenciais[-1]],
                y=[recuperacoes[-1]],
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
                    f"v_final = {potenciais[-1]:.3f}<br>" +
                    f"w_final = {recuperacoes[-1]:.3f}<br>" +
                    f"<extra></extra>"
                )
            ))
    
    # ponto inicial da trajetória base
    fig.add_trace(go.Scatter(
        x=[trajetoria_base_pot[0]],
        y=[trajetoria_base_rec[0]],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='circle', line=dict(color='white', width=2)),
        name="Início Trajetória Base",
        showlegend=False,
        hovertemplate=f"<b>Início Trajetória Base</b><br>v₀ = {caso_info.get('v0_base', 0):.3f}<br>w₀ = {caso_info.get('w0_base', 0):.3f}<br><extra></extra>"
    ))
    
    # ponto final da trajetória base
    fig.add_trace(go.Scatter(
        x=[trajetoria_base_pot[-1]],
        y=[trajetoria_base_rec[-1]],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='x', line=dict(color='white', width=2)),
        name="Fim Trajetória Base",
        showlegend=False,
        hovertemplate=f"<b>Fim Trajetória Base</b><br>v_final = {trajetoria_base_pot[-1]:.3f}<br>w_final = {trajetoria_base_rec[-1]:.3f}<br><extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
                 f"<span style='font-size:16px; color:#555555;'>",
            x=0.45,
            y=0.92,
            font=dict(size=16)
        ),
        xaxis_title="Potencial (v)",
        yaxis_title="Recuperação (w)",
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