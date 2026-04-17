"""
Utilitários para o pipeline do oscilador harmônico.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score

CORES_PALETA = [
    '#FF1493', '#00FF00', '#FF4500', '#00BFFF', '#FFD700',
    '#8B00FF', '#FF6347', '#00FA9A', '#DC143C', '#1E90FF',
    '#FF8C00', '#32CD32', '#FF00FF', '#00CED1', '#FF69B4',
    '#7FFF00', '#8A2BE2', '#00FF7F', '#FF2400', '#0000CD',
]


def formatar_numero_pt_br(numero):
    """Formata número no padrão brasileiro com 3 casas decimais."""
    try:
        valor = float(numero)
        return f"{valor:.3f}".replace('.', ',')
    except (ValueError, TypeError):
        return str(numero)


def cria_grafico_3d(solucao, sistemas_descricao):
    """
    Cria visualização 3D com todas as trajetórias de todos os sistemas no espaço de fases
    """
    fig = go.Figure()
    
    n_sistemas = solucao['n_sistemas']
    n_condicoes = solucao['n_condicoes']
        
    for i_sistema in range(n_sistemas):
        cor = CORES_PALETA[i_sistema % len(CORES_PALETA)]
        freq = solucao['frequencias_lineares'][i_sistema]
        omega = solucao['frequencias_angulares'][i_sistema]
        
        for i_cond in range(n_condicoes):
            x0, v0 = solucao['condicoes_iniciais'][i_cond]
            amplitude = solucao['amplitudes'][i_cond, i_sistema]
            energia_total = solucao['energia_mecanica'][-1, i_cond, i_sistema]
            
            rotulo = (f"<b>{sistemas_descricao[i_sistema]}</b><br>" +
                     f"f = {freq:.3f} Hz<br>" +
                     f"T = {1.0/freq:.3f} s<br>" +
                     f"x₀ = {x0:.3f} m, v₀ = {v0:.3f} m/s<br>" +
                     f"A = {amplitude:.3f} m<br>" +
                     f"E = {energia_total:.3f} J/kg")
            
            if i_cond == 0:
                nome = f"Sistema {i_sistema}: ω={omega:.3f} rad/s"
                show_legend = True
            else:
                nome = f"Traj_S{i_sistema}_C{i_cond}"
                show_legend = False
            
            fig.add_trace(go.Scatter3d(
                x=solucao['posicao'][:, i_cond, i_sistema],
                y=solucao['velocidade'][:, i_cond, i_sistema],
                z=solucao['tempo'],
                mode='lines',
                line=dict(color=cor, width=1.5),
                name=nome,
                legendgroup=f'sistema_{i_sistema}',
                showlegend=show_legend,
                opacity=0.9,
                hovertemplate=rotulo + '<br>Posição: %{x:.3f} m<br>Velocidade: %{y:.3f} m/s<br>Tempo: %{z:.3f} s<extra></extra>'
            ))
    
    fig.update_layout(
        title=(
            "Espaço de Fases 3D<br>" +
            f"<sup>{n_sistemas} sistemas | "
            f"{n_condicoes} condições iniciais por sistema | Total de {n_sistemas * n_condicoes} trajetórias</sup>"
        ),
        scene=dict(
            xaxis_title="Posição (m)",
            yaxis_title="Velocidade (m/s)",
            zaxis_title="Tempo (s)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1)
            )
        ),
        width=1200,
        height=900,
        legend=dict(
            title="Sistemas",
            yanchor="top",
            y=0.80,
            xanchor="left",
            x=0.80,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Black",
            borderwidth=1,
            font=dict(size=12)
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Arial"
        )
    )
    
    return fig


def cria_grafico_2d(solucao, sistemas_descricao):
    """
    Cria um único gráfico 2D com todas as trajetórias de todos os sistemas no espaço de fases.
    Os pontos das condições iniciais são destacados em preto.
    """
    fig = go.Figure()
    
    n_sistemas = solucao['n_sistemas']
    n_condicoes = solucao['n_condicoes']
    
    for i_sistema in range(n_sistemas):
        cor = CORES_PALETA[i_sistema % len(CORES_PALETA)]
        freq = solucao['frequencias_lineares'][i_sistema]
        omega = solucao['frequencias_angulares'][i_sistema]
        x0_list = []
        v0_list = []
        
        for i_cond in range(n_condicoes):
            x0, v0 = solucao['condicoes_iniciais'][i_cond]
            amplitude = solucao['amplitudes'][i_cond, i_sistema]
            x0_list.append(x0)
            v0_list.append(v0)
            
            # cria nome legível para a legenda (mostra apenas o primeiro de cada sistema)
            if i_cond == 0:  # mostra apenas um item por sistema na legenda
                nome = f"Sistema {i_sistema}: ω={omega:.3f} rad/s"
                show_legend = True
            else:
                nome = f"Traj_S{i_sistema}_C{i_cond}"
                show_legend = False
            
            fig.add_trace(go.Scatter(
                x=solucao['posicao'][:, i_cond, i_sistema],
                y=solucao['velocidade'][:, i_cond, i_sistema],
                mode='lines',
                line=dict(color=cor, width=1.0),
                name=nome,
                legendgroup=f'sistema_{i_sistema}',
                showlegend=show_legend,
                opacity=0.8,
                hovertemplate=(
                    f"<b>{sistemas_descricao[i_sistema]}</b><br>" +
                    f"f = {freq:.3f} Hz<br>" +
                    f"T = {1.0/freq:.3f} s<br>" +
                    f"x₀ = {x0:.3f} m, v₀ = {v0:.3f} m/s<br>" +
                    f"A = {amplitude:.3f} m<br>" +
                    f"Posição: %{{x:.3f}} m<br>" +
                    f"Velocidade: %{{y:.3f}} m/s<br>" +
                    f"<extra></extra>"
                )
            ))
        
        fig.add_trace(go.Scatter(
            x=x0_list,
            y=v0_list,
            mode='markers',
            marker=dict(
                color='black',
                size=8,
                symbol='circle',
                line=dict(color='white', width=1)
            ),
            name=f'Condições Iniciais - Sistema {i_sistema}',
            legendgroup=f'sistema_{i_sistema}',
            showlegend=False,
            hovertemplate=(
                f"<b>Condição Inicial - {sistemas_descricao[i_sistema]}</b><br>" +
                f"x₀ = %{{x:.3f}} m<br>" +
                f"v₀ = %{{y:.3f}} m/s<br>" +
                f"<extra></extra>"
            )
        ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='black', size=8, symbol='circle', line=dict(color='white', width=1)),
        name='Condições Iniciais',
        showlegend=True
    ))
    
    fig.update_layout(
        title=(
            "Espaço de Fases 2D<br>" +
            f"<sup>{n_sistemas} sistemas | "
            f"{n_condicoes} condições iniciais por sistema | Total de {n_sistemas * n_condicoes} trajetórias</sup>"
        ),
        xaxis_title="Posição (m)",
        yaxis_title="Velocidade (m/s)",
        width=1400,
        height=1000,
        legend=dict(
            title="Sistemas",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.00,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Black",
            borderwidth=1,
            font=dict(size=12)
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Arial"
        ),
        plot_bgcolor='black',
        paper_bgcolor='black',
        xaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=2,
            title_font_color='white',
            tickfont_color='white'
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=2,
            title_font_color='white',
            tickfont_color='white'
        ),
        title_font_color='white'
    )
    
    return fig


def cria_grafico_previsoes_mlp(predictions, y_true, titulo="Previsões do Modelo MLP"):
    """
    Cria gráficos de dispersão para visualizar as previsões do modelo MLP.
    
    Args:
        predictions: array com as previsões (n_samples, 3) - [x, v, t]
        y_true: array com os valores reais (n_samples, 3) - [x, v, t]
        titulo: título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Posição', 'Velocidade', 'Tempo'),
        horizontal_spacing=0.1
    )
    
    cores = ['blue', 'green', 'orange']
    nomes = ['Posição', 'Velocidade', 'Tempo']
    unidades = ['m', 'm/s', 's']
    
    for i in range(3):
        # adiciona pontos de dispersão
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
                    f"Valor Real: %{{x:.3f}} {unidades[i]}<br>" +
                    f"Valor Previsto: %{{y:.3f}} {unidades[i]}<br>" +
                    f"<extra></extra>"
                )
            ),
            row=1, col=i+1
        )
        
        # adiciona linha y=x (referência)
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
            title_text=f'Valor Real {nomes[i]} ({unidades[i]})',
            row=1, col=i+1,
            showgrid=True,
            gridcolor='lightgray'
        )
        
        fig.update_yaxes(
            title_text=f'Valor Previsto {nomes[i]} ({unidades[i]})',
            row=1, col=i+1,
            showgrid=True,
            gridcolor='lightgray'
        )
    
    r2_pos = r2_score(y_true[:, 0], predictions[:, 0])
    r2_vel = r2_score(y_true[:, 1], predictions[:, 1])
    r2_tempo = r2_score(y_true[:, 2], predictions[:, 2])
    
    fig.update_layout(
        title=dict(
            text=f"{titulo}<br>" +
                 f"<sup>R² Posição: {r2_pos:.4f} | R² Velocidade: {r2_vel:.4f} | R² Tempo: {r2_tempo:.4f}</sup>",
            x=0.5,
            font=dict(size=16)
        ),
        width=1200,
        height=500,
        showlegend=True,
        legend=dict(
            x=-10.00,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    return fig