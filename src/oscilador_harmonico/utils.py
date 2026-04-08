"""
Utilitários para o pipeline do oscilador harmônico.
"""

import numpy as np
import plotly.graph_objects as go

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
    Cria um único gráfico 2D com todas as trajetórias de todos os sistemas no espaço de fases
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
            
            if i_cond == 0:
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