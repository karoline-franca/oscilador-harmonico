"""
Utilitários para o pipeline do oscilador harmônico.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score

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
            f"<sup> Nro de sistemas: {n_sistemas} | "
            f"Nro de condições iniciais por sistema: {n_condicoes} | Total de {n_sistemas * n_condicoes} trajetórias</sup>"
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
            f"<sup>Nro de sistemas: {n_sistemas} | "
            f"Nro de condições iniciais por sistema: {n_condicoes} | Total de {n_sistemas * n_condicoes} trajetórias</sup>"
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


# gráfico saída x,v

def cria_grafico_real_previsto_mlp(predictions, y_true, titulo="Previsões do Modelo MLP"):
    """
    Cria gráficos de dispersão para visualizar as previsões do modelo MLP.
    
    Args:
        predictions: array com as previsões (n_samples, 2) - [x, v]
        y_true: array com os valores reais (n_samples, 2) - [x, v]
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
    unidades = ['m', 'm/s']
    
    for i in range(2):
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
    
    rmse_pos = np.sqrt(mean_squared_error(y_true[:, 0], predictions[:, 0]))
    rmse_vel = np.sqrt(mean_squared_error(y_true[:, 1], predictions[:, 1]))
    r2_pos = r2_score(y_true[:, 0], predictions[:, 0])
    r2_vel = r2_score(y_true[:, 1], predictions[:, 1])
    
    fig.update_layout(
        title=dict(
            text=f"{titulo}<br>" +
                 f"<sup>RMSE Posição: {rmse_pos:.4f} m | RMSE Velocidade: {rmse_vel:.4f} m/s</sup><br>" +
                 f"<sup>R² Posição: {r2_pos:.4f} | R² Velocidade: {r2_vel:.4f}</sup>",
            x=0.5,
            font=dict(size=16)
        ),
        width=1000,
        height=500,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    return fig

def cria_grafico_distribuicao_dados(
    y_pos_train, y_vel_train,
    y_pos_val, y_vel_val,
    y_pos_test, y_vel_test,
    titulo="Distribuição dos Dados no Espaço de Fases"
):
    """
    Cria gráfico 2D mostrando a distribuição dos dados de treino, validação e teste no espaço de fases.
    
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
    
    # treino
    fig.add_trace(go.Scatter(
        x=y_pos_train.flatten(),
        y=y_vel_train.flatten(),
        mode='markers',
        name='Dados de Treino (70%)',
        marker=dict(
            color='#00FF00',
            size=3,
            opacity=0.5,
            symbol='circle'
        ),
        hovertemplate=(
            f"<b>Dados de Treino</b><br>" +
            f"Posição: %{{x:.3f}} m<br>" +
            f"Velocidade: %{{y:.3f}} m/s<br>" +
            f"<extra></extra>"
        )
    ))
    
    # validação
    fig.add_trace(go.Scatter(
        x=y_pos_val.flatten(),
        y=y_vel_val.flatten(),
        mode='markers',
        name='Dados de Validação (20%)',
        marker=dict(
            color='#8A2BE2',
            size=3,
            opacity=0.5,
            symbol='square'
        ),
        hovertemplate=(
            f"<b>Dados de Validação</b><br>" +
            f"Posição: %{{x:.3f}} m<br>" +
            f"Velocidade: %{{y:.3f}} m/s<br>" +
            f"<extra></extra>"
        )
    ))
    
    # teste
    fig.add_trace(go.Scatter(
        x=y_pos_test.flatten(),
        y=y_vel_test.flatten(),
        mode='markers',
        name='Dados de Teste (10%)',
        marker=dict(
            color='#FF2400',
            size=3,
            opacity=0.5,
            symbol='diamond'
        ),
        hovertemplate=(
            f"<b>Dados de Teste</b><br>" +
            f"Posição: %{{x:.3f}} m<br>" +
            f"Velocidade: %{{y:.3f}} m/s<br>" +
            f"<extra></extra>"
        )
    ))
    
    n_train = len(y_pos_train)
    n_val = len(y_pos_val)
    n_test = len(y_pos_test)
    total = n_train + n_val + n_test
    
    fig.update_layout(
        title=dict(
            text=f"{titulo}<br>" +
                 f"<sup>Treino: {n_train} ({n_train/total*100:.1f}%) | " +
                 f"Validação: {n_val} ({n_val/total*100:.1f}%) | " +
                 f"Teste: {n_test} ({n_test/total*100:.1f}%)</sup>",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="Posição (m)",
        yaxis_title="Velocidade (m/s)",
        width=1400,
        height=1000,
        legend=dict(
            x=0.75,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest',
        plot_bgcolor='black',
        xaxis=dict(
            showgrid=False,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1
        )
    )
    
    return fig

def cria_grafico_previsoes_espaco_fases(
    y_pos_true, y_vel_true,
    y_pos_pred, y_vel_pred,
    titulo="Previsões do Modelo no Espaço de Fases"
):
    """
    Cria gráfico 2D mostrando as previsões do modelo no espaço de fases.
    
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
    
    # previsões do modelo
    fig.add_trace(go.Scatter(
        x=y_pos_pred.flatten(),
        y=y_vel_pred.flatten(),
        mode='markers',
        name='MLP',
        marker=dict(
            color='#FF4500',
            size=3,
            opacity=0.6,
            symbol='diamond'
        ),
        hovertemplate=(
            f"<b>MLP</b><br>" +
            f"Posição: %{{x:.3f}} m<br>" +
            f"Velocidade: %{{y:.3f}} m/s<br>" +
            f"<extra></extra>"
        )
    ))

    # dados reais
    fig.add_trace(go.Scatter(
        x=y_pos_true.flatten(),
        y=y_vel_true.flatten(),
        mode='markers',
        name='Dados Reais',
        marker=dict(
            color='#00BFFF',
            size=3,
            opacity=0.6,
            symbol='circle'
        ),
        hovertemplate=(
            f"<b>Dados Reais</b><br>" +
            f"Posição: %{{x:.3f}} m<br>" +
            f"Velocidade: %{{y:.3f}} m/s<br>" +
            f"<extra></extra>"
        )
    ))

    rmse_pos = np.sqrt(mean_squared_error(y_pos_true.flatten(), y_pos_pred.flatten()))
    rmse_vel = np.sqrt(mean_squared_error(y_vel_true.flatten(), y_vel_pred.flatten()))
    r2_pos = r2_score(y_pos_true.flatten(), y_pos_pred.flatten())
    r2_vel = r2_score(y_vel_true.flatten(), y_vel_pred.flatten())
    
    fig.update_layout(
        title=dict(
            text=f"{titulo}<br>" +
                 f"<sup>RMSE Posição: {rmse_pos:.4f} m | RMSE Velocidade: {rmse_vel:.4f} m/s</sup><br>" +
                 f"<sup>R² Posição: {r2_pos:.4f} | R² Velocidade: {r2_vel:.4f}</sup>",
            x=0.5,
            font=dict(size=14)
        ),
        xaxis_title="Posição (m)",
        yaxis_title="Velocidade (m/s)",
        width=1400,
        height=1000,
        legend=dict(
            title="Legenda",
            x=0.75,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest',
        plot_bgcolor='black',
        paper_bgcolor='black',
        xaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1,
            title_font_color='white',
            tickfont_color='white'
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1,
            title_font_color='white',
            tickfont_color='white'
        ),
        title_font_color='white'
    )
    
    return fig

def cria_grafico_interpolacao_completo(
    tempos_lista,
    posicoes_lista,
    velocidades_lista,
    casos_info,
    titulo="Interpolação do Modelo: Posição e Velocidade vs Tempo"
):
    """
    Cria gráfico 2D combinando posição e velocidade no tempo.
    
    Args:
        tempos_lista: Lista de arrays com os tempos para cada caso
        posicoes_lista: Lista de arrays com as posições previstas para cada caso
        velocidades_lista: Lista de arrays com as velocidades previstas para cada caso
        casos_info: Lista de dicionários com informações dos casos (nome, x0, v0, omega, cor, nome_legenda)
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
    
    for i, (tempos, posicoes, velocidades) in enumerate(zip(tempos_lista, posicoes_lista, velocidades_lista)):
        caso = casos_info[i]
        
        nome_legenda = caso.get('nome_legenda', caso['nome'])
        
        cor_velocidade = clarear_cor(caso['cor'], fator=0.6)
        
        # posição
        fig.add_trace(go.Scatter(
            x=tempos,
            y=posicoes,
            mode='lines',
            name=f"{nome_legenda} - Posição",
            line=dict(color=caso['cor'], width=2, dash='solid'),
            legendgroup=f"posicao_{i}",
            hovertemplate=(
                f"<b>{nome_legenda} - Posição</b><br>" +
                f"x₀ = {caso['x0']:.3f} m<br>" +
                f"v₀ = {caso['v0']:.3f} m/s<br>" +
                f"ω = {caso['omega']:.3f} rad/s<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Posição: %{{y:.3f}} m<br>" +
                f"<extra></extra>"
            )
        ))
        
        # velocidade
        fig.add_trace(go.Scatter(
            x=tempos,
            y=velocidades,
            mode='lines',
            name=f"{nome_legenda} - Velocidade",
            line=dict(color=cor_velocidade, width=2, dash='solid'),
            legendgroup=f"velocidade_{i}",
            hovertemplate=(
                f"<b>{nome_legenda} - Velocidade</b><br>" +
                f"x₀ = {caso['x0']:.3f} m<br>" +
                f"v₀ = {caso['v0']:.3f} m/s<br>" +
                f"ω = {caso['omega']:.3f} rad/s<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Velocidade: %{{y:.3f}} m/s<br>" +
                f"<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title=dict(
            text=f"{titulo}<br>",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="Posição (m) / Velocidade (m/s)",
        width=1400,
        height=900,
        legend=dict(
            x=1.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest',
        plot_bgcolor='black',
        paper_bgcolor='black',
        xaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1,
            title_font_color='white',
            tickfont_color='white'
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1,
            title_font_color='white',
            tickfont_color='white'
        ),
        title_font_color='white'
    )
    
    return fig


def cria_grafico_interpolacao_espaco_fases(
    posicoes_lista,
    velocidades_lista,
    casos_info,
    titulo="Interpolação do Modelo no Espaço de Fases"
):
    """
    Cria gráfico 2D mostrando as trajetórias previstas no espaço de fases.
    
    Args:
        posicoes_lista: Lista de arrays com as posições previstas para cada caso
        velocidades_lista: Lista de arrays com as velocidades previstas para cada caso
        casos_info: Lista de dicionários com informações dos casos (nome, x0, v0, omega, cor, nome_legenda)
        titulo: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    for i, (posicoes, velocidades) in enumerate(zip(posicoes_lista, velocidades_lista)):
        caso = casos_info[i]
        
        nome_legenda = caso.get('nome_legenda', caso['nome'])
        
        # trajetória
        fig.add_trace(go.Scatter(
            x=posicoes,
            y=velocidades,
            mode='lines',
            name=nome_legenda,
            line=dict(color=caso['cor'], width=2),
            hovertemplate=(
                f"<b>{nome_legenda}</b><br>" +
                f"x₀ = {caso['x0']:.3f} m<br>" +
                f"v₀ = {caso['v0']:.3f} m/s<br>" +
                f"ω = {caso['omega']:.3f} rad/s<br>" +
                f"Posição: %{{x:.3f}} m<br>" +
                f"Velocidade: %{{y:.3f}} m/s<br>" +
                f"<extra></extra>"
            )
        ))
        
        # ponto inicial
        fig.add_trace(go.Scatter(
            x=[posicoes[0]],
            y=[velocidades[0]],
            mode='markers',
            marker=dict(
                color=caso['cor'],
                size=10,
                symbol='circle',
                line=dict(color='white', width=1)
            ),
            name=f"Início - {nome_legenda}",
            showlegend=False,
            hovertemplate=(
                f"<b>Condição Inicial - {nome_legenda}</b><br>" +
                f"x₀ = {caso['x0']:.3f} m<br>" +
                f"v₀ = {caso['v0']:.3f} m/s<br>" +
                f"<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title=dict(
            text=f"{titulo}<br>",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="Posição (m)",
        yaxis_title="Velocidade (m/s)",
        width=1400,
        height=1000,
        legend=dict(
            x=0.75,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest',
        plot_bgcolor='black',
        paper_bgcolor='black',
        xaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1,
            title_font_color='white',
            tickfont_color='white'
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1,
            title_font_color='white',
            tickfont_color='white'
        ),
        title_font_color='white'
    )
    
    return fig

def cria_grafico_interpolacao_pontual_mlp(
    predictions,
    y_true,
    titulo="Interpolação Pontual de Posição e Velocidade"
):
    """
    Cria gráficos de dispersão para visualizar a interpolação pontual do modelo MLP.
    
    Args:
        predictions: array com as previsões (n_samples, 2) - [x, v]
        y_true: array com os valores reais (n_samples, 2) - [x, v]
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
    unidades = ['m', 'm/s']
    
    for i in range(2):
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
    
    rmse_pos = np.sqrt(mean_squared_error(y_true[:, 0], predictions[:, 0]))
    rmse_vel = np.sqrt(mean_squared_error(y_true[:, 1], predictions[:, 1]))
    r2_pos = r2_score(y_true[:, 0], predictions[:, 0])
    r2_vel = r2_score(y_true[:, 1], predictions[:, 1])
    
    fig.update_layout(
        title=dict(
            text=f"{titulo}<br>" +
                 f"<sup>RMSE Posição: {rmse_pos:.4f} m | RMSE Velocidade: {rmse_vel:.4f} m/s</sup><br>" +
                 f"<sup>R² Posição: {r2_pos:.4f} | R² Velocidade: {r2_vel:.4f}</sup>",
            x=0.5,
            font=dict(size=16)
        ),
        width=1000,
        height=500,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    return fig


def cria_grafico_interpolacao_pontual_espaco_fases(
    y_pos_true, y_vel_true,
    y_pos_pred, y_vel_pred,
    titulo="Interpolação Pontual no Espaço de Fases"
):
    """
    Cria gráfico 2D mostrando a interpolação pontual do modelo no espaço de fases.
    
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
    
    # previsões do modelo
    fig.add_trace(go.Scatter(
        x=y_pos_pred.flatten(),
        y=y_vel_pred.flatten(),
        mode='markers',
        name='MLP',
        marker=dict(
            color='#FF4500',
            size=3,
            opacity=0.6,
            symbol='diamond'
        ),
        hovertemplate=(
            f"<b>MLP</b><br>" +
            f"Posição: %{{x:.3f}} m<br>" +
            f"Velocidade: %{{y:.3f}} m/s<br>" +
            f"<extra></extra>"
        )
    ))

    # dados reais
    fig.add_trace(go.Scatter(
        x=y_pos_true.flatten(),
        y=y_vel_true.flatten(),
        mode='markers',
        name='Solução Analítica',
        marker=dict(
            color='#00BFFF',
            size=3,
            opacity=0.6,
            symbol='circle'
        ),
        hovertemplate=(
            f"<b>Solução Analítica</b><br>" +
            f"Posição: %{{x:.3f}} m<br>" +
            f"Velocidade: %{{y:.3f}} m/s<br>" +
            f"<extra></extra>"
        )
    ))

    rmse_pos = np.sqrt(mean_squared_error(y_pos_true.flatten(), y_pos_pred.flatten()))
    rmse_vel = np.sqrt(mean_squared_error(y_vel_true.flatten(), y_vel_pred.flatten()))
    r2_pos = r2_score(y_pos_true.flatten(), y_pos_pred.flatten())
    r2_vel = r2_score(y_vel_true.flatten(), y_vel_pred.flatten())
    
    fig.update_layout(
        title=dict(
            text=f"{titulo}<br>" +
                 f"<sup>RMSE Posição: {rmse_pos:.4f} m | RMSE Velocidade: {rmse_vel:.4f} m/s</sup><br>" +
                 f"<sup>R² Posição: {r2_pos:.4f} | R² Velocidade: {r2_vel:.4f}</sup>",
            x=0.5,
            font=dict(size=14)
        ),
        xaxis_title="Posição (m)",
        yaxis_title="Velocidade (m/s)",
        width=1400,
        height=1000,
        legend=dict(
            title="Legenda",
            x=0.75,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest',
        plot_bgcolor='black',
        paper_bgcolor='black',
        xaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1,
            title_font_color='white',
            tickfont_color='white'
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1,
            title_font_color='white',
            tickfont_color='white'
        ),
        title_font_color='white'
    )
    
    return fig

def cria_grafico_interpolacao_pontual_completo(
    tempos_lista,
    posicoes_previstas_lista,
    velocidades_previstas_lista,
    posicoes_reais_lista,
    velocidades_reais_lista,
    casos_info,
    titulo="Interpolação Pontual: Posição e Velocidade vs Tempo"
):
    """
    Cria gráfico 2D combinando posição e velocidade no tempo para interpolação pontual.
    
    Args:
        tempos_lista: Lista de arrays com os tempos interpolados para cada caso
        posicoes_previstas_lista: Lista de arrays com as posições previstas pelo MLP
        velocidades_previstas_lista: Lista de arrays com as velocidades previstas pelo MLP
        posicoes_reais_lista: Lista de arrays com as posições reais (interpolação linear)
        velocidades_reais_lista: Lista de arrays com as velocidades reais (interpolação linear)
        casos_info: Lista de dicionários com informações dos casos (x0, v0, omega, cor)
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
        
        nome_sistema = f"ω={caso['omega']:.1f} rad/s, x₀={caso['x0']:.2f}m, v₀={caso['v0']:.2f}m/s"
        
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
                f"<b>{nome_sistema} - Posição (MLP)</b><br>" +
                f"x₀ = {caso['x0']:.3f} m<br>" +
                f"v₀ = {caso['v0']:.3f} m/s<br>" +
                f"ω = {caso['omega']:.3f} rad/s<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Posição Prevista: %{{y:.3f}} m<br>" +
                f"<extra></extra>"
            )
        ))
        
        # posição real (interpolação linear)
        fig.add_trace(go.Scatter(
            x=tempos,
            y=pos_real,
            mode='lines',
            name=f"{nome_sistema} - Posição (Real)",
            line=dict(color=caso['cor'], width=1.5, dash='dot'),
            legendgroup=f"posicao_real_{i}",
            hovertemplate=(
                f"<b>{nome_sistema} - Posição (Real)</b><br>" +
                f"x₀ = {caso['x0']:.3f} m<br>" +
                f"v₀ = {caso['v0']:.3f} m/s<br>" +
                f"ω = {caso['omega']:.3f} rad/s<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Posição Real: %{{y:.3f}} m<br>" +
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
                f"<b>{nome_sistema} - Velocidade (MLP)</b><br>" +
                f"x₀ = {caso['x0']:.3f} m<br>" +
                f"v₀ = {caso['v0']:.3f} m/s<br>" +
                f"ω = {caso['omega']:.3f} rad/s<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Velocidade Prevista: %{{y:.3f}} m/s<br>" +
                f"<extra></extra>"
            )
        ))
        
        # velocidade real (interpolação linear)
        fig.add_trace(go.Scatter(
            x=tempos,
            y=vel_real,
            mode='lines',
            name=f"{nome_sistema} - Velocidade (Real)",
            line=dict(color=cor_velocidade, width=1.5, dash='dot'),
            legendgroup=f"velocidade_real_{i}",
            hovertemplate=(
                f"<b>{nome_sistema} - Velocidade (Real)</b><br>" +
                f"x₀ = {caso['x0']:.3f} m<br>" +
                f"v₀ = {caso['v0']:.3f} m/s<br>" +
                f"ω = {caso['omega']:.3f} rad/s<br>" +
                f"Tempo: %{{x:.3f}} s<br>" +
                f"Velocidade Real: %{{y:.3f}} m/s<br>" +
                f"<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title=dict(
            text=f"{titulo}<br>",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="Posição (m) / Velocidade (m/s)",
        width=1400,
        height=900,
        legend=dict(
            x=1.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest',
        plot_bgcolor='black',
        paper_bgcolor='black',
        xaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1,
            title_font_color='white',
            tickfont_color='white'
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='darkgray',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1,
            title_font_color='white',
            tickfont_color='white'
        ),
        title_font_color='white'
    )
    
    return fig


def cria_grafico_interpolacao_entre_trajetorias_espaco_fases(
    trajetoria1_pos,
    trajetoria1_vel,
    trajetoria2_pos,
    trajetoria2_vel,
    interpolacoes_lista,
    casos_info,
    titulo="Interpolação entre Trajetórias no Espaço de Fases",
    cores_paleta=CORES_PALETA
):
    """
    Cria gráfico 2D mostrando as duas trajetórias originais e as trajetórias interpoladas no espaço de fases.
    
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
        name=f"Trajetória 1 (α=0): x₀={caso_info.get('x0_1', 0):.3f} m, v₀={caso_info.get('v0_1', 0):.3f} m/s",
        line=dict(color='blue', width=3, dash='solid'),
        hovertemplate=(
            f"<b>Trajetória 1 (α=0)</b><br>" +
            f"x₀ = {caso_info.get('x0_1', 0):.3f} m<br>" +
            f"v₀ = {caso_info.get('v0_1', 0):.3f} m/s<br>" +
            f"Posição: %{{x:.3f}} m<br>" +
            f"Velocidade: %{{y:.3f}} m/s<br>" +
            f"<extra></extra>"
        )
    ))
    
    # trajetória 2 (alpha = 1)
    fig.add_trace(go.Scatter(
        x=trajetoria2_pos,
        y=trajetoria2_vel,
        mode='lines',
        name=f"Trajetória 2 (α=1): x₀={caso_info.get('x0_2', 0):.3f} m, v₀={caso_info.get('v0_2', 0):.3f} m/s",
        line=dict(color='red', width=3, dash='solid'),
        hovertemplate=(
            f"<b>Trajetória 2 (α=1)</b><br>" +
            f"x₀ = {caso_info.get('x0_2', 0):.3f} m<br>" +
            f"v₀ = {caso_info.get('v0_2', 0):.3f} m/s<br>" +
            f"Posição: %{{x:.3f}} m<br>" +
            f"Velocidade: %{{y:.3f}} m/s<br>" +
            f"<extra></extra>"
        )
    ))
    
    # trajetórias interpoladas (0 < alpha < 1)
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
            
            # linha da trajetória interpolada - mostra x0 e v0 na legenda
            fig.add_trace(go.Scatter(
                x=posicoes,
                y=velocidades,
                mode='lines',
                name=f"Trajetória Interpolada (α={alpha:.2f}): x₀={x0_interp:.3f} m, v₀={v0_interp:.3f} m/s",
                line=dict(color=cor, width=3, dash='dash'),
                opacity=0.7,
                hovertemplate=(
                    f"<b>Interpolação α={alpha:.2f}</b><br>" +
                    f"x₀_interp = {x0_interp:.3f} m<br>" +
                    f"v₀_interp = {v0_interp:.3f} m/s<br>" +
                    f"Posição: %{{x:.3f}} m<br>" +
                    f"Velocidade: %{{y:.3f}} m/s<br>" +
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
                name=f"Início α={alpha:.2f}",
                showlegend=False,
                hovertemplate=(
                    f"<b>Condição Inicial α={alpha:.2f}</b><br>" +
                    f"x₀ = {x0_interp:.3f} m<br>" +
                    f"v₀ = {v0_interp:.3f} m/s<br>" +
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
        hovertemplate=f"<b>Início Trajetória 1</b><br>x₀ = {caso_info.get('x0_1', 0):.3f} m<br>v₀ = {caso_info.get('v0_1', 0):.3f} m/s<br><extra></extra>"
    ))
    
    # ponto inicial da trajetória 2
    fig.add_trace(go.Scatter(
        x=[trajetoria2_pos[0]],
        y=[trajetoria2_vel[0]],
        mode='markers',
        marker=dict(color='red', size=12, symbol='circle', line=dict(color='white', width=2)),
        name="Início Trajetória 2",
        showlegend=False,
        hovertemplate=f"<b>Início Trajetória 2</b><br>x₀ = {caso_info.get('x0_2', 0):.3f} m<br>v₀ = {caso_info.get('v0_2', 0):.3f} m/s<br><extra></extra>"
    ))

    fig.add_annotation(
        x=-0.02,
        y=1.05,
        xref="paper",
        yref="paper",
        text="<b>Interpolação Linear entre Condições Iniciais:</b><br>" +
             "x₀(α) = (1-α)·x₀₁ + α·x₀₂<br>" +
             "v₀(α) = (1-α)·v₀₁ + α·v₀₂",
        showarrow=False,
        font=dict(size=12, color='black'),
        bgcolor='white',
        bordercolor='black',
        borderwidth=1,
        borderpad=4,
        align='left'
    )
     
    fig.update_layout(
        title=dict(
            text=f"{titulo}<br><sub>Interpolação entre duas trajetórias no espaço de fases</sub>",
            x=0.5,
            font=dict(size=16, color='white')
        ),
        xaxis_title="Posição (m)",
        yaxis_title="Velocidade (m/s)",
        width=1400,
        height=1000,
        legend=dict(
            x=0.95,
            y=1.05,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(color='black', size=10)
        ),
        hovermode='closest',
        plot_bgcolor='black',
        paper_bgcolor='black',
        xaxis=dict(
            showgrid=True,
            gridcolor='darkgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1,
            title_font_color='white',
            tickfont_color='white'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='darkgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1,
            title_font_color='white',
            tickfont_color='white'
        ),
        title_font_color='white'
    )
    
    return fig