import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Função para obter dados históricos de preços
def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# Função para calcular médias móveis
def calculate_moving_averages(data, short_window, long_window):
    data['SMA_short'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    return data

# Função para identificar tendências de alta
def identify_uptrends(data):
    data['Uptrend'] = data['SMA_short'] > data['SMA_long']
    return data

# Parâmetros do ativo e período
#Ativos = ['ECOR3.SA', 'MGLU3.SA', 'SMTO3.SA', 'TEND3.SA']
Ativos = ['MULT3.SA', 'BRFS3.SA', 'ENAT3.SA', 'EMBR3.SA']
ticker = Ativos[3]
start_date = '2023-01-01'
end_date = '2024-08-01'
short_window = 9
long_window = 21

# Obter dados do ativo
stock_data = get_stock_data(ticker, start_date, end_date)

# Calcular médias móveis
stock_data = calculate_moving_averages(stock_data, short_window, long_window)

# Identificar tendências de alta
stock_data = identify_uptrends(stock_data)

# Criar gráfico dinâmico
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

# Adicionar trace para o preço de fechamento
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Preço de Fechamento', line=dict(color='black')), row=1, col=1)

# Adicionar trace para SMA de curto prazo
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_short'], mode='lines', name=f'SMA {short_window} Dias', line=dict(color='blue')), row=1, col=1)

# Adicionar trace para SMA de longo prazo
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_long'], mode='lines', name=f'SMA {long_window} Dias', line=dict(color='red')), row=1, col=1)

# Adicionar área sombreada para a tendência de alta
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Tendência de Alta', line=dict(color='green', width=0), fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.3)', showlegend=True, opacity=0.3), row=1, col=1)

# Atualizar layout do gráfico
fig.update_layout(
    title=f'Tendência de Alta para {ticker}',
    xaxis_title='Data',
    yaxis_title='Preço',
    legend_title='Legenda',
    hovermode='x'
)

# Mostrar gráfico
fig.show()


