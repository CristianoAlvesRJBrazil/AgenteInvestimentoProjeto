#!pip install yfinance

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Função para buscar dados históricos do yfinance
def fetch_historical_data(ticker, inicio_dados, final_dados):
    stock_data = yf.download(ticker, start=inicio_dados, end=final_dados)['Close'].mean(axis=1)
    return stock_data

# Função para prever futuras contagens usando o método de Monte Carlo
def monte_carlo_prediction(prices, num_simulations=1000, num_days=90):
    log_returns = np.log(1 + prices.pct_change())
    mean_return = log_returns.mean()
    std_return = log_returns.std()

    simulations = []

    for _ in range(num_simulations):
        simulation_prices = []
        last_price = prices.iloc[-1]

        for _ in range(num_days):
            next_price = last_price * np.exp(mean_return + std_return * np.random.normal())
            simulation_prices.append(next_price)
            last_price = next_price

        simulations.append(simulation_prices)

    return simulations

def main(ticker, inicio_dados, final_dados):

    # Buscar dados históricos
    historical_prices = fetch_historical_data(ticker, inicio_dados, final_dados)

    # Realizar previsões usando o método de Monte Carlo
    num_simulations = 1000
    num_days = 60
    predicted_prices = monte_carlo_prediction(historical_prices, num_simulations, num_days)

    # Plotar os dados históricos e as previsões
    plt.figure(figsize=(14, 7))
    plt.plot(historical_prices.index, historical_prices, label='Histórico de Preços')

    # Adicionar previsões ao gráfico
    last_date = historical_prices.index[-1]
    future_dates = pd.date_range(start=last_date, periods=num_days+1, inclusive='right')

    for simulation in predicted_prices:
        plt.plot(future_dates, simulation, color='red', alpha=0.01)

    plt.title(f"Simulação de Monte Carlo para {ticker} - Próximos {num_days} dias")
    plt.xlabel("Data")
    plt.ylabel("Preço de Fechamento")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    
    # Configurações iniciais
    ticker = ['EMBR3.SA', 'PETR4.SA']  # Substitua pelo ticker desejado
    inicio_dados = '2024-01-01'
    final_dados = '2024-05-01'
    main(ticker, inicio_dados, final_dados)
    
    
    