import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class InvestidorEstrategistaReversaoMedia:
    def __init__(self, window=20, threshold=2, capital_inicial=10000):
        self.window = window
        self.threshold = threshold
        self.cash = capital_inicial
        self.stocks = 0
        self.valor_total = self.cash
        self.historical_data = pd.DataFrame(columns=['Price', 'SMA', 'Upper', 'Lower', 'Volume', 'Action'])

    def calcular_sma_e_bandas(self, precos):
        precos_series = pd.Series(precos)
        sma = precos_series.rolling(window=self.window).mean()
        std = precos_series.rolling(window=self.window).std()
        upper_band = sma + self.threshold * std
        lower_band = sma - self.threshold * std
        return sma, upper_band, lower_band

    def aplicar_estrategia(self, precos_historicos, volume_historico):
        for i in range(len(precos_historicos)):
            price = precos_historicos[i]
            volume = volume_historico[i]

            if i >= self.window - 1:
                sma, upper_band, lower_band = self.calcular_sma_e_bandas(precos_historicos[:i+1])
                action = None

                if price < lower_band[i] and volume > np.mean(volume_historico[:i+1]):
                    if self.cash > price:
                        quantity = int(self.cash / price)
                        self.cash -= quantity * price
                        self.stocks += quantity
                        action = 'Compra'

                elif price > upper_band[i] and volume > np.mean(volume_historico[:i+1]):
                    if self.stocks > 0:
                        self.cash += self.stocks * price
                        self.stocks = 0
                        action = 'Venda'

                self.valor_total = self.cash + self.stocks * price

                self.historical_data = pd.concat([self.historical_data, pd.DataFrame({
                    'Price': [price],
                    'SMA': [sma[i]],
                    'Upper': [upper_band[i]],
                    'Lower': [lower_band[i]],
                    'Volume': [volume],
                    'Action': [action]
                })], ignore_index=True)

    def plot_strategy(self, ticker):
        plt.figure(figsize=(14, 7))

        plt.plot(self.historical_data['Price'], label='Preço de Fechamento')
        plt.plot(self.historical_data['SMA'], label=f'SMA ({self.window})', linestyle='--')
        plt.plot(self.historical_data['Upper'], label=f'Banda Superior ({self.threshold}σ)', linestyle='--')
        plt.plot(self.historical_data['Lower'], label=f'Banda Inferior ({self.threshold}σ)', linestyle='--')

        compra = self.historical_data[self.historical_data['Action'] == 'Compra']
        venda = self.historical_data[self.historical_data['Action'] == 'Venda']

        plt.scatter(compra.index, compra['Price'], label='Compra', marker='^', color='green', alpha=1)
        plt.scatter(venda.index, venda['Price'], label='Venda', marker='v', color='red', alpha=1)

        plt.title(f'Estratégia de Reversão à Média com Volume para {ticker}')
        plt.xlabel('Tempo')
        plt.ylabel('Preço')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Configurações do Investidor Estrategista
    window = 20
    threshold = 2
    capital_inicial = 10000

    # Obtendo os dados históricos da PETR4.SA nos últimos 3 anos
    inicio_dados = '2021-01-01'
    final_dados = '2024-01-01'
    Carteira_Vencedora = ['EMBR3.SA', 'TEND3.SA', 'SOMA3.SA', 'ECOR3.SA']
    ticker = Carteira_Vencedora[3]
    dados = yf.download(ticker, start=inicio_dados, end=final_dados)
    data = dados['Close']
    volume = dados['Volume']

    # Criando o objeto do Investidor Estrategista
    estrategista_reversao_media = InvestidorEstrategistaReversaoMedia(window, threshold, capital_inicial)

    # Aplicando a estratégia com os dados históricos
    estrategista_reversao_media.aplicar_estrategia(data.values, volume.values)

    # Calculando o lucro final
    lucro_final_reversao_media = estrategista_reversao_media.valor_total - capital_inicial

    # Exibindo o resultado
    print(f"Lucro do Investidor Estrategista (Reversão à Média e Volume) para {ticker}: {lucro_final_reversao_media:.2f} BRL")

    # Plotando o gráfico para visualização da estratégia
    estrategista_reversao_media.plot_strategy(ticker)





