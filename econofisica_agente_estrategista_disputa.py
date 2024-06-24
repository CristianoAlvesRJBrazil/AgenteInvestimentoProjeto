import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class InvestidorEstrategista:
    def __init__(self, short_window=5, long_window=20, alpha_short=0.2, alpha_long=0.1):
        self.short_window = short_window
        self.long_window = long_window
        self.alpha_short = alpha_short
        self.alpha_long = alpha_long
        self.prices = []
        self.ema_short = None
        self.ema_long = None
        self.cash = 10000  # Capital inicial em dinheiro
        self.stocks = 0    # Inicialmente não possui ações
        self.valor_total = self.cash  # Valor total inicial é o capital em dinheiro
        self.historical_data = pd.DataFrame(columns=['Price', 'EMA_Short', 'EMA_Long', 'Action'])

    def aplicar_estrategia(self, precos_historicos):
        for price in precos_historicos:
            self.prices.append(price)

            if len(self.prices) >= self.short_window:
                self.ema_short = self.calcular_ema(self.prices[-self.short_window:], self.alpha_short)

            if len(self.prices) >= self.long_window:
                self.ema_long = self.calcular_ema(self.prices[-self.long_window:], self.alpha_long)

            action = None

            if self.ema_short is not None and self.ema_long is not None:
                current_price = price

                if self.ema_short > self.ema_long:
                    if self.cash > current_price:
                        quantity = int(self.cash / current_price)
                        self.cash -= quantity * current_price
                        self.stocks += quantity
                        action = 'Compra'

                elif self.ema_short < self.ema_long:
                    if self.stocks > 0:
                        self.cash += self.stocks * current_price
                        self.stocks = 0
                        action = 'Venda'

                self.valor_total = self.cash + self.stocks * current_price

            self.historical_data = pd.concat([
                self.historical_data,
                pd.DataFrame({
                    'Price': [price],
                    'EMA_Short': [self.ema_short],
                    'EMA_Long': [self.ema_long],
                    'Action': [action]
                })
            ], ignore_index=True)

    def calcular_ema(self, precos, alpha):
        ema = precos[0]
        for preco in precos[1:]:
            ema = alpha * preco + (1 - alpha) * ema
        return ema

    def plot_strategy(self, ticker):
        plt.figure(figsize=(14, 7))

        plt.plot(self.historical_data['Price'], label='Preço de Fechamento')
        plt.plot(self.historical_data['EMA_Short'], label=f'EMA Curta ({self.short_window})', linestyle='--')
        plt.plot(self.historical_data['EMA_Long'], label=f'EMA Longa ({self.long_window})', linestyle='--')

        compra = self.historical_data[self.historical_data['Action'] == 'Compra']
        venda = self.historical_data[self.historical_data['Action'] == 'Venda']

        plt.scatter(compra.index, compra['Price'], label='Compra', marker='^', color='green', alpha=1)
        plt.scatter(venda.index, venda['Price'], label='Venda', marker='v', color='red', alpha=1)

        plt.title(f'Estratégia de EMA para {ticker}')
        plt.xlabel('Tempo')
        plt.ylabel('Preço')
        plt.legend()
        plt.grid(True)
        plt.show()

class InvestidorEstrategistaRSI:
    def __init__(self, rsi_period=14, rsi_buy_threshold=30, rsi_sell_threshold=70):
        self.rsi_period = rsi_period
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.prices = []
        self.rsi = None
        self.cash = 10000  # Capital inicial em dinheiro
        self.stocks = 0    # Inicialmente não possui ações
        self.valor_total = self.cash  # Valor total inicial é o capital em dinheiro
        self.historical_data = pd.DataFrame(columns=['Price', 'RSI', 'Action'])

    def aplicar_estrategia(self, precos_historicos):
        for price in precos_historicos:
            self.prices.append(price)

            if len(self.prices) > self.rsi_period:
                self.rsi = self.calcular_rsi(self.prices[-(self.rsi_period+1):])

            action = None

            if self.rsi is not None:
                current_price = price

                if self.rsi < self.rsi_buy_threshold:
                    if self.cash > current_price:
                        quantity = int(self.cash / current_price)
                        self.cash -= quantity * current_price
                        self.stocks += quantity
                        action = 'Compra'

                elif self.rsi > self.rsi_sell_threshold:
                    if self.stocks > 0:
                        self.cash += self.stocks * current_price
                        self.stocks = 0
                        action = 'Venda'

                self.valor_total = self.cash + self.stocks * current_price

            self.historical_data = pd.concat([
                self.historical_data,
                pd.DataFrame({
                    'Price': [price],
                    'RSI': [self.rsi],
                    'Action': [action]
                })
            ], ignore_index=True)

    def calcular_rsi(self, precos):
        delta = np.diff(precos)
        gain = np.maximum(delta, 0)
        loss = -np.minimum(delta, 0)

        avg_gain = np.mean(gain[:self.rsi_period])
        avg_loss = np.mean(loss[:self.rsi_period])

        if avg_loss == 0:
            rs = float('inf')
        else:
            rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))
        return rsi

    def plot_strategy(self, ticker):
        fig, axs = plt.subplots(2, figsize=(14, 10))

        axs[0].plot(self.historical_data['Price'], label='Preço de Fechamento')
        axs[0].set_title(f'Estratégia de RSI para {ticker}')
        axs[0].set_xlabel('Tempo')
        axs[0].set_ylabel('Preço')
        axs[0].legend()
        axs[0].grid(True)

        compra = self.historical_data[self.historical_data['Action'] == 'Compra']
        venda = self.historical_data[self.historical_data['Action'] == 'Venda']

        axs[0].scatter(compra.index, compra['Price'], label='Compra', marker='^', color='green', alpha=1)
        axs[0].scatter(venda.index, venda['Price'], label='Venda', marker='v', color='red', alpha=1)

        axs[1].plot(self.historical_data['RSI'], label='RSI', color='purple')
        axs[1].axhline(30, linestyle='--', alpha=0.5, color='red', label='Sobrevendido (30)')
        axs[1].axhline(70, linestyle='--', alpha=0.5, color='green', label='Sobrecomprado (70)')
        axs[1].set_xlabel('Tempo')
        axs[1].set_ylabel('RSI')
        axs[1].legend()
        axs[1].grid(True)

        plt.show()

if __name__ == "__main__":
    # Configurações para os dois investidores
    short_window = 5
    long_window = 20
    alpha_short = 0.2
    alpha_long = 0.1
    rsi_period = 14
    rsi_buy_threshold = 30
    rsi_sell_threshold = 70

    # Obtendo os dados históricos da PETR4.SA nos últimos 3 anos
    inicio_dados = '2024-01-01'
    final_dados = '2024-06-23'
    Carteira_Vencedora = ['EMBR3.SA', 'TEND3.SA', 'SOMA3.SA', 'ECOR3.SA']
    ticker = Carteira_Vencedora[3]
    data = yf.download(ticker, start=inicio_dados, end=final_dados)['Close']

    # Criando os objetos dos investidores
    estrategista_ema = InvestidorEstrategista(short_window, long_window, alpha_short, alpha_long)
    estrategista_rsi = InvestidorEstrategistaRSI(rsi_period, rsi_buy_threshold, rsi_sell_threshold)

    # Aplicando as estratégias com os dados históricos
    estrategista_ema.aplicar_estrategia(data)
    estrategista_rsi.aplicar_estrategia(data)

    # Calculando os lucros finais
    lucro_final_ema = estrategista_ema.valor_total - 10000  # 10000 é o capital inicial
    lucro_final_rsi = estrategista_rsi.valor_total - 10000      # 10000 é o capital inicial

    # Exibindo os resultados
    print(f"Lucro do Investidor Estrategista (EMA) para {ticker}: {lucro_final_ema:.2f} BRL")
    print(f"Lucro do Investidor Estrategista (RSI) para {ticker}: {lucro_final_rsi:.2f} BRL")

    # Plotando os gráficos para visualização das estratégias
    estrategista_ema.plot_strategy(ticker)
    estrategista_rsi.plot_strategy(ticker)





