import yfinance as yf
import pandas as pd
import numpy as np

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

    def apply_strategy(self, historical_prices):
        for price in historical_prices:
            self.prices.append(price)

            if len(self.prices) >= self.short_window:
                self.ema_short = self.calculate_ema(self.prices[-self.short_window:], self.alpha_short)

            if len(self.prices) >= self.long_window:
                self.ema_long = self.calculate_ema(self.prices[-self.long_window:], self.alpha_long)

            if self.ema_short is not None and self.ema_long is not None:
                current_price = price

                if self.ema_short > self.ema_long:
                    if self.cash > current_price:
                        quantity = int(self.cash / current_price)
                        self.cash -= quantity * current_price
                        self.stocks += quantity

                elif self.ema_short < self.ema_long:
                    if self.stocks > 0:
                        self.cash += self.stocks * current_price
                        self.stocks = 0

                self.valor_total = self.cash + self.stocks * current_price

    def calculate_ema(self, prices, alpha):
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

if __name__ == "__main__":
    # Definindo as configurações iniciais
    short_window = 5
    long_window = 20
    alpha_short = 0.2
    alpha_long = 0.1

    # Obtendo os dados históricos da PETR4.SA nos últimos 3 anos
    inicio_dados = '2023-05-02'
    final_dados = '2023-08-02'
    Carteira_Vencedora = ['IRBR3.SA', 'EQTL3.SA', 'BBAS3.SA', 'EMBR3.SA']
    #ticker = Carteira_Vencedora
    ticker = Carteira_Vencedora[3]
    #data = yf.download(ticker, period='1y')['Close']
    #data = yf.download(ticker, start=inicio_dados, end=final_dados)['Close'].mean(axis=1)
    data = yf.download(ticker, start=inicio_dados, end=final_dados)['Close']

    # Criando o objeto do Investidor Estrategista
    estrategista = InvestidorEstrategista(short_window, long_window, alpha_short, alpha_long)

    # Aplicando a estratégia com os dados históricos
    estrategista.apply_strategy(data)

    # Calculando o lucro final
    lucro_final = estrategista.valor_total - 10000  # 10000 é o capital inicial

    # Exibindo o resultado
    print(f"Lucro do Investidor Estrategista para {ticker}: {lucro_final:.2f} BRL")
