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

    def aplicar_estrategia(self, precos_historicos):
        for preco in precos_historicos:
            self.prices.append(preco)

            if len(self.prices) >= self.short_window:
                self.ema_short = self.calcular_ema(self.prices[-self.short_window:], self.alpha_short)

            if len(self.prices) >= self.long_window:
                self.ema_long = self.calcular_ema(self.prices[-self.long_window:], self.alpha_long)

            if self.ema_short is not None and self.ema_long is not None:
                preco_atual = preco

                if self.ema_short > self.ema_long:
                    if self.cash > preco_atual:
                        quantidade = int(self.cash / preco_atual)
                        self.cash -= quantidade * preco_atual
                        self.stocks += quantidade

                elif self.ema_short < self.ema_long:
                    if self.stocks > 0:
                        self.cash += self.stocks * preco_atual
                        self.stocks = 0

                self.valor_total = self.cash + self.stocks * preco_atual

    def calcular_ema(self, precos, alpha):
        ema = precos[0]
        for preco in precos[1:]:
            ema = alpha * preco + (1 - alpha) * ema
        return ema

if __name__ == "__main__":
    # Definindo as configurações iniciais
    short_window = 5
    long_window = 20
    alpha_short = 0.2
    alpha_long = 0.1

    # Obtendo os dados históricos dos Ativos da Carteira
    inicio_dados = '2020-01-01'
    final_dados = '2024-06-01'
    carteira_vencedora = ['WEGE3.SA', 'SBSP3.SA', 'BBSE3.SA', 'CIEL3.SA']
    #ticker = carteira_vencedora
    #data = yf.download(ticker, start=inicio_dados, end=final_dados)['Close'].mean(axis=1)
        
    ticker = carteira_vencedora[0]
    data = yf.download(ticker, start=inicio_dados, end=final_dados)['Close']

    # Criando o objeto do Investidor Estrategista
    estrategista = InvestidorEstrategista(short_window, long_window, alpha_short, alpha_long)

    # Aplicando a estratégia com os dados históricos
    estrategista.aplicar_estrategia(data)

    # Calculando o lucro final
    lucro_final = estrategista.valor_total - 10000  # 10000 é o capital inicial

    # Exibindo o resultado
    print(f"Lucro do Investidor Estrategista para {ticker}: {lucro_final:.2f} BRL")

