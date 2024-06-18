import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class StockPredictor:
    def __init__(self, ticker, data_inicio, data_fim, janela=180):
        self.ticker = ticker
        self.data_inicio = data_inicio
        self.data_fim = data_fim
        self.janela = janela
        self.dados = self.obter_dados()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.modelo = None

    def obter_dados(self):
        """Baixa dados históricos dos preços de fechamento do ativo."""
        dados = yf.download(self.ticker, start=self.data_inicio, end=self.data_fim)['Close'].mean(axis=1)
        return dados

    def preprocessar_dados(self):
        """Preprocessa os dados para serem usados na rede neural."""
        dados_escala = self.scaler.fit_transform(self.dados.values.reshape(-1, 1))

        X_train, y_train = [], []
        for i in range(self.janela, len(dados_escala)):
            X_train.append(dados_escala[i - self.janela:i, 0])
            y_train.append(dados_escala[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        return X_train, y_train

    def construir_modelo(self):
        """Constrói e compila o modelo LSTM."""
        modelo = Sequential()
        modelo.add(LSTM(units=100, return_sequences=True, input_shape=(self.janela, 1)))
        modelo.add(Dropout(0.2))
        modelo.add(LSTM(units=250, return_sequences=False))
        modelo.add(Dropout(0.2))
        modelo.add(Dense(units=1))

        modelo.compile(optimizer='adam', loss='mean_squared_error')
        return modelo
    
    def treinar_modelo(self, X_train, y_train, epochs=5, batch_size=64):
        """Treina o modelo LSTM."""
        self.modelo = self.construir_modelo()
        self.modelo.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def prever_precos_futuros(self, dias_previsao=60):
        """Prevê os preços futuros começando do último preço conhecido."""
        sequencia_recente = self.dados[-self.janela:].values
        sequencia_recente_escala = self.scaler.transform(sequencia_recente.reshape(-1, 1))

        previsoes = []
        for _ in range(dias_previsao):
            X_input = np.array(sequencia_recente_escala[-self.janela:]).reshape(1, self.janela, 1)
            predicao_escala = self.modelo.predict(X_input)
            predicao = self.scaler.inverse_transform(predicao_escala)
            previsoes.append(predicao[0, 0])

            # Adiciona a previsão ao final da sequência e remove o primeiro elemento
            sequencia_recente_escala = np.append(sequencia_recente_escala, predicao_escala, axis=0)

        return previsoes

    def simular_monte_carlo(self, dias_previsao=60, num_simulacoes=1000):
        """Executa Simulação de Monte Carlo baseada nas previsões da RNN."""
        previsao_base = self.prever_precos_futuros(dias_previsao)
        retorno_diario = np.diff(np.log(previsao_base))  # Calcular retornos log diários
        media_ret = np.mean(retorno_diario)
        std_ret = np.std(retorno_diario)

        previsoes_mc = np.zeros((num_simulacoes, dias_previsao))
        ultimo_preco_conhecido = self.dados.values[-1]  # Último preço conhecido

        for i in range(num_simulacoes):
            simulacao = []
            ultimo_preco = ultimo_preco_conhecido
            for _ in range(dias_previsao):
                variacao = np.random.normal(media_ret, std_ret)
                ultimo_preco = ultimo_preco * np.exp(variacao)
                simulacao.append(ultimo_preco)
            previsoes_mc[i, :] = simulacao

        return previsoes_mc

    def plotar_previsao_com_mc(self, dias_previsao=60, num_simulacoes=1000):
        """Plota os preços reais, previstos e as simulações de Monte Carlo."""
        previsao_base = self.prever_precos_futuros(dias_previsao)
        previsoes_mc = self.simular_monte_carlo(dias_previsao, num_simulacoes)

        plt.figure(figsize=(14, 7))
        plt.plot(self.dados.values, color='blue', label='Preço Real')
        plt.plot(range(len(self.dados), len(self.dados) + dias_previsao), previsao_base, color='red', label='Previsão de Preço')

        # Plotar as simulações de Monte Carlo
        for sim in previsoes_mc:
            plt.plot(range(len(self.dados), len(self.dados) + dias_previsao), sim, color='gray', alpha=0.005)

        plt.title(f'Previsão de Preços para {self.ticker} com Monte Carlo')
        plt.xlabel('Data')
        plt.ylabel('Preço')
        plt.legend()
        plt.show()
        
        return previsao_base[-1], previsoes_mc.max().max()

def main(ticker, data_inicio, data_fim):
    
    # Criação do previsor de ações
    previsor = StockPredictor(ticker, data_inicio, data_fim)

    # Preprocessar dados e treinar o modelo
    X_train, y_train = previsor.preprocessar_dados()
    previsor.treinar_modelo(X_train, y_train)

    # Prever preços futuros e plotar os resultados com Monte Carlo
    #previsor.plotar_previsao_com_mc(dias_previsao=60, num_simulacoes=1000)
    
    ultimo_dado = previsor.obter_dados()
    ultimo_dado_real = ultimo_dado[-1]
    ultimo_dado_simulado, ultimo_dado_simulado_mc = previsor.plotar_previsao_com_mc()
    
    print(f"Ultimo Real: {ultimo_dado_real}")
    print(f"Ultimo Simulado: {ultimo_dado_simulado}")
    print(f"Ultimo Simulado MC: {ultimo_dado_simulado_mc}")
    
if __name__ == "__main__":
    
    # Parâmetros de entrada
    df = pd.read_csv('Resultados/resultados01.csv')
    title_column = df.columns[0]
    ticker = [title_column[2:10], title_column[14:22], title_column[26:34], title_column[38:46]] # Substitua pelo ticker desejado
    data_inicio = '2022-01-02'
    data_fim = '2023-05-02'
    
    main(ticker, data_inicio, data_fim)

