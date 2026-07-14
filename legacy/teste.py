import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class ModeloFinanceiro:
    def __init__(self, ticker, data_inicial, data_final, tamanho_sequencia, num_simulacoes, num_dias):
        self.ticker = ticker
        self.data_inicial = data_inicial
        self.data_final = data_final
        self.tamanho_sequencia = tamanho_sequencia
        self.num_simulacoes = num_simulacoes
        self.num_dias = num_dias
        self.dados = None
        self.normalizador = MinMaxScaler(feature_range=(0, 1))
        self.modelo = None
        self.previsao = None
        self.precos_previstos = None

    def baixar_dados_acao(self):
        self.dados = yf.download(self.ticker, start=self.data_inicial, end=self.data_final)['Close']
        self.dados['Close'] = self.dados.mean(axis=1)
        self.dados = self.dados.reset_index()
        self.dados['Date'] = pd.to_datetime(self.dados['Date'])
        self.dados['Days'] = (self.dados['Date'] - self.dados['Date'].min()).dt.days
        self.dados['Close'] = self.dados['Close'].astype(float)

    def criar_sequencias(self, dados_normalizados):
        X, y = [], []
        for i in range(len(dados_normalizados) - self.tamanho_sequencia):
            seq_X = dados_normalizados[i:i + self.tamanho_sequencia, 0]
            seq_y = dados_normalizados[i + self.tamanho_sequencia, 0]
            X.append(seq_X)
            y.append(seq_y)
        X = np.array(X).reshape(-1, self.tamanho_sequencia, 1)  # Corrigir forma para (amostras, timesteps, features)
        return X, np.array(y)

    def preparar_dados(self):
        dados_normalizados = self.normalizador.fit_transform(self.dados['Close'].values.reshape(-1, 1))
        X, y = self.criar_sequencias(dados_normalizados)
        tamanho_treino = int(len(X) * 0.8)
        X_treino, X_teste = X[:tamanho_treino], X[tamanho_treino:]
        y_treino, y_teste = y[:tamanho_treino], y[tamanho_treino:]
        return X_treino, X_teste, y_treino, y_teste

    def construir_modelo(self):
        self.modelo = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(self.tamanho_sequencia, 1)),
            Dropout(0.2),
            LSTM(units=300),
            Dense(units=1)
        ])
        self.modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mean_squared_error')
    

    def treinar_modelo(self, X_treino, y_treino):
        self.modelo.fit(X_treino, y_treino, epochs=100, batch_size=32, verbose=1)

    def prever(self, X_teste):
        self.previsao = self.modelo.predict(X_teste)
        self.precos_previstos = self.normalizador.inverse_transform(self.previsao)
        return self.precos_previstos

    def avaliar_modelo(self, y_teste):
        y_teste_desnormalizado = self.normalizador.inverse_transform(y_teste.reshape(-1, 1))
        mse = mean_squared_error(y_teste_desnormalizado, self.precos_previstos)
        mae = mean_absolute_error(y_teste_desnormalizado, self.precos_previstos)
        print(f'MSE: {mse:.2f}, MAE: {mae:.2f}')
        return mse, mae

    def calcular_diferenca_variabilidade(self, y_teste):
        y_teste_desnormalizado = self.normalizador.inverse_transform(y_teste.reshape(-1, 1))
        diferenca_rnn = y_teste_desnormalizado.flatten() - self.precos_previstos.flatten()
        media_dif_rnn = np.mean(diferenca_rnn)
        desvio_padrao_rnn = np.std(diferenca_rnn)
        return media_dif_rnn, desvio_padrao_rnn

    def simulacao_monte_carlo(self, media, desvio_padrao):
        ultimo_preco = self.dados['Close'].values[-1]
        precos_simulados = np.zeros((self.num_dias, self.num_simulacoes))
        
        for i in range(self.num_simulacoes):
            retornos_diarios = np.random.normal(loc=media, scale=desvio_padrao, size=self.num_dias)
            serie_precos = [ultimo_preco]
            for j in range(self.num_dias):
                proximo_preco = serie_precos[-1] + retornos_diarios[j]
                serie_precos.append(proximo_preco)
            precos_simulados[:, i] = serie_precos[:self.num_dias]
        
        return precos_simulados

    def calcular_prob_acima_preco_atual(self, precos_simulados):
        preco_atual = self.dados['Close'].values[-1]
        contagem_acima = np.sum(precos_simulados[-1, :] > preco_atual)
        prob_acima = contagem_acima / self.num_simulacoes
        return prob_acima

    def visualizar_resultados(self, precos_simulados):
        ultima_data = self.dados['Date'].values[-1]
        datas_futuras = pd.date_range(start=ultima_data, periods=self.num_dias)
        
        plt.figure(figsize=(14, 7))
        plt.plot(self.dados['Date'], self.dados['Close'], label='Preço Real')

        indice_inicio_previsao = len(self.dados) - len(self.precos_previstos)
        plt.plot(self.dados['Date'].values[indice_inicio_previsao:], self.precos_previstos, color='blue', linestyle='--', label='Previsão RNN-LSTM')
        
        for i in range(self.num_simulacoes):
            plt.plot(datas_futuras, precos_simulados[:, i], color='grey', alpha=0.1)
        
        plt.xlabel('Data')
        plt.ylabel('Preço de Fechamento')
        plt.title(f"Simulação de Monte Carlo dos Preços Futuros da Carteira: {self.ticker} (RNN-LSTM)")
        plt.grid(True)
        plt.legend()
        plt.show()

    def estatisticas_simulacao(self, precos_simulados):
        simulacao_df_rnn = pd.DataFrame(precos_simulados)
        estatisticas_simulacao_rnn = simulacao_df_rnn.describe().mean(axis=1)
        print(estatisticas_simulacao_rnn)

    @staticmethod
    def executar(ticker, data_inicial, data_final, tamanho_sequencia, num_simulacoes, num_dias):
        # Uso da classe
        modelo = ModeloFinanceiro(
            ticker = ticker,
            data_inicial = data_inicial,
            data_final = data_final,
            tamanho_sequencia = tamanho_sequencia,
            num_simulacoes = num_simulacoes,
            num_dias = num_dias
        )

        modelo.baixar_dados_acao()
        X_treino, X_teste, y_treino, y_teste = modelo.preparar_dados()
        modelo.construir_modelo()
        modelo.treinar_modelo(X_treino, y_treino)
        precos_previstos = modelo.prever(X_teste)
        modelo.avaliar_modelo(y_teste)
        media_dif_rnn, desvio_padrao_rnn = modelo.calcular_diferenca_variabilidade(y_teste)
        precos_simulados_rnn = modelo.simulacao_monte_carlo(media_dif_rnn, desvio_padrao_rnn)
        prob_acima_preco_atual = modelo.calcular_prob_acima_preco_atual(precos_simulados_rnn)
        print(f'Probabilidade dos preços futuros estarem acima do preço atual: {prob_acima_preco_atual:.2%}')
        #modelo.visualizar_resultados(precos_simulados_rnn)
        #modelo.estatisticas_simulacao(precos_simulados_rnn)

        return precos_simulados_rnn, prob_acima_preco_atual

if __name__ == "__main__":
    # Buscando a carteira para o Backtesting, inserindo os intervalos para análise e outros
    df = pd.read_csv('Resultados/resultados01.csv')
    title_column = df.columns[0]
    tickers = [title_column[2:10], title_column[14:22], title_column[26:34], title_column[38:46]]
    data_inicial = '2023-01-01'
    data_final = '2024-05-01'
    tamanho_sequencia = 20
    num_simulacoes = 10000
    num_dias = 60
    
    # Armazenar todas as simulações
    todas_simulacoes = []
    resultados_simulacoes = []
    for i in range(5):
        #print(f"Executando simulação para o ticker {tickers}")
        precos_simulados_rnn, prob_acima_preco_atual = ModeloFinanceiro.executar(tickers, data_inicial, data_final, tamanho_sequencia, num_simulacoes, num_dias)
        resultados_simulacoes.append(prob_acima_preco_atual)
    
        # Calculando a média dos preços futuros simulados
        media_precos_futuros = np.mean(resultados_simulacoes)
        todas_simulacoes.append(media_precos_futuros)
        
        # Visualizar resultados
        ultima_data = pd.to_datetime(data_final)
        datas_futuras = pd.date_range(start=ultima_data, periods=num_dias)
    
    print(f"Média das probabilidades: {np.mean(todas_simulacoes)}")
    
    """
            plt.figure(figsize=(14, 7))
            plt.plot(datas_futuras, media_precos_futuros, label=f'Média dos Preços Futuros {tickers}')
            plt.xlabel('Data')
            plt.ylabel('Preço de Fechamento')
            plt.title(f'Média das Simulações de Monte Carlo dos Preços Futuros: {tickers}')
            plt.grid(True)
            #plt.legend()
            plt.show()
    """
