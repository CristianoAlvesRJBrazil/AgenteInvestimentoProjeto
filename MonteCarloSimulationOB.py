import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class MonteCarloSimulation:
    def __init__(self, ticker, num_simulacoes, num_dias, inicio_dados, final_dados):
        self.ticker = ticker
        self.num_simulacoes = num_simulacoes
        self.num_dias = num_dias
        self.inicio_dados = inicio_dados
        self.final_dados = final_dados
        self.precos_simulados_df = None
        self.preco_atual = None
    
    def obter_dados_historicos(self):
        """Obtém os dados históricos do ativo."""
        try:
            dados_estoque = yf.download(self.ticker, start=self.inicio_dados, end=self.final_dados)['Close']
            return dados_estoque
        except Exception as e:
            print(f"Ocorreu um erro ao obter dados históricos: {e}")
            return None

    def simular_precos(self):
        """Executa a simulação de Monte Carlo para o ativo."""
        dados_estoque = self.obter_dados_historicos()
        if dados_estoque is None:
            return None

        retornos_diarios = dados_estoque.pct_change().dropna()
        retorno_medio = retornos_diarios.mean()
        desvio_padrao = retornos_diarios.std()

        numeros_aleatorios = np.random.normal(size=(self.num_simulacoes, self.num_dias))
        precos_simulados = np.zeros((self.num_simulacoes, self.num_dias))
        precos_simulados[:, 0] = dados_estoque.iloc[-1]

        for i in range(1, self.num_dias):
            precos_simulados[:, i] = precos_simulados[:, i - 1] * (1 + retorno_medio + desvio_padrao * numeros_aleatorios[:, i])

        self.precos_simulados_df = pd.DataFrame(precos_simulados, columns=[f"Dia {i+1}" for i in range(self.num_dias)])
        self.preco_atual = dados_estoque.iloc[-1]
        return self.precos_simulados_df

    def calcular_probabilidade_acima_preco_atual(self):
        """Calcula a probabilidade do preço futuro estar acima do preço atual."""
        if self.precos_simulados_df is None:
            print("Simulação não foi executada. Execute `simular_precos` primeiro.")
            return None

        num_acima_preco_atual = sum(self.precos_simulados_df.iloc[:, -1] > self.preco_atual)
        probabilidade = num_acima_preco_atual / self.num_simulacoes
        return probabilidade

    def plotar_precos_simulados(self):
        """Plota os preços simulados."""
        if self.precos_simulados_df is None:
            print("Simulação não foi executada. Execute `simular_precos` primeiro.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.precos_simulados_df.T, color='gray', alpha=0.1)
        plt.title(f'Simulações de Monte Carlo para {self.ticker}')
        plt.xlabel('Dias')
        plt.ylabel('Preço')
        plt.show()

    def plotar_distribuicao_precos_futuros(self):
        """Plota a distribuição dos preços futuros simulados."""
        if self.precos_simulados_df is None:
            print("Simulação não foi executada. Execute `simular_precos` primeiro.")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(self.precos_simulados_df.iloc[:, -1], bins=50, color='blue', edgecolor='black')
        plt.axvline(self.preco_atual, color='red', linestyle='dashed', linewidth=2, label='Preço Atual')
        plt.title('Distribuição dos Preços Futuros Simulados')
        plt.xlabel('Preço Futuro')
        plt.ylabel('Frequência')
        plt.legend()
        plt.show()

def main():
    Ativos = ['EMBR3.SA', 'BRKM5.SA', 'BBAS3.SA', 'STBP3.SA']
    ticker = Ativos[0]
    inicio_dados = '2023-01-01'
    final_dados = '2023-05-01'
    num_simulacoes = 1000
    num_dias = 60

    simulacao = MonteCarloSimulation(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)

    precos_simulados_df = simulacao.simular_precos()

    if precos_simulados_df is not None:
        probabilidade_acima_preco_atual = simulacao.calcular_probabilidade_acima_preco_atual()
        print(f"Probabilidade do preço futuro estar acima do preço atual após 60 dias: {probabilidade_acima_preco_atual:.2%}")
        
        simulacao.plotar_precos_simulados()
        simulacao.plotar_distribuicao_precos_futuros()

if __name__ == "__main__":
    main()
