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
            dados_estoque = yf.download(self.ticker, start=self.inicio_dados, end=self.final_dados)['Close'].mean(axis=1)
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

def main(ticker, num_simulacoes, num_dias, inicio_dados, final_dados):

    simulacao = MonteCarloSimulation(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)

    precos_simulados_df = simulacao.simular_precos()

    if precos_simulados_df is not None:
        probabilidade_acima_preco_atual = simulacao.calcular_probabilidade_acima_preco_atual()
        print(f"Resultados da Predição:")
        print(f"Probabilidade do preço futuro estar acima do preço atual nos próximos 60 dias: {probabilidade_acima_preco_atual:.2%}")
        
        simulacao.plotar_precos_simulados()
        simulacao.plotar_distribuicao_precos_futuros()

if __name__ == "__main__":
    # Importação das bibliotecas necessárias
    import lista_ativos_setores
    from carteira_aleatoria_otima_orientado_objeto import PortfolioOptimization
    
    # Intervalo de dados e retorno acumulado desejando
    inicio_dados = '2024-01-01'
    final_dados = '2024-05-01'
    valor_desejado = 1.24
    
    # Acessar a lista de ativos e escolher a carteira otimizada    
    lista_ativos = lista_ativos_setores.lista_atual_ibxx_02
    portfolio = PortfolioOptimization(lista_ativos, inicio_dados, final_dados)
    carteira_vencedora = portfolio.otimizar_carteira(valor_desejado)
        
    # Chamar a classe MonteCarloSimulationCarteiraOB
    ticker = carteira_vencedora
    num_simulacoes = 1000
    num_dias = 60
    main(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)


