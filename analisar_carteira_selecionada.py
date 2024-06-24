import pandas as pd
import csv
from lista_ativos_setores import lista_atual_ibxx_02 as laibxx
from carteira_aleatoria_otima_orientado_objeto import PortfolioOptimization
from MonteCarloSimulationCarteiraOB import MonteCarloSimulation

class PortfolioAnalyzer:
    def __init__(self, lista_ativos, inicio_dados, final_dados, valor_desejado, num_simulacoes=1000, num_dias=60):
        self.lista_ativos = lista_ativos
        self.inicio_dados = inicio_dados
        self.final_dados = final_dados
        self.valor_desejado = valor_desejado
        self.num_simulacoes = num_simulacoes
        self.num_dias = num_dias
        self.carteira_vencedora = None
        self.resultados = []

    def selecionar_carteira_vencedora(self):
        otimizador = PortfolioOptimization(self.lista_ativos, self.inicio_dados, self.final_dados)
        self.carteira_vencedora = otimizador.otimizar_carteira(self.valor_desejado)
        print(f"Carteira_Vencedora = {self.carteira_vencedora}.")

    def realizar_simulacao_monte_carlo(self):
        if self.carteira_vencedora is None:
            raise ValueError("Carteira vencedora não definida. Execute 'selecionar_carteira_vencedora' primeiro.")

        simulacao = MonteCarloSimulation(self.carteira_vencedora, self.num_simulacoes, self.num_dias, self.inicio_dados, self.final_dados)
        
        for _ in range(100):
            precos_simulados_df = simulacao.simular_precos()
            probabilidade_acima_preco_atual = simulacao.calcular_probabilidade_acima_preco_atual()
            self.resultados.append(probabilidade_acima_preco_atual)

    def salvar_resultados_csv(self, filename='resultados.csv'):
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([self.carteira_vencedora])  # Escreve o cabeçalho
            for resultado in self.resultados:
                writer.writerow([resultado])
        print(f"Resultados salvos em {filename}.")

    def analisar_resultados(self, filename='resultados.csv'):
        df = pd.read_csv(filename)
        
        # Resumo estatístico
        resumo_estatistico = df.describe()
        print(f"Resumo estatístico:\n {resumo_estatistico}")
        
        # Calcular a média
        media = df.mean().iloc[0]
        print(f"A média é: {round(media, 4)}")

        # Calcular a mediana
        mediana = df.median().iloc[0]
        print(f"A mediana é: {round(mediana, 4)}")

        # Calcular a moda
        moda = df.mode().iloc[0, 0]
        print(f"A moda é: {round(moda, 4)}")

        print(f"Probabilidade do preço futuro estar acima do esperado: {round(mediana, 4)*100}%")
        print(f"Carteira_Vencedora = {self.carteira_vencedora}.")

    def executar_analise_completa(self):
        self.selecionar_carteira_vencedora()
        self.realizar_simulacao_monte_carlo()
        self.salvar_resultados_csv()
        self.analisar_resultados()

if __name__ == "__main__":
    # Parâmetros de entrada
    lista_ativos = laibxx
    inicio_dados = '2023-01-01'
    final_dados = '2023-05-01'
    valor_desejado = 1.20

    # Criar instância do analisador de portfólio
    analisador = PortfolioAnalyzer(lista_ativos, inicio_dados, final_dados, valor_desejado)

    # Executar análise completa
    analisador.executar_analise_completa()
    

