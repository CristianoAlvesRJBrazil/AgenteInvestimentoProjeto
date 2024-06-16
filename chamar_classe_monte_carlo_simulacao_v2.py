# Importação das bibliotecas necessárias
import lista_ativos_setores
import carteira_aleatoria_otima_orientado_objeto
import MonteCarloSimulationCarteiraOB
import MonteCarloSimulationV2

class PortfolioSimulation:
    """
    Classe para simulação de portfólios e execução de simulações de Monte Carlo.
    """

    def __init__(self, inicio_dados, final_dados, valor_desejado):
        """
        Inicializa a simulação com as datas e o retorno desejado.

        :param inicio_dados: Data de início para os dados.
        :param final_dados: Data final para os dados.
        :param valor_desejado: Retorno acumulado desejado.
        """
        self.inicio_dados = inicio_dados
        self.final_dados = final_dados
        self.valor_desejado = valor_desejado
        self.lista_ativos = lista_ativos_setores.lista_ibov  # Lista de ativos a ser utilizada

    def otimizar_carteira(self):
        """
        Otimiza a carteira com base na lista de ativos e datas fornecidas.

        :return: Carteira otimizada.
        """
        portfolio = carteira_aleatoria_otima_orientado_objeto.PortfolioOptimization(
            self.lista_ativos, self.inicio_dados, self.final_dados
        )
        self.carteira_vencedora = portfolio.otimizar_carteira(self.valor_desejado)  # Realiza a otimização
        return self.carteira_vencedora

    def simular_monte_carlo(self, num_simulacoes=1000, num_dias=60):
        """
        Executa a simulação de Monte Carlo na carteira otimizada.

        :param num_simulacoes: Número de simulações a serem executadas.
        :param num_dias: Número de dias para a simulação.
        """
        if not hasattr(self, 'carteira_vencedora'):
            raise AttributeError("Carteira vencedora não otimizada. Execute otimizar_carteira primeiro.")
        MonteCarloSimulationCarteiraOB.main(
            self.carteira_vencedora, num_simulacoes, num_dias,
            self.inicio_dados, self.final_dados
        )

    def plotar_monte_carlo(self):
        """
        Plota o gráfico da simulação de Monte Carlo junto com o histórico de preços.
        """
        if not hasattr(self, 'carteira_vencedora'):
            raise AttributeError("Carteira vencedora não otimizada. Execute otimizar_carteira primeiro.")
        MonteCarloSimulationV2.main(
            self.carteira_vencedora, self.inicio_dados, self.final_dados
        )

    def run(self):
        """
        Executa a otimização da carteira e a simulação de Monte Carlo em sequência.
        """
        self.otimizar_carteira()  # Otimiza a carteira
        print(f"Carteira vencedora = {self.carteira_vencedora}")  # Exibe a carteira vencedora
        self.simular_monte_carlo()  # Executa a simulação de Monte Carlo
        self.plotar_monte_carlo()  # Plota o resultado da simulação

if __name__ == "__main__":
    # Intervalo de dados e retorno acumulado desejado
    inicio_dados = '2023-01-02'
    final_dados = '2023-05-02'
    valor_desejado = 1.24

    # Cria e executa a simulação de portfólio
    simulacao = PortfolioSimulation(inicio_dados, final_dados, valor_desejado)
    simulacao.run()

