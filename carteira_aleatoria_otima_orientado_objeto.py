import numpy as np
import pandas as pd
import yfinance as yf

class PortfolioOptimization:
    def __init__(self, lista_ativos, inicio_dados, final_dados):
        self.lista_ativos = lista_ativos
        self.inicio_dados = inicio_dados
        self.final_dados = final_dados
        self.dados_ativos = self.obter_dados_ativos()
        self.retornos_ativos = self.calcular_retornos_percentuais(self.dados_ativos)

    def obter_dados_ativos(self):
        """Obtém os dados históricos de fechamento dos ativos."""
        dados = yf.download(self.lista_ativos, start=self.inicio_dados, end=self.final_dados)['Close']
        return dados

    def calcular_retornos_percentuais(self, dados):
        """Calcula os retornos percentuais diários dos ativos."""
        retornos = dados.pct_change().dropna()
        return retornos

    def simular_carteiras(self, num_carteiras=100, num_ativos=4):
        """Simula carteiras aleatórias com os ativos disponíveis."""
        carteiras = []
        for _ in range(num_carteiras):
            ativos_carteira = np.random.choice(self.lista_ativos, size=num_ativos, replace=False)
            retorno_carteira = self.retornos_ativos[ativos_carteira].mean(axis=1)
            carteiras.append((ativos_carteira, retorno_carteira))
        return carteiras

    def calcular_retorno_medio(self, retorno_carteira):
        """Calcula o retorno médio do período."""
        retorno_medio = retorno_carteira.mean()
        return retorno_medio

    def calcular_desvio_padrao(self, retorno_carteira):
        """Calcula o desvio padrão do período."""
        desvio_padrao = retorno_carteira.std()
        return desvio_padrao

    def calcular_retorno_acumulado(self, retorno_carteira):
        """Calcula o retorno acumulado do período."""
        retorno_acumulado = (1 + retorno_carteira).cumprod().iloc[-1]
        return retorno_acumulado

    def encontrar_carteira_vencedora(self, carteiras, retorno_ibov):
        """Encontra a carteira com o maior retorno acumulado acima do Ibovespa."""
        retorno_acumulado_carteiras = [self.calcular_retorno_acumulado(carteira[1]) for carteira in carteiras]
        def calculo(carteira):
            retorno_carteira = carteira[1]
            return (self.calcular_retorno_medio(retorno_carteira) - retorno_ibov.mean()) * self.calcular_retorno_acumulado(carteira[1])
        carteira_vencedora = max(carteiras, key=calculo)
        return carteira_vencedora

    def obter_dados_ibov(self, inicio_dados, final_dados):
        lista_indicador = ['^BVSP']
        ibov = yf.download(lista_indicador, start=inicio_dados, end=final_dados)['Close']
        retorno_ibov = self.calcular_retornos_percentuais(ibov)
        return retorno_ibov

    def otimizar_carteira(self, valor_desejado, max_iteracoes=10000):
        iteracao_atual = 0
        carteira_vencedora = None
        retorno_ibov = self.obter_dados_ibov(self.inicio_dados, self.final_dados)

        while iteracao_atual < max_iteracoes:
            carteiras_simuladas = self.simular_carteiras()
            carteira_vencedora = self.encontrar_carteira_vencedora(carteiras_simuladas, retorno_ibov)
            retorno_acumulado_carteira = self.calcular_retorno_acumulado(carteira_vencedora[1])

            if retorno_acumulado_carteira >= valor_desejado:
                break

            iteracao_atual += 1

        if iteracao_atual == max_iteracoes:
            print(f"Não foi possível encontrar uma carteira que atendesse ao valor desejado de retorno acumulado após {max_iteracoes} iterações.")
        else:
            retorno_medio_carteira = self.calcular_retorno_medio(carteira_vencedora[1])
            desvio_padrao = self.calcular_desvio_padrao(carteira_vencedora[1])
            dias_uteis_ano = 252
            media_anual = (1 + retorno_medio_carteira) ** dias_uteis_ano - 1
            desvio_padrao_anual = desvio_padrao * np.sqrt(dias_uteis_ano)

            print("Carteira Vencedora:")
            print("Ativos =", carteira_vencedora[0])
            print("Retorno Médio Diário:", retorno_medio_carteira)
            print("Desvio Padrão Diário:", desvio_padrao)
            print(f"Média Anual dos Retornos: {media_anual:.4f}")
            print(f"Desvio Padrão Anual dos Retornos: {desvio_padrao_anual:.4f}")
            print("Retorno Acumulado:", retorno_acumulado_carteira)

if __name__ == "__main__":
    lista_ativos = ['ABEV3.SA', 'AZUL4.SA', 'B3SA3.SA', 'BBAS3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BBSE3.SA', 'BRAP4.SA',
                    'BRFS3.SA', 'BRKM5.SA', 'CASH3.SA', 'CCRO3.SA', 'CIEL3.SA', 'CMIG4.SA', 'COGN3.SA', 'CPFE3.SA',
                    'CPLE6.SA', 'CRFB3.SA', 'CSAN3.SA', 'CSNA3.SA', 'CVCB3.SA', 'CYRE3.SA', 'DIRR3.SA', 'ECOR3.SA',
                    'ELET3.SA', 'ELET6.SA', 'EMBR3.SA', 'ENGI11.SA', 'EQTL3.SA', 'EZTC3.SA', 'GGBR4.SA', 'GOAU4.SA',
                    'GOLL4.SA', 'GRND3.SA', 'GUAR3.SA', 'HYPE3.SA', 'IRBR3.SA', 'ITSA4.SA', 'ITUB4.SA', 'JBSS3.SA',
                    'KLBN11.SA', 'LREN3.SA', 'MGLU3.SA', 'MRFG3.SA', 'MRVE3.SA', 'MULT3.SA', 'NEOE3.SA', 'NTCO3.SA',
                    'PCAR3.SA', 'PETR3.SA', 'PETR4.SA', 'POMO4.SA', 'POSI3.SA', 'PRIO3.SA', 'QUAL3.SA', 'RADL3.SA',
                    'RAIL3.SA', 'RAPT4.SA', 'RENT3.SA', 'SANB11.SA', 'SBSP3.SA', 'SCAR3.SA', 'SUZB3.SA', 'TAEE11.SA',
                    'TCSA3.SA', 'TECN3.SA', 'TEKA3.SA', 'TIMS3.SA', 'TOTS3.SA', 'UGPA3.SA', 'USIM5.SA', 'VALE3.SA',
                    'VIVT3.SA', 'WEGE3.SA', 'YDUQ3.SA']

    inicio_dados = '2024-01-29'
    final_dados = '2024-05-29'
    print(f"Para teste inicial, verifique se existe carteira que retone 12% de retorno para os 4 meses analisandos, digite: 1.12")
    valor_desejado = float(input("Informe o valor desejado de retorno acumulado: "))

    otimizador = PortfolioOptimization(lista_ativos, inicio_dados, final_dados)
    otimizador.otimizar_carteira(valor_desejado)

