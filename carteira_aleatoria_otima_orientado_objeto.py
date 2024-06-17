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

    def simular_carteiras(self, num_carteiras=1000, num_ativos=4):
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
        """Obtém os dados históricos de fechamento do Ibovespa."""
        lista_indicador = ['^BVSP']
        ibov = yf.download(lista_indicador, start=inicio_dados, end=final_dados)['Close']
        retorno_ibov = self.calcular_retornos_percentuais(ibov)
        return retorno_ibov

    def otimizar_carteira(self, valor_desejado, max_iteracoes=10000):
        """Obtém a carteira otimizada com rendimento acima do Ibovespa."""
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
            retorno_medio_carteira_percentual = retorno_medio_carteira * 100
            desvio_padrao = self.calcular_desvio_padrao(carteira_vencedora[1])
            desvio_padrao_percentual = desvio_padrao * 100
            dias_uteis_ano = 252
            media_anual = ((1 + retorno_medio_carteira) ** dias_uteis_ano - 1) * 100
            desvio_padrao_anual = (desvio_padrao * np.sqrt(dias_uteis_ano)) * 100

            print("Resultados da Carteira Vencedora:")
            print(f"Carteira Vencedora = {carteira_vencedora[0]}")
            print(f"Retorno Médio Diário: {retorno_medio_carteira_percentual:.2f}%")
            print(f"Desvio Padrão Diário: {desvio_padrao_percentual:.2f}%")
            print(f"Retorno Médio Anual da Carteira: {media_anual:.2f}%")
            print(f"Desvio Padrão do Retorno Médio Anual da Carteira: {desvio_padrao_anual:.2f}%")
            print(f"Retorno Acumulado: {retorno_acumulado_carteira:.4f}")
            
            # Criar uma lista de ativo com for 
            ativo = []
            for vencedora in carteira_vencedora[0]:
                ativo.append(vencedora)
            return ativo, retorno_acumulado_carteira

if __name__ == "__main__":
    # Lista de ativos da B3
    import lista_ativos_setores as las
    lista_ativos = las.lista_atual_ibxx_02
    
    # Intervalo de Datas para análise 
    inicio_dados = '2023-01-02'
    final_dados = '2023-05-02'
    print(f"Para teste inicial, verifique se existe carteira que retone 12% de retorno para os 4 meses analisandos, digite: 1.12")
    valor_desejado = float(input("Informe o valor desejado de retorno acumulado: "))

    # Chamando o otmizador de carteira, selecionando o retorno desejado, selecionando a carteira vencedora
    otimizador = PortfolioOptimization(lista_ativos, inicio_dados, final_dados)
    ativo = otimizador.otimizar_carteira(valor_desejado)
    print(f"Carteira Vencedora = {ativo}")

