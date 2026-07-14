"""Simulação de Monte Carlo multivariada de preços (GBM discreto com correlação).

Aperfeiçoamentos sobre a versão original do projeto:
- simula os ativos em conjunto preservando a matriz de correlação (Cholesky),
  em vez de colapsar a carteira numa única série média — a diversificação
  passa a existir também na simulação;
- usa log-retornos: preços simulados nunca ficam negativos e a volatilidade
  escala com o preço;
- probabilidade calculada sobre o valor final da carteira em todas as
  simulações (a versão funcional antiga media outra coisa: a fração de dias
  de UMA trajetória acima do preço inicial);
- vetorizado e com semente reprodutível.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .carteira import Carteira
from .dados import calcular_retornos


def _cholesky_robusto(cov: np.ndarray) -> np.ndarray:
    """Cholesky com regularização progressiva para matrizes quase singulares."""
    jitter = 0.0
    escala = np.mean(np.diag(cov)) or 1.0
    for _ in range(8):
        try:
            return np.linalg.cholesky(cov + jitter * np.eye(len(cov)))
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10.0, escala * 1e-12)
    raise np.linalg.LinAlgError("Matriz de covariância não é positiva definida.")


class SimulacaoMonteCarlo:
    """Simula trajetórias futuras de preços de uma carteira a partir do histórico.

    Parameters
    ----------
    precos_historicos:
        DataFrame de preços (uma coluna por ticker) do período de estimação.
    carteira:
        Carteira com tickers e pesos; se None, pesos iguais sobre todas as colunas.
    seed:
        Semente do gerador para reprodutibilidade.
    """

    def __init__(
        self,
        precos_historicos: pd.DataFrame,
        carteira: Carteira | None = None,
        seed: int | None = None,
    ):
        if precos_historicos.empty:
            raise ValueError("Preços históricos vazios.")
        self.carteira = carteira or Carteira(list(precos_historicos.columns))
        self.precos = precos_historicos[self.carteira.tickers].dropna()
        if len(self.precos) < 30:
            raise ValueError(
                f"Histórico insuficiente ({len(self.precos)} dias com dados de todos "
                "os ativos); mínimo de 30 para estimar média e covariância."
            )
        log_retornos = calcular_retornos(self.precos, log_retornos=True)
        self.media = log_retornos.mean().to_numpy()
        self.cov = log_retornos.cov().to_numpy()
        self.ultimo_preco = self.precos.iloc[-1].to_numpy()
        self.rng = np.random.default_rng(seed)
        self.trajetorias_precos: np.ndarray | None = None  # (sims, dias+1, ativos)

    def simular(self, num_dias: int, num_simulacoes: int) -> np.ndarray:
        """Gera trajetórias de preços; shape (num_simulacoes, num_dias + 1, num_ativos).

        O índice 0 do eixo de tempo é o último preço real observado; cada passo
        seguinte aplica um log-retorno sorteado de N(media, cov) com choques
        correlacionados entre os ativos.
        """
        num_ativos = len(self.carteira.tickers)
        cholesky = _cholesky_robusto(self.cov)
        normais = self.rng.standard_normal((num_simulacoes, num_dias, num_ativos))
        choques = normais @ cholesky.T
        log_ret = self.media + choques

        trajetorias = np.empty((num_simulacoes, num_dias + 1, num_ativos))
        trajetorias[:, 0, :] = self.ultimo_preco
        trajetorias[:, 1:, :] = self.ultimo_preco * np.exp(np.cumsum(log_ret, axis=1))
        self.trajetorias_precos = trajetorias
        return trajetorias

    def valores_carteira(self) -> np.ndarray:
        """Valor da carteira (buy-and-hold, base 1.0) em cada trajetória; shape (sims, dias+1)."""
        if self.trajetorias_precos is None:
            raise RuntimeError("Execute simular() antes.")
        quantidades = self.carteira.pesos / self.ultimo_preco
        return self.trajetorias_precos @ quantidades

    def probabilidade_acima(self, alvo: float = 1.0) -> float:
        """Fração das simulações cujo valor FINAL da carteira supera ``alvo``.

        ``alvo`` é relativo ao valor atual: 1.0 = preço de hoje, 1.2 = +20%.
        """
        valores = self.valores_carteira()
        return float(np.mean(valores[:, -1] > alvo))

    def estatisticas_finais(self) -> dict[str, float]:
        """Distribuição do retorno bruto da carteira no fim do horizonte."""
        finais = self.valores_carteira()[:, -1]
        percentis = np.percentile(finais, [5, 25, 50, 75, 95])
        return {
            "media": float(finais.mean()),
            "desvio_padrao": float(finais.std(ddof=1)),
            "p05": float(percentis[0]),
            "p25": float(percentis[1]),
            "mediana": float(percentis[2]),
            "p75": float(percentis[3]),
            "p95": float(percentis[4]),
            "prob_acima_atual": self.probabilidade_acima(1.0),
        }
