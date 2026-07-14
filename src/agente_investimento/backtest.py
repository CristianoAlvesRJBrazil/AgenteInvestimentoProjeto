"""Backtesting walk-forward e avaliação honesta de previsões.

Correções centrais em relação ao projeto original:

1. O veredito antigo era ``if retorno_simulado and retorno_real > 1.00`` —
   por precedência de operadores isso ignora a previsão e declara acerto
   sempre que o retorno real foi positivo. Aqui o acerto compara a DIREÇÃO
   prevista (probabilidade de alta do Monte Carlo) com a direção realizada
   (``avaliar_previsao``).

2. A seleção da carteira acontece só com dados da janela de seleção; a
   avaliação usa exclusivamente a janela seguinte (out-of-sample). O
   walk-forward repete isso ao longo do tempo para medir a taxa de acerto
   com várias amostras em vez de uma única janela.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import dados, otimizacao
from .carteira import Carteira, retorno_periodo_buy_and_hold
from .monte_carlo import SimulacaoMonteCarlo


def avaliar_previsao(probabilidade_alta: float, retorno_acumulado_real: float, limiar: float = 0.5) -> bool:
    """True se a direção prevista bateu com a realizada.

    A previsão é de alta quando a probabilidade Monte Carlo supera o limiar;
    a realidade é de alta quando o retorno bruto do período supera 1.0.
    """
    previu_alta = probabilidade_alta > limiar
    subiu = retorno_acumulado_real > 1.0
    return previu_alta == subiu


def teste_binomial_acertos(num_acertos: int, num_total: int) -> float:
    """p-valor do teste binomial contra acerto ao acaso (p = 0.5), unilateral.

    p-valor baixo (< 0.05) indica taxa de acerto acima do que a sorte explica.
    """
    if num_total == 0:
        return float("nan")
    return float(stats.binomtest(num_acertos, num_total, p=0.5, alternative="greater").pvalue)


def ler_resultado_legado(caminho: str | Path) -> tuple[list[str], pd.Series]:
    """Lê um CSV do formato antigo (Resultados/resultadosNN.csv).

    O cabeçalho é a lista de tickers serializada como string — parseada com
    ``ast.literal_eval``, o que corrige o bug do fatiamento por posição fixa
    (``title_column[2:10]``), que quebrava com tickers de 6 caracteres como
    BPAC11.SA. As linhas são as probabilidades registradas.
    """
    df = pd.read_csv(caminho)
    tickers = ast.literal_eval(df.columns[0])
    if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
        raise ValueError(f"Cabeçalho de {caminho} não é uma lista de tickers.")
    probabilidades = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    return tickers, probabilidades


@dataclass
class JanelaBacktest:
    """Par de janelas contíguas: seleção (in-sample) e avaliação (out-of-sample)."""

    selecao_inicio: pd.Timestamp
    selecao_fim: pd.Timestamp
    avaliacao_fim: pd.Timestamp


def gerar_janelas(
    inicio: str,
    fim: str,
    meses_selecao: int = 4,
    meses_avaliacao: int = 3,
    passo_meses: int = 3,
) -> list[JanelaBacktest]:
    """Gera janelas walk-forward entre ``inicio`` e ``fim``."""
    janelas = []
    cursor = pd.Timestamp(inicio)
    limite = pd.Timestamp(fim)
    while True:
        selecao_fim = cursor + pd.DateOffset(months=meses_selecao)
        avaliacao_fim = selecao_fim + pd.DateOffset(months=meses_avaliacao)
        if avaliacao_fim > limite:
            break
        janelas.append(JanelaBacktest(cursor, selecao_fim, avaliacao_fim))
        cursor = cursor + pd.DateOffset(months=passo_meses)
    return janelas


def rodar_walk_forward(
    universo: list[str],
    inicio: str,
    fim: str,
    meses_selecao: int = 4,
    meses_avaliacao: int = 3,
    passo_meses: int = 3,
    num_carteiras: int = 1000,
    num_ativos: int = 4,
    num_simulacoes: int = 5000,
    criterio: str = "sharpe",
    seed: int | None = 42,
) -> pd.DataFrame:
    """Roda o backtest walk-forward completo e devolve um DataFrame por janela.

    Em cada janela: seleciona a carteira in-sample, estima a probabilidade de
    alta com Monte Carlo (horizonte = janela de avaliação) e confronta com o
    retorno realmente observado fora da amostra, além do Ibovespa no mesmo
    período.
    """
    janelas = gerar_janelas(inicio, fim, meses_selecao, meses_avaliacao, passo_meses)
    if not janelas:
        raise ValueError("Período curto demais para formar ao menos uma janela.")

    linhas = []
    for i, janela in enumerate(janelas):
        sel_ini = janela.selecao_inicio.strftime("%Y-%m-%d")
        sel_fim = janela.selecao_fim.strftime("%Y-%m-%d")
        aval_fim = janela.avaliacao_fim.strftime("%Y-%m-%d")

        precos_selecao = dados.baixar_precos(universo, sel_ini, sel_fim)
        if precos_selecao.shape[1] < num_ativos:
            continue
        retornos_selecao = dados.calcular_retornos(precos_selecao)
        benchmark_selecao = dados.calcular_retornos(dados.baixar_benchmark(sel_ini, sel_fim))

        resultado = otimizacao.buscar_carteira_aleatoria(
            retornos_selecao,
            num_carteiras=num_carteiras,
            num_ativos=num_ativos,
            criterio=criterio,
            retornos_benchmark=benchmark_selecao,
            seed=None if seed is None else seed + i,
        )
        cart = resultado.carteira

        num_dias_uteis = int(np.busday_count(sel_fim, aval_fim))
        simulacao = SimulacaoMonteCarlo(
            precos_selecao, cart, seed=None if seed is None else seed + i
        )
        simulacao.simular(num_dias=max(num_dias_uteis, 1), num_simulacoes=num_simulacoes)
        prob_alta = simulacao.probabilidade_acima(1.0)

        precos_avaliacao = dados.baixar_precos(cart.tickers, sel_fim, aval_fim)
        if sorted(precos_avaliacao.columns) != sorted(cart.tickers):
            continue  # algum ativo sumiu na janela futura; sem como avaliar
        retorno_real = retorno_periodo_buy_and_hold(precos_avaliacao, cart)

        ibov = dados.baixar_benchmark(sel_fim, aval_fim)
        retorno_ibov = float(ibov.iloc[-1] / ibov.iloc[0])

        linhas.append({
            "selecao_inicio": sel_ini,
            "selecao_fim": sel_fim,
            "avaliacao_fim": aval_fim,
            "carteira": ", ".join(cart.tickers),
            "prob_alta_mc": prob_alta,
            "retorno_real": retorno_real,
            "retorno_ibov": retorno_ibov,
            "acertou_direcao": avaliar_previsao(prob_alta, retorno_real),
            "bateu_ibov": retorno_real > retorno_ibov,
        })

    return pd.DataFrame(linhas)


def resumir_walk_forward(resultado: pd.DataFrame) -> dict[str, float]:
    """Estatísticas agregadas do walk-forward, com teste de significância."""
    if resultado.empty:
        return {}
    acertos = int(resultado["acertou_direcao"].sum())
    total = len(resultado)
    return {
        "janelas": total,
        "taxa_acerto_direcao": acertos / total,
        "p_valor_binomial": teste_binomial_acertos(acertos, total),
        "taxa_bateu_ibov": float(resultado["bateu_ibov"].mean()),
        "retorno_medio_carteiras": float(resultado["retorno_real"].mean()),
        "retorno_medio_ibov": float(resultado["retorno_ibov"].mean()),
    }
