"""Seleção e otimização de carteiras.

Dois níveis, combináveis:
1. ``buscar_carteira_aleatoria`` — busca aleatória de combinações de N ativos
   com pesos iguais (o método do projeto original), mas pontuada por métricas
   de risco-retorno (Sharpe por padrão) em vez do critério ad hoc antigo, que
   é mantido como ``criterio="excesso_ibov"`` para comparação.
2. ``otimizar_pesos`` — dado um conjunto de ativos, encontra os pesos long-only
   que maximizam o critério (SLSQP/scipy).

IMPORTANTE: a pontuação aqui é sempre in-sample. A validação honesta de
qualquer carteira selecionada acontece no módulo ``backtest`` (walk-forward),
nunca na própria janela de seleção.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import metricas
from .carteira import Carteira, retornos_carteira


@dataclass
class ResultadoSelecao:
    carteira: Carteira
    criterio: str
    valor_criterio: float
    metricas: dict[str, float]


def _funcao_criterio(nome: str, retornos_benchmark: pd.Series | None):
    """Retorna função retornos_da_carteira -> pontuação (maior = melhor)."""
    if nome == "sharpe":
        return metricas.sharpe
    if nome == "sortino":
        return metricas.sortino
    if nome == "retorno_acumulado":
        return metricas.retorno_acumulado
    if nome == "excesso_ibov":
        if retornos_benchmark is None:
            raise ValueError("criterio='excesso_ibov' exige retornos do benchmark.")
        media_benchmark = retornos_benchmark.mean()

        def criterio_original(retornos: pd.Series) -> float:
            # Critério do projeto original, mantido para comparação histórica.
            return float((retornos.mean() - media_benchmark) * metricas.retorno_acumulado(retornos))

        return criterio_original
    raise ValueError(f"Critério desconhecido: {nome!r}")


def buscar_carteira_aleatoria(
    retornos_ativos: pd.DataFrame,
    num_carteiras: int = 1000,
    num_ativos: int = 4,
    criterio: str = "sharpe",
    retornos_benchmark: pd.Series | None = None,
    seed: int | None = None,
) -> ResultadoSelecao:
    """Sorteia combinações de ativos (pesos iguais) e devolve a de melhor pontuação."""
    universo = list(retornos_ativos.columns)
    if len(universo) < num_ativos:
        raise ValueError(f"Universo com {len(universo)} ativos; preciso de {num_ativos}.")

    pontuar = _funcao_criterio(criterio, retornos_benchmark)
    rng = np.random.default_rng(seed)

    melhor_combo: tuple[str, ...] | None = None
    melhor_valor = -np.inf
    avaliadas: set[tuple[str, ...]] = set()

    for _ in range(num_carteiras):
        combo = tuple(sorted(rng.choice(universo, size=num_ativos, replace=False)))
        if combo in avaliadas:
            continue
        avaliadas.add(combo)
        serie = retornos_ativos[list(combo)].dropna().mean(axis=1)
        if serie.empty:
            continue
        valor = pontuar(serie)
        if np.isfinite(valor) and valor > melhor_valor:
            melhor_valor = valor
            melhor_combo = combo

    if melhor_combo is None:
        raise RuntimeError("Nenhuma carteira válida encontrada na busca aleatória.")

    carteira = Carteira(list(melhor_combo))
    serie = retornos_carteira(retornos_ativos, carteira)
    return ResultadoSelecao(
        carteira=carteira,
        criterio=criterio,
        valor_criterio=melhor_valor,
        metricas=metricas.resumo(serie),
    )


def otimizar_pesos(
    retornos_ativos: pd.DataFrame,
    tickers: list[str],
    criterio: str = "sharpe",
    retornos_benchmark: pd.Series | None = None,
) -> ResultadoSelecao:
    """Otimiza os pesos (long-only, soma 1) de um conjunto fixo de ativos."""
    pontuar = _funcao_criterio(criterio, retornos_benchmark)
    matriz = retornos_ativos[tickers].dropna()
    if matriz.empty:
        raise ValueError("Sem retornos comuns aos ativos informados.")

    num_ativos = len(tickers)
    pesos_iniciais = np.full(num_ativos, 1.0 / num_ativos)

    def objetivo(pesos: np.ndarray) -> float:
        valor = pontuar(matriz @ pesos)
        return -valor if np.isfinite(valor) else 1e9

    resultado = minimize(
        objetivo,
        pesos_iniciais,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * num_ativos,
        constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1.0}],
    )
    pesos = resultado.x if resultado.success else pesos_iniciais

    carteira = Carteira(list(tickers), pesos)
    serie = retornos_carteira(retornos_ativos, carteira)
    return ResultadoSelecao(
        carteira=carteira,
        criterio=criterio,
        valor_criterio=pontuar(serie),
        metricas=metricas.resumo(serie),
    )
