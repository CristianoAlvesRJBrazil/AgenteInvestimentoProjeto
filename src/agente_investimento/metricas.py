"""Métricas de risco e retorno para séries de retornos diários.

Convenções:
- ``retornos`` são retornos simples diários (0.01 = +1%).
- ``retorno_acumulado`` é bruto: 1.20 significa +20% no período (mesma
  convenção do ``valor_desejado`` do projeto original).
- Anualização assume 252 dias úteis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DIAS_UTEIS_ANO = 252


def _taxa_diaria(taxa_anual: float) -> float:
    return (1.0 + taxa_anual) ** (1.0 / DIAS_UTEIS_ANO) - 1.0


def retorno_acumulado(retornos: pd.Series) -> float:
    """Retorno bruto do período: valor final / valor inicial."""
    return float((1.0 + retornos).prod())


def retorno_anualizado(retornos: pd.Series) -> float:
    if len(retornos) == 0:
        return float("nan")
    bruto = (1.0 + retornos).prod()
    return float(bruto ** (DIAS_UTEIS_ANO / len(retornos)) - 1.0)


def volatilidade_anualizada(retornos: pd.Series) -> float:
    return float(retornos.std(ddof=1) * np.sqrt(DIAS_UTEIS_ANO))


def sharpe(retornos: pd.Series, taxa_livre_risco_anual: float = 0.0) -> float:
    """Índice de Sharpe anualizado sobre o excesso de retorno diário."""
    excesso = retornos - _taxa_diaria(taxa_livre_risco_anual)
    desvio = excesso.std(ddof=1)
    if not np.isfinite(desvio) or desvio == 0:
        return float("nan")
    return float(excesso.mean() / desvio * np.sqrt(DIAS_UTEIS_ANO))


def sortino(retornos: pd.Series, taxa_livre_risco_anual: float = 0.0) -> float:
    """Como o Sharpe, mas penalizando apenas a volatilidade dos dias negativos."""
    excesso = retornos - _taxa_diaria(taxa_livre_risco_anual)
    negativos = excesso[excesso < 0]
    if len(negativos) == 0:
        return float("nan")
    desvio_negativo = np.sqrt((negativos**2).mean())
    if desvio_negativo == 0:
        return float("nan")
    return float(excesso.mean() / desvio_negativo * np.sqrt(DIAS_UTEIS_ANO))


def max_drawdown(retornos: pd.Series) -> float:
    """Maior queda do valor da carteira em relação ao pico anterior (negativo)."""
    valor = (1.0 + retornos).cumprod()
    pico = valor.cummax()
    return float((valor / pico - 1.0).min())


def var_historico(retornos: pd.Series, nivel: float = 0.95) -> float:
    """Value at Risk histórico diário no nível dado (positivo = perda)."""
    return float(-np.quantile(retornos, 1.0 - nivel))


def cvar_historico(retornos: pd.Series, nivel: float = 0.95) -> float:
    """Perda média nos dias piores que o VaR (Expected Shortfall)."""
    corte = np.quantile(retornos, 1.0 - nivel)
    cauda = retornos[retornos <= corte]
    if len(cauda) == 0:
        return float("nan")
    return float(-cauda.mean())


def resumo(retornos: pd.Series, taxa_livre_risco_anual: float = 0.0) -> dict[str, float]:
    """Painel com as principais métricas da série de retornos."""
    return {
        "retorno_acumulado": retorno_acumulado(retornos),
        "retorno_anualizado": retorno_anualizado(retornos),
        "volatilidade_anualizada": volatilidade_anualizada(retornos),
        "sharpe": sharpe(retornos, taxa_livre_risco_anual),
        "sortino": sortino(retornos, taxa_livre_risco_anual),
        "max_drawdown": max_drawdown(retornos),
        "var_95": var_historico(retornos),
        "cvar_95": cvar_historico(retornos),
    }
