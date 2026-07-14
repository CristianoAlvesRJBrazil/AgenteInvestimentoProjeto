"""Representação de carteira com pesos explícitos.

Corrige duas inconsistências do projeto original:
- lá a "carteira" ora era a média dos *retornos* (seleção), ora a média dos
  *preços* (simulação/backtest) — duas carteiras diferentes;
- a média de preços equivale a comprar 1 ação de cada papel, fazendo um papel
  de R$80 pesar 10x mais que um de R$8 no resultado.

Aqui os pesos são explícitos (iguais por padrão) e há duas formas de retorno:
com rebalanceamento diário (pesos constantes) e buy-and-hold (compra e carrega).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Carteira:
    """Conjunto de ativos com pesos que somam 1 (iguais se não informados)."""

    tickers: list[str]
    pesos: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if not self.tickers:
            raise ValueError("Carteira precisa de ao menos um ticker.")
        if self.pesos is None:
            self.pesos = np.full(len(self.tickers), 1.0 / len(self.tickers))
        else:
            self.pesos = np.asarray(self.pesos, dtype=float)
            if len(self.pesos) != len(self.tickers):
                raise ValueError("Número de pesos difere do número de tickers.")
            if (self.pesos < 0).any():
                raise ValueError("Pesos negativos não são suportados (carteira long-only).")
            soma = self.pesos.sum()
            if soma <= 0:
                raise ValueError("A soma dos pesos deve ser positiva.")
            self.pesos = self.pesos / soma


def retornos_carteira(retornos_ativos: pd.DataFrame, carteira: Carteira) -> pd.Series:
    """Retorno diário da carteira com rebalanceamento diário (pesos constantes)."""
    faltando = [t for t in carteira.tickers if t not in retornos_ativos.columns]
    if faltando:
        raise KeyError(f"Retornos ausentes para: {faltando}")
    matriz = retornos_ativos[carteira.tickers].dropna()
    return matriz @ carteira.pesos


def valor_buy_and_hold(precos: pd.DataFrame, carteira: Carteira, valor_inicial: float = 1.0) -> pd.Series:
    """Valor da carteira comprando no primeiro dia do período e mantendo as posições.

    O capital inicial é dividido conforme os pesos e convertido em quantidades
    fixas de cada papel; o valor diário é a soma das posições.
    """
    faltando = [t for t in carteira.tickers if t not in precos.columns]
    if faltando:
        raise KeyError(f"Preços ausentes para: {faltando}")
    matriz = precos[carteira.tickers].dropna()
    if matriz.empty:
        raise ValueError("Sem datas com preço para todos os ativos da carteira.")
    quantidades = valor_inicial * carteira.pesos / matriz.iloc[0].to_numpy()
    return matriz @ quantidades


def retorno_periodo_buy_and_hold(precos: pd.DataFrame, carteira: Carteira) -> float:
    """Retorno bruto do período (valor final / valor inicial) sem rebalanceamento."""
    valor = valor_buy_and_hold(precos, carteira)
    return float(valor.iloc[-1] / valor.iloc[0])
