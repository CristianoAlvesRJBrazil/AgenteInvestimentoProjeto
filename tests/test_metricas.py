import numpy as np
import pandas as pd
import pytest

from agente_investimento import metricas


def test_retorno_acumulado_conhecido():
    retornos = pd.Series([0.10, -0.10])
    assert metricas.retorno_acumulado(retornos) == pytest.approx(0.99)


def test_retorno_acumulado_positivo():
    retornos = pd.Series([0.01] * 10)
    assert metricas.retorno_acumulado(retornos) == pytest.approx(1.01**10)


def test_max_drawdown_queda_de_50_por_cento():
    # Sobe para 2.0 e cai para 1.0: drawdown de -50%.
    retornos = pd.Series([1.0, -0.5])
    assert metricas.max_drawdown(retornos) == pytest.approx(-0.5)


def test_max_drawdown_serie_so_de_altas_eh_zero():
    retornos = pd.Series([0.01, 0.02, 0.03])
    assert metricas.max_drawdown(retornos) == pytest.approx(0.0)


def test_sharpe_maior_para_menor_risco_mesmo_retorno():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0.001, 0.01, 500))
    arriscada = pd.Series(rng.normal(0.001, 0.05, 500))
    assert metricas.sharpe(base) > metricas.sharpe(arriscada)


def test_sharpe_retorno_constante_sem_variancia_eh_nan():
    retornos = pd.Series([0.01] * 50)
    assert np.isnan(metricas.sharpe(retornos))


def test_sortino_ignora_volatilidade_de_alta():
    # Série A: volatilidade só para cima; série B: mesma média com quedas.
    subidas = pd.Series([0.0, 0.04, 0.0, 0.04, 0.0, 0.04, -0.001, 0.0])
    quedas = pd.Series([0.02, -0.02, 0.03, -0.02, 0.02, -0.02, 0.03, 0.02])
    assert metricas.sortino(subidas) > metricas.sortino(quedas)


def test_var_e_cvar_historicos():
    # 10% dos dias com perda de 10%: o quantil de 5% cai dentro dessa cauda.
    retornos = pd.Series(np.concatenate([np.full(90, 0.01), np.full(10, -0.10)]))
    var = metricas.var_historico(retornos, nivel=0.95)
    cvar = metricas.cvar_historico(retornos, nivel=0.95)
    assert var == pytest.approx(0.10, abs=1e-6)
    assert cvar == pytest.approx(0.10, abs=1e-6)
    assert cvar >= var - 1e-12


def test_resumo_contem_todas_as_chaves():
    rng = np.random.default_rng(1)
    retornos = pd.Series(rng.normal(0.0005, 0.02, 252))
    painel = metricas.resumo(retornos)
    esperadas = {
        "retorno_acumulado", "retorno_anualizado", "volatilidade_anualizada",
        "sharpe", "sortino", "max_drawdown", "var_95", "cvar_95",
    }
    assert esperadas == set(painel)
    assert all(np.isfinite(v) for v in painel.values())
