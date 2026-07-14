import numpy as np
import pandas as pd
import pytest

from agente_investimento import otimizacao


def gerar_universo_sintetico(seed=11):
    """8 ativos: dois claramente superiores (média alta, vol baixa)."""
    rng = np.random.default_rng(seed)
    datas = pd.bdate_range("2023-01-02", periods=250)
    series = {}
    for i in range(6):
        series[f"RUIM{i}3.SA"] = rng.normal(0.0, 0.03, len(datas))
    series["BOM13.SA"] = rng.normal(0.003, 0.008, len(datas))
    series["BOM23.SA"] = rng.normal(0.003, 0.008, len(datas))
    return pd.DataFrame(series, index=datas)


def test_busca_aleatoria_encontra_ativos_superiores():
    retornos = gerar_universo_sintetico()
    resultado = otimizacao.buscar_carteira_aleatoria(
        retornos, num_carteiras=300, num_ativos=2, criterio="sharpe", seed=5
    )
    assert set(resultado.carteira.tickers) == {"BOM13.SA", "BOM23.SA"}
    assert resultado.valor_criterio > 0
    assert "sharpe" in resultado.metricas


def test_busca_aleatoria_e_reprodutivel_com_seed():
    retornos = gerar_universo_sintetico()
    r1 = otimizacao.buscar_carteira_aleatoria(retornos, 50, 3, seed=9)
    r2 = otimizacao.buscar_carteira_aleatoria(retornos, 50, 3, seed=9)
    assert r1.carteira.tickers == r2.carteira.tickers


def test_universo_menor_que_carteira_gera_erro():
    retornos = gerar_universo_sintetico()[["BOM13.SA", "BOM23.SA"]]
    with pytest.raises(ValueError):
        otimizacao.buscar_carteira_aleatoria(retornos, 10, num_ativos=4)


def test_criterio_excesso_ibov_exige_benchmark():
    retornos = gerar_universo_sintetico()
    with pytest.raises(ValueError, match="benchmark"):
        otimizacao.buscar_carteira_aleatoria(retornos, 10, 2, criterio="excesso_ibov")


def test_criterio_excesso_ibov_funciona_com_benchmark():
    retornos = gerar_universo_sintetico()
    benchmark = pd.Series(
        np.random.default_rng(3).normal(0.0002, 0.01, len(retornos)), index=retornos.index
    )
    resultado = otimizacao.buscar_carteira_aleatoria(
        retornos, 100, 2, criterio="excesso_ibov", retornos_benchmark=benchmark, seed=1
    )
    assert len(resultado.carteira.tickers) == 2


def test_criterio_desconhecido_gera_erro():
    retornos = gerar_universo_sintetico()
    with pytest.raises(ValueError, match="desconhecido"):
        otimizacao.buscar_carteira_aleatoria(retornos, 10, 2, criterio="mágico")


def test_otimizar_pesos_respeita_restricoes_e_melhora_criterio():
    retornos = gerar_universo_sintetico()
    tickers = ["BOM13.SA", "RUIM03.SA", "RUIM13.SA"]

    resultado = otimizacao.otimizar_pesos(retornos, tickers, criterio="sharpe")
    pesos = resultado.carteira.pesos

    assert pesos.sum() == pytest.approx(1.0)
    assert (pesos >= -1e-9).all()

    # Deve concentrar no ativo bom e superar a carteira de pesos iguais.
    from agente_investimento import metricas
    sharpe_igual = metricas.sharpe(retornos[tickers].mean(axis=1))
    assert resultado.valor_criterio >= sharpe_igual
    assert pesos[0] == max(pesos)
