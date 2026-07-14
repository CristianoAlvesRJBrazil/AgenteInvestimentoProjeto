import numpy as np
import pandas as pd
import pytest

from agente_investimento.carteira import Carteira
from agente_investimento.monte_carlo import SimulacaoMonteCarlo, _cholesky_robusto


def gerar_precos_sinteticos(num_dias=300, correlacao=0.6, seed=7):
    """Dois ativos GBM com correlação conhecida, para testar a simulação."""
    rng = np.random.default_rng(seed)
    cov = np.array([[0.0004, correlacao * 0.02 * 0.03], [correlacao * 0.02 * 0.03, 0.0009]])
    cholesky = np.linalg.cholesky(cov)
    choques = rng.standard_normal((num_dias, 2)) @ cholesky.T
    log_ret = 0.0005 + choques
    precos = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    datas = pd.bdate_range("2023-01-02", periods=num_dias)
    return pd.DataFrame(precos, index=datas, columns=["AAAA3.SA", "BBBB3.SA"])


def test_shapes_e_dia_zero():
    precos = gerar_precos_sinteticos()
    sim = SimulacaoMonteCarlo(precos, seed=1)
    trajetorias = sim.simular(num_dias=60, num_simulacoes=500)
    assert trajetorias.shape == (500, 61, 2)
    assert np.allclose(trajetorias[:, 0, :], precos.iloc[-1].to_numpy())


def test_precos_simulados_sempre_positivos():
    precos = gerar_precos_sinteticos()
    sim = SimulacaoMonteCarlo(precos, seed=2)
    trajetorias = sim.simular(num_dias=120, num_simulacoes=2000)
    assert (trajetorias > 0).all()


def test_semente_reproduz_resultado():
    precos = gerar_precos_sinteticos()
    t1 = SimulacaoMonteCarlo(precos, seed=42).simular(30, 100)
    t2 = SimulacaoMonteCarlo(precos, seed=42).simular(30, 100)
    assert np.array_equal(t1, t2)


def test_correlacao_preservada_na_simulacao():
    correlacao_alvo = 0.6
    precos = gerar_precos_sinteticos(correlacao=correlacao_alvo)
    sim = SimulacaoMonteCarlo(precos, seed=3)
    trajetorias = sim.simular(num_dias=250, num_simulacoes=200)

    log_ret = np.diff(np.log(trajetorias), axis=1)
    correlacoes = [np.corrcoef(log_ret[i, :, 0], log_ret[i, :, 1])[0, 1] for i in range(200)]
    correlacao_media = float(np.mean(correlacoes))

    correlacao_historica = float(
        np.log(precos / precos.shift(1)).dropna().corr().iloc[0, 1]
    )
    assert correlacao_media == pytest.approx(correlacao_historica, abs=0.1)


def test_probabilidade_entre_zero_e_um_e_alvos_extremos():
    precos = gerar_precos_sinteticos()
    sim = SimulacaoMonteCarlo(precos, seed=4)
    sim.simular(num_dias=60, num_simulacoes=1000)
    prob = sim.probabilidade_acima(1.0)
    assert 0.0 <= prob <= 1.0
    assert sim.probabilidade_acima(0.0) == 1.0      # todo preço é positivo
    assert sim.probabilidade_acima(1e9) == 0.0      # alvo inalcançável


def test_valores_carteira_comeca_em_um():
    precos = gerar_precos_sinteticos()
    cart = Carteira(list(precos.columns), pesos=[0.3, 0.7])
    sim = SimulacaoMonteCarlo(precos, cart, seed=5)
    sim.simular(num_dias=10, num_simulacoes=50)
    valores = sim.valores_carteira()
    assert valores.shape == (50, 11)
    assert np.allclose(valores[:, 0], 1.0)


def test_estatisticas_finais_coerentes():
    precos = gerar_precos_sinteticos()
    sim = SimulacaoMonteCarlo(precos, seed=6)
    sim.simular(num_dias=60, num_simulacoes=2000)
    est = sim.estatisticas_finais()
    assert est["p05"] <= est["p25"] <= est["mediana"] <= est["p75"] <= est["p95"]
    assert 0.0 <= est["prob_acima_atual"] <= 1.0


def test_historico_insuficiente_gera_erro():
    precos = gerar_precos_sinteticos(num_dias=10)
    with pytest.raises(ValueError, match="insuficiente"):
        SimulacaoMonteCarlo(precos)


def test_cholesky_robusto_com_matriz_singular():
    # Dois ativos perfeitamente correlacionados: covariância singular.
    cov = np.array([[0.0004, 0.0004], [0.0004, 0.0004]])
    L = _cholesky_robusto(cov)
    assert np.allclose(L @ L.T, cov, atol=1e-8)
