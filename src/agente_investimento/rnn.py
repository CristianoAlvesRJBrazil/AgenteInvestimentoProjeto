"""Modelo híbrido LSTM + Monte Carlo (opcional; requer tensorflow).

Correções sobre as versões originais (RNN_Monte_Carlo_POO_Carteira*.py):

- SEM vazamento de dados: a normalização é ajustada apenas no treino e depois
  aplicada ao teste (antes o MinMaxScaler via a série inteira).
- O modelo trabalha com LOG-RETORNOS, não com níveis de preço; a simulação de
  Monte Carlo dos resíduos compõe retornos multiplicativamente, então os
  preços simulados nunca ficam negativos (antes somava resíduos em R$ ao preço).
- Sementes fixadas para reprodutibilidade.
- Comparação obrigatória com o baseline ingênuo (passeio aleatório: prever
  retorno zero). Se a LSTM não bate o baseline, o híbrido não deve ser usado
  no lugar do Monte Carlo puro.

Instalação: ``pip install -e ".[rnn]"``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    TENSORFLOW_DISPONIVEL = True
except ImportError:  # pragma: no cover - depende do ambiente
    TENSORFLOW_DISPONIVEL = False

from .dados import calcular_retornos


class PrevisorLSTM:
    """LSTM univariada sobre log-retornos de uma série de preços.

    O uso previsto é sobre a série de VALOR da carteira (ver
    ``carteira.valor_buy_and_hold``), não sobre a média de preços dos ativos.
    """

    def __init__(
        self,
        precos: pd.Series,
        tamanho_sequencia: int = 20,
        fracao_treino: float = 0.8,
        seed: int = 42,
    ):
        if not TENSORFLOW_DISPONIVEL:
            raise ImportError(
                "tensorflow não instalado. Instale com: pip install -e \".[rnn]\""
            )
        if precos.isna().any():
            precos = precos.dropna()
        self.precos = precos.astype(float)
        self.log_retornos = calcular_retornos(self.precos, log_retornos=True)
        self.tamanho_sequencia = tamanho_sequencia
        self.fracao_treino = fracao_treino
        self.seed = seed
        self.modelo = None
        self._escala_min: float | None = None
        self._escala_max: float | None = None
        self.residuos_teste: np.ndarray | None = None

        minimo_necessario = int(tamanho_sequencia / (1 - fracao_treino)) + tamanho_sequencia
        if len(self.log_retornos) < minimo_necessario:
            raise ValueError(
                f"Série curta demais ({len(self.log_retornos)} retornos) para "
                f"sequências de {tamanho_sequencia} com fração de treino {fracao_treino}."
            )

    # -- normalização ajustada SÓ no treino (correção do vazamento) ----------
    def _ajustar_escala(self, valores_treino: np.ndarray) -> None:
        self._escala_min = float(valores_treino.min())
        self._escala_max = float(valores_treino.max())

    def _transformar(self, valores: np.ndarray) -> np.ndarray:
        amplitude = (self._escala_max - self._escala_min) or 1.0
        return (valores - self._escala_min) / amplitude

    def _reverter(self, valores: np.ndarray) -> np.ndarray:
        amplitude = (self._escala_max - self._escala_min) or 1.0
        return valores * amplitude + self._escala_min

    def _criar_sequencias(self, serie: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        janelas, alvos = [], []
        for i in range(len(serie) - self.tamanho_sequencia):
            janelas.append(serie[i : i + self.tamanho_sequencia])
            alvos.append(serie[i + self.tamanho_sequencia])
        return (
            np.asarray(janelas).reshape(-1, self.tamanho_sequencia, 1),
            np.asarray(alvos),
        )

    def treinar(self, epochs: int = 50, batch_size: int = 32, verbose: int = 0) -> dict[str, float]:
        """Treina a LSTM e devolve métricas de teste comparadas ao baseline ingênuo."""
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        valores = self.log_retornos.to_numpy()
        corte = int(len(valores) * self.fracao_treino)
        treino_bruto, teste_bruto = valores[:corte], valores[corte:]

        self._ajustar_escala(treino_bruto)
        treino = self._transformar(treino_bruto)
        # O teste usa as últimas sequências do treino como contexto inicial.
        teste = self._transformar(np.concatenate([treino_bruto[-self.tamanho_sequencia :], teste_bruto]))

        X_treino, y_treino = self._criar_sequencias(treino)
        X_teste, y_teste = self._criar_sequencias(teste)

        self.modelo = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.tamanho_sequencia, 1)),
            Dropout(0.2),
            LSTM(32),
            Dense(1),
        ])
        self.modelo.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error")
        self.modelo.fit(X_treino, y_treino, epochs=epochs, batch_size=batch_size, verbose=verbose)

        previsto = self._reverter(self.modelo.predict(X_teste, verbose=0).ravel())
        real = self._reverter(y_teste)
        self.residuos_teste = real - previsto

        mse_lstm = float(np.mean((real - previsto) ** 2))
        # Baseline passeio aleatório: melhor previsão do log-retorno de amanhã é ~0.
        mse_ingenuo = float(np.mean(real**2))
        return {
            "mse_lstm": mse_lstm,
            "mse_baseline_ingenuo": mse_ingenuo,
            "lstm_bate_baseline": mse_lstm < mse_ingenuo,
        }

    def simular_monte_carlo(self, num_dias: int, num_simulacoes: int) -> np.ndarray:
        """Simula preços futuros com resíduos da LSTM como fonte de incerteza.

        Cada passo compõe: preço * exp(resíduo sorteado de N(media, desvio) dos
        resíduos de teste). Retorna shape (num_simulacoes, num_dias + 1).
        """
        if self.residuos_teste is None:
            raise RuntimeError("Execute treinar() antes.")
        media = float(np.mean(self.residuos_teste))
        desvio = float(np.std(self.residuos_teste, ddof=1))

        rng = np.random.default_rng(self.seed)
        log_ret = rng.normal(media, desvio, size=(num_simulacoes, num_dias))
        ultimo = float(self.precos.iloc[-1])

        trajetorias = np.empty((num_simulacoes, num_dias + 1))
        trajetorias[:, 0] = ultimo
        trajetorias[:, 1:] = ultimo * np.exp(np.cumsum(log_ret, axis=1))
        return trajetorias

    def probabilidade_acima(self, trajetorias: np.ndarray, alvo_relativo: float = 1.0) -> float:
        """Fração das simulações com preço final acima de ``alvo_relativo`` × preço atual."""
        ultimo = float(self.precos.iloc[-1])
        return float(np.mean(trajetorias[:, -1] > alvo_relativo * ultimo))
