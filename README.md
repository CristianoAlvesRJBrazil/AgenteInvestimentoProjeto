# Agente de Investimento — B3

Sistema quantitativo de análise de carteiras da B3: seleção de carteira sobre um
universo de ativos, simulação de Monte Carlo multivariada da distribuição de
retornos futuros, métricas de risco e **backtesting walk-forward** honesto
(seleção in-sample, avaliação out-of-sample).

Versão 2.0 — reescrita do projeto original (preservado em [`legacy/`](legacy/))
com correção de bugs que invalidavam os resultados e modernização da metodologia.

## Instalação

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

# opcional, para o modelo LSTM:
.venv/bin/pip install -e ".[rnn]"
```

## Uso rápido

```bash
# Pipeline completo: universo -> seleção -> Monte Carlo -> relatório
.venv/bin/python scripts/rodar_pipeline.py --inicio 2024-01-02 --fim 2024-05-01

# Backtest walk-forward (a validação que importa)
.venv/bin/python scripts/rodar_backtest.py --inicio 2022-01-01 --fim 2024-07-01

# Reavaliação dos 30 experimentos históricos com a lógica corrigida
.venv/bin/python scripts/reavaliar_resultados.py

# Testes
.venv/bin/python -m pytest tests/
```

## Estrutura

```
src/agente_investimento/
├── dados.py         # download yfinance + cache parquet; fechamento ajustado
├── universo.py      # índices B3 (scraping com fallback estático) + validação
├── carteira.py      # carteira com pesos explícitos; rebalanceada e buy-and-hold
├── metricas.py      # Sharpe, Sortino, max drawdown, VaR/CVaR, anualizações
├── monte_carlo.py   # GBM multivariado com correlação (Cholesky), log-retornos
├── otimizacao.py    # busca aleatória pontuada por métricas + otimização de pesos (SLSQP)
├── backtest.py      # walk-forward, veredito de direção, teste binomial, leitor legado
└── rnn.py           # híbrido LSTM + Monte Carlo (opcional, sem vazamento de dados)

scripts/             # CLIs: rodar_pipeline, rodar_backtest, reavaliar_resultados
tests/               # 56 testes unitários (pytest), sem dependência de rede
legacy/              # código original preservado para referência
Resultados/          # CSVs dos experimentos originais + saídas novas
```

## Correções em relação ao projeto original

Bugs que afetavam as conclusões:

1. **Veredito do backtest** — `if retorno_simulado and retorno_real > 1.00`
   declarava "acerto" sempre que o retorno real era positivo, ignorando a
   previsão (precedência de operadores). Agora `avaliar_previsao` compara a
   direção prevista (probabilidade Monte Carlo vs. limiar) com a direção
   realizada, e a taxa de acerto agregada passa por teste binomial contra o acaso.
2. **Probabilidade da versão funcional** — `probabilidade_futuro_sem_grafico.py`
   media a fração de *dias* de UMA trajetória acima do preço inicial (e dividia
   pelo nº de colunas), não a fração de *simulações* com preço final acima do
   atual. As duas implementações do projeto divergiam silenciosamente.
3. **Simulação O(n²) com re-download** — o módulo de estatística chamava a
   simulação `num_simulacoes` vezes, cada uma rodando `num_simulacoes`
   trajetórias e baixando os dados de novo (com 1000: 1 milhão de trajetórias e
   1000 downloads). Agora: uma estimação, N trajetórias vetorizadas, cache local.
4. **Tickers por fatiamento fixo** — `title_column[2:10]` quebrava com tickers
   de 6 caracteres (BPAC11, IGTI11). Agora `ast.literal_eval` no leitor legado.
5. **Compatibilidade pandas 2.x / yfinance atual** — `series[-1]` (removido no
   pandas 2), colunas MultiIndex do yfinance novo, `read_html` com `StringIO`.

Aperfeiçoamentos de metodologia:

- **Carteira com pesos explícitos**: antes a "carteira" era a média dos *preços*
  (equivale a 1 ação de cada papel — um papel de R$80 pesava 10x um de R$8) na
  simulação, mas a média dos *retornos* na seleção. Agora pesos monetários
  consistentes nos dois lugares, com versões rebalanceada e buy-and-hold.
- **Monte Carlo multivariado**: os ativos são simulados em conjunto com a
  matriz de covariância dos log-retornos (Cholesky) — a correlação entre os
  papéis, essência da diversificação, agora existe também na simulação. Preços
  em log-retornos nunca ficam negativos.
- **Fechamento ajustado** (splits e dividendos) em vez do fechamento bruto.
- **Critérios de seleção por risco-retorno** (Sharpe, Sortino) além do critério
  original (`excesso_ibov`, mantido para comparação).
- **Walk-forward**: a carteira é escolhida só com dados passados e julgada só
  com dados futuros, janela após janela — elimina o data snooping de validar
  na própria janela de seleção.
- **LSTM sem vazamento**: normalização ajustada apenas no treino, modelo sobre
  log-retornos, comparação obrigatória com baseline ingênuo (passeio aleatório),
  sementes fixas.

## Metodologia (resumo)

1. **Universo**: composição do IBXX/IBOV (online com fallback estático),
   filtrada por disponibilidade de dados no período.
2. **Seleção**: busca aleatória de combinações de N ativos pontuadas in-sample
   pelo critério escolhido; opcionalmente otimização dos pesos (SLSQP, long-only).
3. **Simulação**: GBM discreto multivariado estimado nos log-retornos da janela
   de seleção; distribuição do valor da carteira no horizonte; probabilidade de
   alta = fração das simulações com valor final acima do atual.
4. **Validação**: walk-forward com janelas móveis; acerto = direção prevista ==
   direção realizada; significância via teste binomial; comparação com IBOV.

## Resultados da reavaliação dos experimentos históricos

Reavaliação dos 30 experimentos originais (seleção jan–mai/2023, avaliação
mai–ago/2023) com o veredito corrigido — tabela completa em
`Resultados/reavaliacao_backtesting.csv`:

- **20 carteiras avaliáveis**: 20/20 (100%) acertaram a direção prevista
  (todas previram alta com prob. mediana 0,74–0,93 e todas subiram no período).
- **12/20 (60%) bateram o Ibovespa**, que subiu 19,6% na janela de avaliação.
- **10 carteiras ficaram inavaliáveis** por tickers sem dados hoje no Yahoo
  Finance (fechamentos de capital e renomeações desde 2023 — ex.: CIEL3,
  AZUL4, TRPL4, NTCO3, CCRO3) — motivo pelo qual o pipeline novo valida o
  universo antes de qualquer análise.
- A carteira com pesos monetários iguais rendeu consistentemente MAIS que a
  cesta original de média de preços (ex.: 1,41 vs. 1,26) — o método antigo
  subponderava os papéis baratos, que mais subiram.

**Ressalva estatística importante**: os 30 experimentos compartilham a mesma
janela de avaliação, num trimestre em que o Ibovespa subiu ~20% — as
observações não são independentes e qualquer carteira comprada tendia a subir.
O acerto de 100% confirma o funcionamento do processo, não a habilidade
preditiva. A medida honesta é o walk-forward abaixo.

## Walk-forward 2022–2024 (out-of-sample)

Oito janelas independentes (seleção 4 meses, avaliação 3 meses, critério
Sharpe, universo IBOV estático) — detalhe em `Resultados/walk_forward.csv`:

| Métrica | Resultado |
|---|---|
| Janelas avaliadas | 8 |
| Acerto de direção | 37,5% (p-valor binomial 0,86) |
| Carteiras que bateram o IBOV | 37,5% |
| Retorno médio das carteiras | 0,975 |
| Retorno médio do IBOV | 1,028 |

Ou seja: **fora da amostra, a estratégia atual (comprar os vencedores dos
últimos 4 meses) não mostra poder preditivo** — o resultado espetacular da
janela única de 2023 era efeito do mercado em alta, não de habilidade. Repare
também que as probabilidades Monte Carlo saem superconfiantes (0,95+ em
janelas que caíram): a carteira selecionada pelo melhor Sharpe in-sample tem
o drift inflado por construção, e a simulação herda esse viés.

Isso não é um defeito do sistema novo — é o sistema novo **medindo a verdade**.
Com essa régua honesta instalada, dá para iterar a estratégia com método:
encolhimento (shrinkage) do drift na simulação, janelas de estimação mais
longas, critérios alternativos, filtros de regime de mercado etc., sempre
julgados pelo walk-forward.

## Limitações e próximos passos

- Sem custos de transação, impostos ou slippage nos retornos (próximo passo
  natural no walk-forward).
- GBM assume retornos normais i.i.d. — caudas reais são mais pesadas
  (alternativas: bootstrap de blocos, t-Student, GARCH).
- A busca aleatória com pesos iguais é o baseline histórico; fronteira
  eficiente completa e restrições setoriais são evoluções diretas.
- LSTM univariada sobre a carteira; multivariada e features exógenas em aberto.
