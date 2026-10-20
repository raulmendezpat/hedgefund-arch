# Hipótesis registrada: momentum de series de tiempo (TSMOM) en BTC y ETH

Escrita el 2026-10-21, **antes de construir el backtest y de mirar ningún resultado** de esta
estrategia. La configuración primaria y los criterios de éxito no se cambian después de ver los
resultados. La grilla sirve para medir robustez, no para elegir parámetros.

## Idea

Cada activo se compra cuando su propia tendencia reciente es alcista y se vende en corto cuando es
bajista (Moskowitz, Ooi & Pedersen, 2012). A diferencia del momentum cruzado, no compara activos
entre sí: apuesta a que las tendencias de BTC y ETH persisten. Interesa por dos razones:
- **Liquidez:** son los dos activos más líquidos, con costos mínimos y sin riesgo de delisting.
- **Diversificación:** debería tener poca correlación con el momentum cruzado, que es neutral al
  mercado, y por eso podría servir como segunda pata del portafolio.

## Configuración primaria

**Activos:** perpetuos BTC/USDT y ETH/USDT de Binance USDT-M. Las velas de 1 hora y el funding
están completados desde su listado (BTC 2019-09, ETH 2019-11).

**Señal** (al cierre de las 23:00 UTC del día *d*, solo con cierres ≤ *d*):
- s_i = promedio de sign(r_20), sign(r_60) y sign(r_120), donde r_L es el retorno del activo en
  L días.
- s ∈ {−1, −⅓, +⅓, +1}.

**Tamaño:** w_i = s_i × 0.20 / σ_i, con tope de |w_i| ≤ 1.0.
- σ_i es la volatilidad anualizada de los retornos diarios del activo en los últimos 60 días.
- Cada activo apunta a ~20 % de volatilidad, y el portafolio a 25–30 %, parecido al momentum
  cruzado.

**Ejecución:**
- Decisión diaria y fill a la 01:00 UTC; es el mismo motor `hf_research.engine` del momentum.
- Banda de 0.05: no se redimensiona una posición del mismo lado si el cambio es menor.
- Modo largo/corto.

**Costos:**
- Fee de 0.06 % más 3 bp de slippage por lado. Es el tramo 1, porque BTC y ETH son los más
  líquidos.
- Funding real de cada activo.

**Períodos:**
- Desarrollo: 2020-04-01 a 2023-12-31. Empieza ahí porque ETH necesita 120 días de historia.
- Fuera de muestra: desde 2024-01-01.

## Criterios de éxito (todos requeridos)

| # | Criterio |
|---|---|
| 1 | Sharpe fuera de muestra ≥ 0.7 |
| 2 | Peor caída fuera de muestra ≥ −30 % |
| 3 | Sharpe de desarrollo ≥ 0.5 |
| 4 | Cada activo por separado (misma regla, solo BTC o solo ETH) con Sharpe fuera de muestra > 0 |

**Combinación con el momentum cruzado primario:**
- Riesgo igual: la TSMOM se escala a la volatilidad de desarrollo del momentum.
- Es útil si, fuera de muestra, el Sharpe combinado ≥ el del momentum solo + 0.15 **y** la peor
  caída es mejor.
- Se informa la exposición bruta implícita. Si para igualar el riesgo hace falta una exposición
  bruta mayor que 2, se declara no implementable.

**Se informa también, sin ser criterio:**
- Sharpe y peor caída de comprar y mantener 50/50 BTC/ETH con la misma volatilidad, y
  correlación con esa posición.
- Bootstrap por bloques de 20 días: probabilidad de que el Sharpe fuera de muestra sea ≤ 0.
- Simulación horaria de la cuenta (`hf_research.vol_target`): peor caída horaria y si toca el
  kill switch de −35 %.
- Sharpe por año y separado por lado: aporte de los largos y de los cortos.

## Sensibilidad (solo para medir robustez)

- Ventanas: solo 20, solo 60, solo 120, y la combinación 20/60/120.
- Banda: 0.02, 0.05 y 0.10.
- Modo: largo/corto y solo largo.
- Costos al doble en la configuración primaria.

Se informa qué fracción de las 24 combinaciones tiene Sharpe fuera de muestra ≥ 0.5.

## Limitaciones conocidas

- **Pocas apuestas independientes:** son solo dos activos, muy correlacionados (~0.8). El número
  de apuestas es bajo y los intervalos de confianza son anchos. Un resultado de 2–3 años puede
  depender de una o dos tendencias.
- **Régimen:** fuera de muestra (2024–2026) incluye un mercado mayormente alcista. Por eso se
  separa el aporte de largos y cortos y se compara con comprar y mantener.
- **Fuente de datos:** los datos son de Binance y la ejecución sería en Bitget. En BTC y ETH la
  diferencia de precio y de funding es pequeña.
