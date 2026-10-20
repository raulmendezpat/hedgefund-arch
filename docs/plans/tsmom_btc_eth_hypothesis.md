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

## Resultado (2026-10-21): rechazada por poco; es la candidata más robusta hasta ahora

Corrida: `scripts/run_tsmom.py` → `artifacts/research/tsmom_btc_eth_v1/`.

| Criterio | Umbral | Resultado | ¿Cumple? |
|---|---|---|---|
| Sharpe fuera de muestra | ≥ 0.7 | 0.63 | no |
| Peor caída fuera de muestra | ≥ −30 % | −33.7 % | no |
| Sharpe de desarrollo | ≥ 0.5 | 1.10 | sí |
| Cada activo fuera de muestra > 0 | > 0 | BTC 0.46 · ETH 0.70 | sí |

| Combinación con momentum cruzado (fuera de muestra) | Momentum solo | Combinado | ¿Cumple? |
|---|---|---|---|
| Sharpe (≥ +0.15) | 1.00 | 1.11 (+0.11) | no |
| Peor caída | −32.4 % | −24.6 % | sí |
| Exposición bruta de la TSMOM a la escala (p95 ≤ 2) | — | 0.93 | sí |

En desarrollo, la combinación también mejora: Sharpe 0.55 contra 0.33 y peor caída −16 % contra
−24 %. La correlación diaria con el momentum fuera de muestra es 0.12.

**Por qué es la más robusta de todo lo probado** (aun sin pasar):
- **Años:** Sharpe positivo en los 7, de 2020 a 2026: 2.14 · 1.03 · 0.41 · 0.66 · 0.43 · 0.58 · 1.01.
- **Grilla:**
  - Las 24 combinaciones tienen Sharpe positivo fuera de muestra; el 62 % supera 0.5.
  - Medianas: 0.57 fuera de muestra y 1.19 en desarrollo.
- **Costos ×2:** Sharpe fuera de muestra 0.52. La rotación es baja y los costos de BTC y ETH,
  mínimos.
- **Bootstrap:** probabilidad de que el Sharpe fuera de muestra sea ≤ 0 = 13 %.
- **Sin depender de un activo:** BTC y ETH son positivos por separado.
- **No es beta disfrazada:** la correlación con comprar y mantener fuera de muestra es −0.06.
  Comprar y mantener tuvo Sharpe 0.39, con una caída de −38 % a la misma volatilidad.
- **Cuenta horaria:** peor caída −34 % fuera de muestra, sin tocar el kill switch de −35 % por
  muy poco.

**Debilidades:**
- **Retorno por lado:**
  - Desarrollo: largos +35 %/año, cortos +0.5 %/año.
  - Fuera de muestra: largos +17 %/año, cortos +6.5 %/año.
  - Los cortos solo aportaron en el período reciente.
- **Funding:** cuesta 2.4–6.5 %/año, porque los largos pagan funding.
- **Meses positivos:** 38 % fuera de muestra. Gana con pocas tendencias largas.

**Lecturas posteriores a ver los datos (no habilitan a cambiar el primario):** la variante solo
largo (0.85) y la de solo 60 días (0.92) tuvieron mejor Sharpe fuera de muestra. Elegirlas ahora
sería sobreajuste a un período mayormente alcista.

**Conclusión:** la hipótesis, tal como se registró, no pasa. Si se quiere seguir, lo honesto es
validarla con datos nuevos: un paper de la configuración primaria sin cambios, con su propia
compuerta pre-registrada. No corresponde ajustar los umbrales.
