# Hipótesis registrada: volatility targeting del momentum y apalancamiento x1–x4

Escrita el 2026-10-21, **antes de construir el backtest y de mirar ningún resultado** de esta
estrategia. La configuración primaria, los criterios de éxito y las reglas para aceptar un nivel
de apalancamiento no se cambian después de ver los resultados. La grilla sirve para medir
robustez, no para elegir parámetros.

## Idea

El momentum cruzado tiene caídas fuertes y concentradas: las peores llegan cuando el mercado
está más volátil. El volatility targeting achica la exposición cuando la volatilidad reciente
de la estrategia sube y la agranda cuando baja, para mantener un riesgo aproximadamente
constante (Barroso & Santa-Clara, 2015, "Momentum has its moments"). Si funciona, mejora el
Sharpe y reduce las caídas sin cambiar la señal.

## Configuración primaria

**Estrategia base:** el momentum primario (`PRIMARY_STRATEGY`), sin cambios:
- universo top 60;
- rebalanceo semanal;
- costos escalonados de 3/6/12 bp más un fee de 0.06 %;
- 0.5 largo / 0.5 corto (exposición bruta 1.0).

**Escala:** se fija en cada rebalanceo, al decidir el día *d*. Los pesos objetivo de la base
se multiplican por

  k_d = min(cap, objetivo / σ̂_d)

donde:
- **σ̂_d** es la volatilidad anualizada (desviación diaria × √365) de los retornos netos diarios
  de la estrategia base sin escalar, en los **60 días** previos. Usa solo filas ≤ *d*−1, cuyo
  último fill es anterior a la decisión. Si hay menos de 30 días, k = 1.
- **objetivo** = **25 %** anual. Es aproximadamente la volatilidad de la base en desarrollo (27.7 %),
  así que la exposición promedio queda cerca de 1.
- **cap** = **2.0**: la exposición bruta máxima es 2.

**Períodos:** desarrollo 2021-03-01 a 2023-12-31; fuera de muestra desde 2024-01-01.

## Criterios de éxito del volatility targeting (x1, todos requeridos)

| # | Criterio |
|---|---|
| 1 | Sharpe fuera de muestra ≥ el de la base + 0.10 |
| 2 | Sharpe de desarrollo ≥ el de la base (no empeora) |
| 3 | La peor caída fuera de muestra, **a igual volatilidad realizada que la base**, es mejor que la de la base |
| 4 | El peor mes fuera de muestra, a igual volatilidad, es mejor que el de la base |

Para el criterio 3, la serie con targeting se reescala de forma estática a la volatilidad
realizada de la base en el mismo período. Así no se premia simplemente tener menos riesgo.

También se informa, sin que sea criterio, un bootstrap por bloques de 20 días: la probabilidad
de que la mejora del Sharpe fuera de muestra sea ≤ 0.

## Apalancamiento x1, x2, x3, x4

Se aplica a la base y al targeting (objetivo × L, cap × L). Los pesos se multiplican por L y
pasan por el mismo motor: el fee, el slippage y el funding escalan con la exposición.

**Simulación horaria de la cuenta:**
- Cada día se toman los pesos tenidos desde el último fill y los cierres horarios de cada
  símbolo.
- Con eso se reconstruye la equity hora a hora: el mínimo intradiario, no solo el cierre diario.

**Liquidación:**
- Margen cruzado. Hay liquidación si en cualquier hora la equity ≤ margen de mantenimiento ×
  notional bruto vigente.
- Margen de mantenimiento: **2 %** del notional. Es conservador para alts en Bitget, cuyo
  primer tramo está entre 0.5 % y 1.5 %.
- Una liquidación termina la simulación de ese período.

**Kill switch del plan de producción:** se registra la primera hora en que la caída desde el
máximo llega a −25 %.

**Métricas por período:**
- CAGR, Sharpe, peor caída diaria y horaria, y peor día.
- Fecha de liquidación, si la hay, y fecha del primer toque del kill switch.
- Distribución de retornos en ventanas móviles de 30 días: mediana, percentil 5 y 95,
  porcentaje de ventanas ≥ +10 % y ≤ −10 %.

**Un nivel de apalancamiento se considera aceptable para producción** solo si cumple las tres
condiciones, **en desarrollo y fuera de muestra**:
1. sin liquidación;
2. peor caída horaria > −40 %;
3. peor ventana de 30 días > −30 %.

## Sensibilidad (solo para medir robustez)

- Ventana de σ̂: 20, 60 y 120 días.
- Cap: 1.5, 2 y 3.
- Objetivo fijo en 25 %.

Se informa en qué fracción de las 9 combinaciones el Sharpe fuera de muestra supera al de la
base.

## Limitaciones conocidas

- La volatilidad se estima sobre la estrategia base sin escalar (una sombra). Sus retornos
  difieren levemente de los escalados por la banda y los costos.
- Las velas horarias son de Binance; la ejecución sería en Bitget.
- No se modelan los tramos de margen por tamaño de posición ni el ADL del exchange.
- El slippage no crece con el tamaño. Con x3 y x4, una cuenta grande pagaría más impacto de lo
  modelado.
