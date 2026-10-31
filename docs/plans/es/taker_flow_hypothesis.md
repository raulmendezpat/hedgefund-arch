# Hipótesis registrada: flujo agresivo (taker flow) cruzado, validada solo con datos nuevos

Escrita el 2026-10-21. A diferencia de las hipótesis anteriores, esta señal **ya se vio en todo el
historial disponible**: es la secundaria S2 de `open_interest_hypothesis.md`, que salió con Sharpe
1.28 en desarrollo y 0.93 fuera de muestra, elegida entre 4 pruebas. El backtest no puede
validarla. La validación es **exclusivamente el paper**, con datos posteriores a esta fecha y la
configuración congelada aquí.

## Configuración (congelada; es la misma del backtest S2)

**Universo:** el top 60 mensual de Binance USDT-M (mismo cálculo que el momentum), restringido a
símbolos listados en Bitget.

**Dato:**
- `buySellRatio` = volumen comprador agresivo / volumen vendedor agresivo, en barras de 5 minutos.
- Fuente: la API de Binance (`/futures/data/takerlongshortRatio`).
- Coincide con el archivo `sum_taker_long_short_vol_ratio` del backtest a la misma marca de tiempo
  (diferencia máxima 0.0016, por redondeo).

**Score** al cierre del domingo (día *d*):
- Para cada día UTC, el promedio de ln(ratio) de sus barras de 5 minutos.
- Luego, el promedio de esos valores diarios en los 7 días *d*−6..*d*, con al menos 5 días con
  dato.

**Cartera:**
- Se compra el 20 % con mayor score y se vende en corto el 20 % con menor score.
- Pesos iguales, 0.5 largo / 0.5 corto.
- Mínimo 8 activos con score; un símbolo sin score no es elegible.
- Banda de 0.02.

**Calendario y ejecución:** idénticos al paper del momentum.
- Decisión el domingo y fill el lunes a la 01:00 UTC.
- Simulación contra el libro de Bitget, fee de 0.06 % y funding real de Bitget.
- Capital simulado: 10,000 USDT.

## Compuerta de implementación (2027-01-11, 12 rebalanceos)

Mismos umbrales que el paper del momentum:
- ≥ 12 rebalanceos y ≤ 1 perdido;
- slippage mediano por tramo ≤ 3 / 6 / 12 bp;
- correlación diaria paper vs modelo ≥ 0.9;
- brecha de implementación ≥ −5 %/año.

El modelo de comparación usa los pesos que el paper decidió cada semana, con los precios de fill de
Binance y los costos del backtest. Así mide solo la ejecución.

## Decisión sobre la señal (solo con datos desde el primer rebalanceo, 2026-10-26)

| Fecha | Regla |
|---|---|
| 2027-04-19 (26 semanas) | Se **descarta** si el Sharpe anualizado del paper es < −0.5. Si no, sigue corriendo. |
| 2027-10-25 (52 semanas) | Pasa a producción, como segunda pata junto al momentum con mitad del riesgo cada una, solo si el paper tiene **Sharpe ≥ 0.7** y **peor caída ≥ −30 %**, y la compuerta de implementación está aprobada. Si no, se descarta. |

**Se informa además, sin que decida nada:** la combinación 50/50 con el paper del momentum en los
mismos días, y la correlación entre ambos.

**Advertencia de potencia:** un año de datos semanales es poco. Con un Sharpe real de 0.7, la
probabilidad de observar ≥ 0.7 en 52 semanas es ~50 %. La regla es exigente a propósito: un falso
positivo cuesta capital real, y un falso negativo solo cuesta una oportunidad.
