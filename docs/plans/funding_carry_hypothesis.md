# Hipótesis registrada: carry de funding con delta neutral

Escrita el 2026-10-20, **antes de construir el backtest y de mirar ningún resultado** de esta
estrategia. La configuración primaria y los criterios de éxito no se cambian después de ver
resultados. La grilla de sensibilidad sirve para medir robustez, no para elegir parámetros.

## Idea

Cuando el funding de un perpetuo es persistentemente positivo, quienes están largos apalancados
pagan a quienes están cortos. La estrategia está **corta en el perpetuo y larga la misma cantidad
en spot**: no tiene exposición neta al precio y cobra el funding. Los riesgos son tres:
- que el funding baje o se invierta mientras la posición está abierta;
- que el basis (la diferencia entre perpetuo y spot) se mueva en contra;
- los costos de entrar y salir.

## Configuración primaria

**Universo:**
- Cada mes, el top 60 de Binance USDT-M por volumen, el mismo que en momentum
  (`artifacts/research/universe_top60_v1`).
- Se excluyen los símbolos sin par spot `<BASE>/USDT` con precio en Binance ese día.

**Señal** (al cierre de las 23:00 UTC del día *d*, sin mirar el futuro):
- Suma del funding realizado en los últimos 7 días, contando solo los eventos con fecha UTC ≤ *d*.
- Se anualiza como suma de 7 días × 365/7.

**Entrada y salida** (con histéresis):
- Se entra si el funding anualizado es ≥ **10 %**.
- Se sale si baja de **3 %** o si el símbolo sale del universo.
- Máximo **10 posiciones**. Si hay más candidatos, se eligen los de mayor funding. Una posición
  abierta solo se reemplaza si sale por la regla de salida.

**Tamaño:**
- Pesos iguales: **0.07 del capital por par** (spot largo 0.07 + perpetuo corto 0.07).
- Con 10 pares hay 0.70 del capital en spot y el resto queda como margen del perpetuo
  (apalancamiento 3x en el corto, aproximadamente 0.23).
- Si hay menos de 10 candidatos, la parte no usada queda en efectivo, con rendimiento 0.

**Ejecución:**
- Decisión diaria a las 23:00 UTC y ejecución a las 01:00 UTC del día siguiente, igual que en
  momentum.
- Banda de 0.02: no se reajustan pesos por diferencias pequeñas.

**Costos:**
- Perpetuo: 0.06 % de comisión taker.
- Spot: 0.10 % de comisión taker.
- Más slippage por nivel de liquidez en cada pata: 3 / 6 / 12 bp para los rangos 1–10 / 11–30 /
  31–60 del universo.

**Períodos:**
- Desarrollo: 2021-03-01 a 2023-12-31.
- Fuera de muestra: 2024-01-01 hasta el último dato.

## Criterios de éxito (fuera de muestra, después de costos)

Se tienen que cumplir **todos**:

| Criterio | Umbral |
|---|---|
| Sharpe | ≥ 1.5 |
| Meses positivos | ≥ 70 % |
| Peor caída | no peor que −10 % |
| Sharpe sin los 5 símbolos que más aportan | ≥ 1.0 |
| Sharpe en desarrollo (consistencia) | ≥ 0.5 |

**Se reporta, pero no es criterio:**
- Retorno anual. Se espera algo del orden de 5–25 %; no se optimiza.
- Rotación, costo anual, número medio de posiciones y meses sin ninguna posición.
- **Riesgo de margen:** la peor pérdida del perpetuo entre dos ejecuciones, como fracción del
  margen disponible, para ver si un pico de precio podría liquidar el corto antes de rebalancear.

## Prueba de combinación con momentum

La serie diaria neta del carry se combina con la del momentum primario, con **pesos de riesgo
iguales**: cada estrategia escalada a la misma volatilidad, calculada solo con datos de
desarrollo.

La combinación se considera útil si en el período fuera de muestra cumple **las dos** condiciones:
- Sharpe combinado ≥ Sharpe de momentum solo + 0.2.
- Peor caída mejor que la de momentum solo.

## Sensibilidad (solo para medir robustez)

| Parámetro | Valores |
|---|---|
| Umbral de entrada | 5 %, 10 %, 20 % |
| Ventana | 3, 7, 14 días |
| Máximo de posiciones | 5, 10, 20 |
| Costos | doble del primario |

Se reporta qué fracción de la grilla supera Sharpe 1.0 fuera de muestra.

## Limitaciones conocidas

- **Datos de Binance:** el backtest usa spot y funding de Binance; la ejecución sería en Bitget.
  El funding de Bitget fue ~40 % más alto y con correlación 0.4 con el de Binance, así que
  habría que repetir el backtest con datos de Bitget antes de un paper.
- **Margen en dos cuentas:** el spot y el perpetuo viven en cuentas separadas salvo que se use
  la cuenta unificada. Un pico de precio genera pérdida en el corto y ganancia en el spot en
  cuentas distintas. Por eso se reporta el riesgo de margen.
- **Efectivo sin rendimiento:** no se modela el rendimiento de stablecoins sobre el efectivo
  ocioso.
