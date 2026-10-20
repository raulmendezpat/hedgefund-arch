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

## Resultado (2026-10-20): hipótesis rechazada

Corrida: `scripts/run_funding_carry.py` → `artifacts/research/funding_carry_v1/`. 337 perpetuos
del top-60; 270 con par spot en Binance; 46 contratos deslistados.

**Corrección de datos (no de parámetros).** Después del delisting, Binance sigue entregando
velas del perpetuo congeladas en el precio de liquidación, con volumen cero. La primera corrida
mantenía esos pares abiertos con la cobertura rota; DEGO sola restaba −8.9 %. Ahora el
perpetuo se considera muerto desde el inicio de su racha final de velas planas sin volumen
(≥24 h), y el par se cierra ahí. Antes de la corrección el Sharpe fuera de muestra era −0.37.

| Criterio (fuera de muestra) | Umbral | Resultado | ¿Cumple? |
|---|---|---|---|
| Sharpe | ≥ 1.5 | 1.13 | no |
| Meses positivos | ≥ 70 % | 38 % | no |
| Peor caída | ≥ −10 % | −3.7 % | sí |
| Sharpe sin los 5 mejores símbolos | ≥ 1.0 | 1.14 | sí |
| Sharpe en desarrollo | ≥ 0.5 | 3.73 | sí |

- **Retorno:** CAGR fuera de muestra 2.7 % con volatilidad 2.3 %. Funding +5.8 %/año, costos
  −3.4 %/año, precio +0.2 %/año: la cobertura funciona.
- **El edge se está extinguiendo:**
  - Sharpe por año: 2021 8.6 · 2022 −2.6 · 2023 2.3 · 2024 7.8 · 2025 0.6 · 2026 −1.9.
  - Desde 2025 casi todos los meses son levemente negativos: el funding promedio ya no paga
    la rotación.
- **Meses positivos bajos con Sharpe positivo:** el retorno se concentra en pocos meses de
  funding alto (2024-03 +3.4 %, 2025-03 +3.3 %). El resto son pérdidas pequeñas por costos.
- **Costos ×2:** Sharpe fuera de muestra 0.34.
- **Grilla:**
  - 59 % de las configuraciones con Sharpe fuera de muestra ≥ 1; mediana 1.33.
  - Las de entrada 20 % rinden más, entre 8 y 13 %/año.
  - Es una lectura posterior a ver los datos, así que no habilita a cambiar el primario.
- **Combinación con momentum:**
  - En el papel cumple: Sharpe fuera de muestra 1.49 contra 1.00, y peor caída −24 % contra
    −32 %.
  - Pero exige escalar el carry ×19 para igualar la volatilidad de momentum, unas 13 veces el
    capital en notional por pata.
  - Con un short que llegó a subir 93 % en un día (1000SHIB) y una pérdida del perpetuo de hasta
    44 % del margen libre ya sin apalancar, esa escala no es implementable. **No se considera
    válida.**
- **Riesgo de delisting:** aun con la corrección, DEGO costó −1.5 %. El spot se desplomó
  mientras el perpetuo seguía en el universo. Es un riesgo real de la estrategia, no un
  artefacto.

**Conclusión:** el carry de funding delta-neutral, con costos realistas, rinde un 2–4 %/año con
muy baja volatilidad, y la tendencia de 2025–2026 es negativa. No pasa al paper.
