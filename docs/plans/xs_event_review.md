# Revisión semanal de eventos y de mapeo del momentum (solo alertas)

Escrita el 2026-10-28, antes de su primera corrida. Es un monitor: **no cambia la estrategia
registrada ni las órdenes**. Corre aparte del ejecutor, y si falla, el rebalanceo sigue igual.

## Qué hace

Cada lunes, entre que cierra la vela de las 23:00 UTC del domingo (00:00 UTC) y el fill de la
01:00 UTC, calcula la cartera objetivo de la semana con el mismo código que el paper y el ejecutor
(`xs_signal.compute_targets`) y revisa dos cosas:

1. **Mapeo Binance ↔ Bitget** (todos los miembros operables del universo, ~60):
   - compara el precio en los dos exchanges;
   - el cociente debe ser 1, o la potencia de 10 que indica el prefijo (`1000PEPE` en Binance contra
     `PEPE` en Bitget da 1/1000);
   - si se aleja más de un 2 %, los dos símbolos probablemente no son el mismo activo
     (ticker reutilizado, redenominación o alias equivocado) y avisa.
   - Es una regla fija, sin LLM: es más confiable que preguntarle a un modelo.
2. **Eventos** (cada posición de la cartera, ~24):
   - Claude (`claude-sonnet-5-5`, esfuerzo `low`, hasta 3 búsquedas web por token) busca eventos
     discretos de los últimos 14 días o programados para los próximos 7: delistings y listings,
     hackeos, unlocks, migraciones o redenominaciones, acciones regulatorias, depegs, ventas
     del equipo, airdrops y upgrades;
   - ignora comentarios de precio y análisis técnico;
   - califica el riesgo **para el lado de la posición** (un listing nuevo es riesgo para un corto;
     un unlock, para un largo) como `none`, `low`, `medium` o `high`, citando la fuente.

**Notificaciones:** un resumen en cada corrida, y una alerta con sonido por cada mapeo dudoso y
por cada posición con riesgo `medium` o `high`. Si corre después del fill (por ejemplo, porque el
Mac estuvo apagado), lo indica.

## Qué hacer con una alerta

No existe un mecanismo para excluir una posición, y no se agrega: sería cambiar la estrategia.
Con una alerta grave antes del fill, la única acción disponible es pausar el ejecutor en vivo esa
semana (`launchctl unload` de su agente), lo que cuenta como rebalanceo perdido. Es una decisión
manual del dueño de la cuenta.

## Costo

~0.08 USD por posición revisada (tokens más búsquedas): unos 2 USD por semana. Cada corrida
guarda su costo en el reporte.

## Cómo se mide

Cada revisión queda en `runtime/event_review/reviews/<decision_day>.json`, con la cartera, las
calificaciones y las fuentes. **El 2027-05-03 (26 semanas):** se compara el P&L semanal del paper
de las posiciones marcadas (`medium`/`high`) contra las demás.
- Es descriptivo: con pocas alertas por semana no habrá potencia estadística.
- Usar las alertas para operar (por ejemplo, excluir posiciones marcadas) sería una hipótesis
  nueva, con su propio registro y su propio paper.
