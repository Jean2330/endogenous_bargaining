# Lema A.1 — Boundedness of the equilibrium slope (`lem:beta_bounds`)

Corrida de prueba del skill `proof-draft-audit` sobre una demostración ya escrita en
`bargainigpower_draft.tex` (Apéndice A, `app:beta_bounds`). Se arrancó directo en
Auditoría (Ronda 1), sin fase de redacción inicial, siguiendo la regla del skill para
demostraciones preexistentes.

**Convención económica:** este documento es contratación principal-agente con riesgo
moral y responsabilidad limitada, no un juego de señalización — se usó el vocabulario
propio del paper (principal, agente, $\beta_t$, $W_t$, $\underline w_t$, $\overline\Pi$,
$\overline{\text{CE}}$, $\delta$) en vez del registro Emisor/Receptor del skill (calibrado
para otro proyecto).

## Enunciado

**Lemma A.1 (Boundedness of the equilibrium slope).** Bajo la Asunción de regularidad
del bracket binding (`ass:beta_bounds_reg`), a $W_t$ fijo en el régimen binding, si

- (i) $\Pi_t(0) = -\underline w_t > \overline\Pi$,
- (ii) $\text{CE}_t(0) = \underline w_t < \overline{\text{CE}}$,
- (iii) $\Pi_t(1) < \overline\Pi$,

junto con la existencia de $\beta\in(0,1)$ donde ambos superávits son simultáneamente
positivos, entonces la pendiente Nash-óptima $\beta_t^*$ que resuelve la FOC binding
(`eq:foc_binding`) se encuentra en $(0,1)$.

## Ronda 1

### Auditoría — Ronda 1 (Auditor)

Reverificado con sympy: $\Pi_t(0)=-\underline w_t$, $\Pi_t(1)=\underline y-\underline w_t$,
$\text{CE}_t(0)=\underline w_t$, $\Pi_t'=\mu(1-2\beta)+\underline y$,
$\text{CE}_t'=(\mu-\kappa)\beta-\underline y$, $\Pi_t''=-2\mu$, $\text{CE}_t''=\mu-\kappa$, y
la equivalencia entre la Asunción y (i)-(iii) — todo correcto, sin error algebraico.

**Hallazgo 1 (rigor/notación, menor).** "*Π_t(0)>Π̄>Π_t(1), so by the fact above there is
a hi∈(0,1)...*": el "fact about concave functions" enunciado es condicional (si
$g(\beta_0)=c$ entonces $g'(\beta_0)<0$) y no prueba existencia de $\beta_0$. La
existencia viene de IVT (continuidad), no del "fact". La prueba sí nombra IVT
explícitamente para los cruces de $\text{CE}_t$ y para la raíz final de $G$ — inconsistencia
de cita en este punto específico.

**Hallazgo 2 (corrección/cohesión, sustantivo).** Ambos casos cerraban con "*the second
order condition of the Nash problem... ensures this root is the unique maximizer*", sin
prueba ni cita, conflacionando SOC local con unicidad global. El Lema no necesita
unicidad para su conclusión (solo existencia + interioridad). Peor: en el caso
$\mu<\kappa$, la Proposición 1.1 (concavidad de la frontera) del mismo documento dice que
la concavidad ahí **no** es incondicional — requiere condiciones adicionales. Afirmar
unicidad vía SOC sin esas condiciones es inconsistente con la Proposición 1.1.

**Veredicto:** OTRA RONDA (Hallazgo 2 es sustantivo).

## Ronda 2

### Demostración — Ronda 2 (Redactor)

Cambios aplicados directamente en `bargainigpower_draft.tex`:

1. **Hallazgo 1:** separada la existencia de `hi` (ahora atribuida a IVT explícitamente)
   de la conclusión $\Pi_t'(\mathrm{hi})<0$ (que sí viene del "fact").
2. **Hallazgo 2:** reemplazada la oración de unicidad no justificada por un argumento
   autocontenido: el producto de Nash
   $N(\beta):=[\Pi_t(\beta)-\overline\Pi]^{1-\delta}[\text{CE}_t(\beta)-\overline{\text{CE}}]^{\delta}$
   se extiende continuamente a $[\mathrm{lo},\mathrm{hi}]$ (o $[\mathrm{lo},\mathrm{hi}^*]$
   en el caso $\mu<\kappa$) con valor $0$ en ambos extremos (un factor se anula en cada
   uno) y valor positivo en algún punto interior (por la no-vacuidad del bracket
   factible). Por el Teorema del Valor Extremo, el máximo de $N$ se alcanza en un punto
   interior; ahí $N$ es diferenciable, así que ese punto es crítico de $N$, es decir raíz
   de $G=(\log N)'$ — identidad verificada independientemente con sympy
   ($G - (\log N)' \equiv 0$ simbólicamente). Esto identifica a $\beta_t^*$ mismo (no solo
   "una raíz de $G$") como interior, sin invocar concavidad global del producto de Nash.
   Se añadió un comentario entre paréntesis en cada caso, citando la Proposición 1.1
   (a la que se le agregó `\label{prop:frontier_concavity}`, que no tenía) para explicar
   por qué la unicidad sí está disponible cuando $\mu\ge\kappa$ (concavidad incondicional)
   pero deliberadamente no se reclama cuando $\mu<\kappa$ (concavidad condicional).
3. Autocorrección adicional (no señalada por el Auditor): el primer paréntesis citando la
   Proposición 1.1 originalmente decía que esta "pins down uniqueness" sin probar que
   concavidad de la frontera implique cuasi-concavidad del producto de Nash — el mismo
   tipo de salto que motivó el Hallazgo 2. Se suavizó a "the standard sufficient
   condition for uniqueness... we do not verify that implication here."

### Auditoría — Ronda 2 (Auditor)

Relectura completa del Apéndice A tal como quedó en el `.tex`.

- **Corrección matemática:** el argumento de Valor Extremo es válido paso a paso —
  $N$ bien definida y continua en $[\mathrm{lo},\mathrm{hi}]$ (nunca se eleva una base
  negativa a potencia fraccionaria, porque ambos factores son $\ge 0$ en todo el
  intervalo por construcción), $N=0$ solo en los extremos, $N>0$ en el interior por
  no-vacuidad, EVT da un máximo interior, diferenciabilidad ahí, Fermat da
  $N'=0\Rightarrow(\log N)'=0=G$. Verificado simbólicamente que $G\equiv(\log N)'$.
- **Rigor/notación:** Hallazgo 1 resuelto — IVT y el "fact" ahora citados por separado.
  El nuevo paréntesis sobre la Proposición 1.1 ya no afirma la implicación
  concavidad-de-frontera $\Rightarrow$ unicidad-del-producto-de-Nash como probada, la
  presenta como condición suficiente estándar y explícitamente declara que no se
  verifica aquí — nivel de rigor apropiado para un comentario que el propio texto marca
  como no necesario para la conclusión del lema.
- **Cohesión/estructura:** el argumento de EVT se inserta como continuación natural de
  "By the IVT, G has a root in (lo,hi)"; el paréntesis queda al final de cada caso, sin
  interrumpir el flujo principal.
- **Claridad expositiva:** cada paso trae su justificación (IVT, EVT, Fermat,
  diferenciabilidad); no quedan saltos que un lector experto tenga que reconstruir.
- **Registro económico:** se mantiene "Nash product", "Nash-optimal slope", "Nash
  bargaining problem selects" — no degenera en análisis puro.

Único punto menor detectado (no bloqueante): "$N$ extends continuously to $[\mathrm{lo},\mathrm{hi}]$"
es una descripción ligeramente imprecisa — $N$ ya está definida y es continua ahí
directamente por la fórmula (no hace falta "extenderla"). Es una preferencia de
redacción, no un error ni una laguna de rigor.

**Veredicto: CONVERGE.**

## Versión final

Ver `bargainigpower_draft.tex`, Apéndice A (`app:beta_bounds`), líneas ~581-602 (Lemma y
prueba), y la Proposición 1.1 (`prop:frontier_concavity`, sección `subsec:nash`) a la que
ahora se cita por nombre.

## Resumen de convergencia

- Rondas totales: 2
- Estado final: CONVERGE
- Cambios principales entre Ronda 1 y la versión final:
  1. Cita de existencia (IVT) separada de la cita de signo de derivada (el "fact").
  2. Oración de unicidad no justificada reemplazada por un argumento de Valor Extremo
     autocontenido, que identifica a $\beta_t^*$ mismo como interior sin apelar a
     concavidad global no probada.
  3. Se agregó `\label{prop:frontier_concavity}` a la Proposición 1.1 (no tenía) para
     poder citarla por nombre desde el Apéndice A.
  4. El comentario sobre unicidad cuando $\mu\ge\kappa$ se formuló como condición
     suficiente estándar, no como hecho probado en el texto.
