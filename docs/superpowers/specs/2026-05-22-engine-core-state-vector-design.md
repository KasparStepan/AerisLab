# Návrh: Jádro engine — generalizovaný stavový vektor a protokoly

**Datum:** 2026-05-22
**Status:** Navrženo (čeká na review)
**Pod-projekt:** 1 ze 3 (Jádro engine)

---

## Kontext a motivace

AerisLab je 6-DOF engine pro simulaci záchranných systémů (padák–náklad), řešený
KKT formulací s Baumgarteho stabilizací. Cílem širšího refaktoringu je připravit
engine na integraci NN modelu padáku (predikce odporové síly a objemu vzduchu /
přidané hmoty) a na stabilní běh pod adaptivními řešiči (scipy Radau).

Při analýze stávajícího kódu a tří návrhových dokumentů
([Gemini.MD](../../Gemini.MD), [Gemini_world_refactor.md](../../Gemini_world_refactor.md),
a první návrh do solveru) vyplynuly tři propojené problémy:

1. **Solver má natvrdo `13` stavů na těleso.** [solver.py](../../../src/aerislab/core/solver.py)
   (`_pack`, `_unpack_to_world`, `rhs`) předpokládá `13*k` offsety. Tělesa s více
   stavovými veličinami (padák s objemem vzduchu, raketa s palivem) nejdou
   vyjádřit.

2. **Stávající padákové modely mutují skrytý stav uvnitř `rhs`.**
   [parachute_models.py](../../../src/aerislab/models/aerodynamics/parachute_models.py)
   používá akumulátor `air_mass_inside += dm_dt * dt` (s natvrdo `dt=0.01`) a
   konečné diference z `prev_time`/`prev_velocity`. To je nekompatibilní s
   adaptivním řešičem, který volá `rhs` vícekrát na krok, v nemonotónním pořadí
   času a při perturbaci pro Jacobián.

3. **`World` je God Object** (916 řádků): míchá fyzikální kontejner, orchestraci
   času a I/O (logování, tvorba složek, grafy).

### Dekompozice na pod-projekty

Celý refaktoring je na jednu specifikaci příliš velký. Dělí se na tři sekvenční
pod-projekty, každý s vlastním spec → plán → implementace:

1. **Jádro engine (tento dokument):** vytržení `PhysicsWorld` z `World` + zavedení
   `StateProvider` / `AuxDynamics` / `InertialProvider` + dynamický stavový vektor
   `y` + změny solveru (added mass na LHS, Baumgarte kvaternion). Zde vzniká
   obecnost a páteř. State Vector Move přirozeně bydlí v `PhysicsWorld`, proto se
   dělá společně s jeho vytržením.

2. **Orchestrace:** `Simulator` (časová smyčka, události, termination) +
   `OutputManager` (logování, složky, grafy) + „pure snapshot replay" logging,
   který opravuje double-RHS / mutaci stavu při logování (CRIT-3).

3. **Padákový model:** vyhození zastaralých modelů, NN přes `Force` protokol +
   objem→added-mass, ponechání jednoho čistého analytického baselinu pro V&V.

Tento dokument pokrývá **pouze pod-projekt 1.**

---

## Cíle a n-cíle

**Cíle:**

- Solver nezná `13`; pracuje s libovolným počtem stavů na komponentu.
- Doménově hloupé jádro — solver/kontejner neví nic o padáku ani raketě; nová
  fyzika je jen kombinace protokolů, bez zásahu do solveru.
- Korektní a stabilní zacházení s proměnnou hmotou/setrvačností (přidaná hmota
  padáku, úbytek paliva rakety).
- Kvaternion drží jednotkovou normu hladce, bez korumpování Jacobiánu řešiče.
- Zachovat chování stávajících scénářů (regrese) pro tělesa s konstantní hmotou.

**Ne-cíle (mimo tento pod-projekt):**

- `Simulator`/`OutputManager` split a oprava logging double-RHS (pod-projekt 2).
- NN model padáku, ONNX, trénink, slack-tether šňůry (pod-projekt 3).
- Pružná/FEM tělesa, PDE pole, non-holonomní vazby (mimo rozsah celého záměru).

---

## Architektura

### Sekce A — Model komponent a dva tiery stavů

Stavové veličiny nejsou všechny stejné; existují dva druhy:

- **Tier 1 — mechanické DOF (tuhá tělesa):** stavy `[p, q, v, w]` (13). Derivace
  `[v, q̇, a_lin, a_ang]`, kde `a_lin, a_ang` **nepočítá těleso samo** — vychází z
  globálního provázaného KKT řešení `a = M_eff⁻¹(F + JᵀΛ)`. Vlastníkem výpočtu
  derivace je **solver**.

- **Tier 2 — pomocné stavy (např. `V_air`):** žádná algebraická vazba, žádný vliv
  na KKT. Derivaci si komponenta **počítá lokálně sama** (z NN nebo analytického
  modelu). Solver je jen integruje vedle.

Z toho plynou **tři úzké protokoly** (plus stávající `Force`), každý odpovídá na
jednu ortogonální otázku:

```python
class StateProvider(Protocol):          # "Nesu stavy do y?"
    def num_states(self) -> int: ...
    def pack_state(self, out: np.ndarray) -> None: ...   # zapíše do out (svůj slice)
    def unpack_state(self, y: np.ndarray) -> None: ...    # přečte ze svého slice

class AuxDynamics(Protocol):            # "Počítám si derivace lokálně sám?" (Tier 2)
    def compute_derivatives(self, t: float) -> np.ndarray: ...

class InertialProvider(Protocol):       # "Přispívám na LHS (hmota/setrvačnost)?"
    def mass_matrix_world(self) -> np.ndarray: ...      # 6×6, vč. M_added
    def inv_mass_matrix_world(self) -> np.ndarray: ...  # 6×6 inverzní
```

| Protokol | Otázka |
|---|---|
| `StateProvider` | Nesu čísla, která integrátor posouvá v čase? |
| `AuxDynamics` | Počítám si derivace lokálně sám (neřídí mě KKT)? |
| `InertialProvider` | Přispívám na levou stranu (hmota/setrvačnost)? |
| `Force` (existuje) | Přispívám na pravou stranu (zobecněná síla)? |

**Mapování na konkrétní objekty:**

- `RigidBody6DOF`: `StateProvider` (13) + `InertialProvider` (dnes konstantní,
  přepíše se na `M_body + M_added`). Derivace dodá solver.
- Padák (pod-projekt 3): `StateProvider` (`V_air`, …) + `AuxDynamics` (`V̇_air`) +
  `Force` (odpor + `ṁ_added·v`) + přispívá do `M_added`. Vše čte z jednoho `V_air`.
- Raketa (budoucí): `StateProvider` (`m_fuel`) + `AuxDynamics` (rychlost hoření) +
  `InertialProvider` (klesající `m`) + `Force` (tah).

**Proč ne jeden tlustý protokol:** smícháním Tier 1 a Tier 2 do jediného
`compute_derivatives(t)` by těleso muselo nahlížet do globálního KKT (rozbije
izolaci) nebo dělat KKT lokálně (nesmysl). Rozdělení drží solver doménově hloupý
a každý protokol testovatelný zvlášť.

**Hranice (záměrně nepokryto):** mechanické jádro = tuhá 6-DOF tělesa + holonomní
vazby přes KKT. `AuxDynamics` je pro lumped ODE stavy (derivace = funkce stavu a
času), ne pro PDE ani DAE (algebraickou vazbu mezi pomocnými stavy).

### Sekce B — Globální stavový vektor `y` a vlastnictví slices

`PhysicsWorld` (vytržený z `World`) si postaví **registr layoutu** z `num_states()`
každého provideru. Tím ze solveru mizí každé natvrdo psané `13`.

```python
class PhysicsWorld:
    def _build_layout(self) -> None:
        self._layout = []          # [(provider, lo, hi), ...]
        offset = 0
        for p in self.state_providers:        # tělesa + aux komponenty
            n = p.num_states()                # těleso → 13, padák → 1, ...
            self._layout.append((p, offset, offset + n))
            offset += n
        self._n_states = offset               # celková velikost y (ne 13*N)

    def pack_global_state(self) -> np.ndarray:
        y = np.empty(self._n_states)
        for p, lo, hi in self._layout:
            p.pack_state(y[lo:hi])
        return y

    def unpack_global_state(self, y: np.ndarray) -> None:
        for p, lo, hi in self._layout:
            p.unpack_state(y[lo:hi])
```

**Dvě koexistující indexace** (nezaměňovat):

| Indexace | Co indexuje | Velikost | Týká se |
|---|---|---|---|
| Mechanická (KKT) | zobecněné rychlosti/zrychlení pro `M, J, F, a` | 6 / těleso | jen tuhá tělesa |
| Stavová (IVP `y`) | stavy integrované scipy | `num_states()` / provider | všechny providery |

Těleso má 6 mechanických DOF (do KKT) i 13 IVP stavů (do `y`). Generalizovaná
`rhs` obě vrstvy propojí bez `13`:

```python
def rhs(t, y):
    world.unpack_global_state(y)
    world.clear_and_apply_forces(t)           # Force protokol
    a = solve_kkt(*assemble_system(...))      # mechanická zrychlení, 6/těleso

    ydot = np.empty_like(y)
    for p, lo, hi in world._layout:
        ydot[lo:hi] = p.state_derivative(t, a)  # Tier 1 čte a, Tier 2 ignoruje
    return ydot
```

Rozlišení Tier 1 / Tier 2 bude přes protokol (ne `isinstance`); přesný mechanismus
(např. těleso dostane svůj blok `a`, aux provider volá `compute_derivatives`)
doladí implementační plán.

### Sekce C — Změny v solveru

1. **Matice hmotnosti přes `InertialProvider`.** Stávající
   [assemble_system](../../../src/aerislab/core/solver.py) bere
   `b.inv_mass_matrix_world()` přímo. Nově se blok staví z `(M_body + M_added)⁻¹`.
   Anizotropní přidaná hmota → translační blok je plná 3×3
   `m·I + R·diag(m_t, m_t, m_a)·Rᵀ`, invertovaná **analyticky přes rotaci**
   (`R·diag(1/(m+m_t), 1/(m+m_t), 1/(m+m_a))·Rᵀ`), ne `np.linalg.inv`. Pozn.: tím
   se opravuje rozpor v Gemini.MD, kde 1.1 zavádí anizotropní tenzor, ale 3.2
   navrhuje skalární inverzi `1/(m+m_added)`. Pro konstantní těleso `M_added = 0`
   → musí vyjít přesně dnešní `inv_mass_matrix_world`.

2. **Člen `ṁ_added·v` jako `Force`.** Síla od přidané hmoty se rozpadá na dva členy:
   `F = −(ṁ_added·v) − (m_added·a)`. Člen `m_added·a` jde na LHS (do `M_eff`),
   protože závisí na neznámém zrychlení (explicitní zacházení je nestabilní pro
   velkou přidanou hmotu). Člen `ṁ_added·v` jde na RHS přes `Force`, čte stejné
   `V̇_air` jako `M_added` (jeden zdroj pravdy).

3. **Kvaternion — Baumgarteho stabilizace v `q̇`:**
   `q̇ = ½·Ω(ω)·q − k·(qᵀq − 1)·q`. Norma relaxuje exponenciálně (`ṡ ≈ −2k·s` pro
   `s = |q|²−1`), stejná filozofie jako stávající Baumgarte na vazby. **Zrušit**
   `quat_normalize` uvnitř `rhs` ([solver.py:520, 560](../../../src/aerislab/core/solver.py)),
   protože nespojitě koruptuje Jacobián Radau. Důvod, proč to NENÍ KKT vazba:
   `|q|=1` je invariant ODE (`d/dt(qᵀq) = qᵀΩq = 0`, antisymetrie), ne mechanická
   vazba; `q` není v akceleračním vektoru, řádek v `J` by vstříkl fiktivní moment
   do rotační dynamiky.

4. **Generalizovaný packing** ze Sekce B nahradí `13*k` offsety v `_pack` /
   `_unpack_to_world` / `rhs`.

### Datový tok per krok `rhs` (cílový stav)

`unpack_global_state(y)` → `clear_forces` → aktualizace komponent (Tier 2 přečtou
svůj stav) → `apply_all_forces` (Force, vč. `ṁ·v`) → `assemble_system` (M_eff s
added mass) → `solve_kkt` → poskládání `ydot` (Tier 1 z KKT vč. Baumgarte `q̇`,
Tier 2 z `compute_derivatives`). **Žádná mutace skryté historie modelů.**

---

## Testovací strategie (TDD)

Psát testy průběžně, v duchu stávající V&V kultury projektu.

| Test | Co chrání |
|---|---|
| Regrese trajektorie | stávající scénáře (`M_added = 0`) dají stejný výsledek do tolerance |
| Čistota `rhs` | `rhs(t, y)` 2× → identický `ydot` a žádná mutace stavu |
| Roundtrip `y` | `pack → unpack → pack` je identita pro libovolnou sadu providerů |
| Layout | přidání aux stavů neposune indexy těles ani `payload_index` |
| Norma kvaternionu | `\|q\|²−1` exponenciálně klesá (analogie Baumgarte-decay testu) |
| Variabilní hmota | analytický případ proti známému řešení (Ciolkovskij / zachování hybnosti) |

---

## Odsouhlasená rozhodnutí

- **Sémantika hmoty:** hybrid — `V_air` (resp. `V̇_air`) je jediný zdroj pravdy;
  `m_added·a` na LHS, `ṁ_added·v` na RHS přes `Force`.
- **Sekvence:** `PhysicsWorld` + state-vektor společně v jednom pod-projektu;
  `Simulator`/`OutputManager` až jako pod-projekt 2.
- **Kvaternion:** Baumgarteho člen v `q̇` (ne KKT vazba, ne renormalizace na výstupu).
- **Staré modely:** vyhodit v pod-projektu 3, ponechat jeden analytický baseline.

## Otevřené body pro implementační plán

- Přesný mechanismus rozlišení Tier 1 / Tier 2 v `rhs` (čistě přes protokol).
- Vztah `state_providers` ↔ stávající `bodies` / `systems` v `PhysicsWorld`.
- Volba konstanty `k` pro kvaternion Baumgarte (řádově jako existující `alpha`).
- Per-komponentní `atol` pro scipy (škálování stavů těles vs. aux stavů) — možná až
  pod-projekt 2, ale zmínit.
