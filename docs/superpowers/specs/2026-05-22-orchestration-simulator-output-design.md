# Návrh: Orchestrace — Simulator, OutputManager a pure-snapshot logging

**Datum:** 2026-05-22
**Status:** Navrženo (čeká na review)
**Pod-projekt:** 2 ze 3 (Orchestrace)
**Předpoklad:** Pod-projekt 1 (Jádro engine) hotový — `PhysicsWorld`, dynamický
stavový vektor, `compute`-schopná `HybridIVPSolver._make_rhs`.

---

## Kontext a motivace

Po pod-projektu 1 je `PhysicsWorld` čistý fyzikální kontejner se stavovým
vektorem. Třída `World` ale stále nese orchestraci času i veškeré I/O (logování,
tvorba složek, grafy) — to je zbytek God Objectu. Zároveň logování IVP trajektorie
trpí chybou **CRIT-3**: po integraci se v cyklu přes `sol.t` znovu aplikují síly a
znovu řeší KKT (duplicitní kód), což u modelů se skrytým stavem korumpuje jejich
historii.

Tento pod-projekt:
1. rozdělí orchestraci na `Simulator` (čas/smyčka/události) a `OutputManager`
   (disk/logy/grafy),
2. zavede jedinou čistou vyhodnocovací funkci `compute_snapshot`, sdílenou
   solverem i logováním („pure snapshot replay"),
3. zachová `World` jako tenkou zpětně kompatibilní fasádu.

---

## Cíle a ne-cíle

**Cíle:**

- `Simulator` a `OutputManager` jako samostatné, testovatelné jednotky.
- Logování IVP přes `compute_snapshot` — žádný duplicitní force/KKT kód, žádná
  mutace historie modelů při zápisu (oprava CRIT-3 pro čisté síly).
- `World` zachová celé stávající veřejné API a chování (examples a testy beze změny).
- Volitelné per-komponentní `atol` škálování pro smíšený stavový vektor.

**Ne-cíle:**

- Retirování starých padákových modelů a NN (pod-projekt 3). Dokud existují
  stavové modely (`AdvancedParachute` s `air_mass_inside += …`), pure replay je
  korektní jen pro čisté síly; plné odstranění korupce dozní v pod-projektu 3.
- Změny fyziky/solveru nad rámec extrakce `compute_snapshot`.

---

## Architektura

### Sekce A — `compute_snapshot` (jedna čistá funkce)

Vyjmeme tělo `HybridIVPSolver._make_rhs` (z pod-projektu 1) do samostatné funkce,
kterou volá **solver i logger**. Funkce nesmí mutovat skrytou historii modelů;
jediné vedlejší efekty jsou doménově neškodné cache pro logger
(`body.force_categories`, vazbové síly).

```python
# core/snapshot.py
def compute_snapshot(physics, t, y, alpha, beta, quat_stab_k):
    """
    Čistá evaluace fyziky ve stavu (t, y). Vrací ydot pro solver a zároveň
    naplní force_categories + vazbové síly v tělesech (pro logger).
    Žádná mutace časové historie modelů.
    """
    physics.unpack_global_state(y)
    for b in physics.bodies:
        b.clear_forces()
    physics.apply_all_forces(t)                 # systémy + per-body + global + interaction
    Minv, J, F, rhs_v, _ = assemble_system(physics.bodies, physics.constraints, alpha, beta)
    a, lam = solve_kkt(Minv, J, F, rhs_v)
    _store_constraint_forces(physics.bodies, J, lam)   # cache pro logger
    return _assemble_ydot(physics, a, t, quat_stab_k)  # Tier 1 z KKT, Tier 2 lokálně
```

- `physics.apply_all_forces(t)` — nová metoda na `PhysicsWorld`, která sjednotí
  pořadí aplikace sil (systémy → per-body → global → interaction). Dnes je tahle
  smyčka duplikovaná v `World.step`, `HybridIVPSolver.rhs` i v logging replay.
- `solver.integrate` nastaví `rhs = lambda t, y: compute_snapshot(physics, t, y, …)`.
- Logger po integraci volá tutéž `compute_snapshot` v každém `sol.t`, pak zapíše.

### Sekce B — `Simulator` (orchestrace času)

```python
# core/simulator.py
class Simulator:
    def __init__(self, physics: PhysicsWorld, output: OutputManager | None = None,
                 payload_index=0, ground_z=0.0):
        self.physics = physics
        self.output = output
        self.payload_index = payload_index
        self.ground_z = ground_z
        self.t = 0.0
        self.t_touchdown = None
        self.termination_callback = None

    def run(self, solver, duration, dt, ...): ...        # fixed-step (dnešní World.run/step)
    def integrate_to(self, solver, t_end, ...): ...      # adaptivní (dnešní World.integrate_to)
    def set_termination_callback(self, fn): ...
```

`Simulator` drží `t`, `t_touchdown`, `payload_index`, `ground_z`,
`termination_callback` (orchestrace „kdy zastavit"). Touchdown event i fixed-step
termination logika se přesouvají sem. Po dokončení deleguje logování/grafy na
`OutputManager`, je-li nastaven.

### Sekce C — `OutputManager` (data a disk)

```python
# core/output.py
class OutputManager:
    def __init__(self, name, output_dir=Path("output"), auto_timestamp=True,
                 auto_save_plots=False):
        ...  # vytvoří output_path/{logs,plots}, inicializuje CSVLogger
    @property
    def logger(self) -> CSVLogger: ...
    def log(self, physics, t): ...
    def save_plots(self, physics, bodies=None, show=False): ...
```

Přebírá `enable_logging`, tvorbu složek, `CSVLogger`, `save_plots` z dnešního
`World`. Logging trajektorie po IVP integraci běží přes `compute_snapshot` (Sekce
A) → žádná duplicitní evaluace, žádná korupce.

### Sekce D — `World` jako tenká fasáda

`World` přestane sám orchestrovat; složí si `PhysicsWorld` + `Simulator` +
(volitelně) `OutputManager` a deleguje. Celé dnešní veřejné API zůstává:

```python
# core/simulation.py
class World:
    def __init__(self, ground_z=0.0, payload_index=0, simulation_name=None, ...):
        self.physics = PhysicsWorld(atmosphere=atmosphere)
        self.output = OutputManager(...) if simulation_name else None
        self.sim = Simulator(self.physics, self.output, payload_index, ground_z)

    # delegace kontejneru
    def add_body(self, b): return self.physics.add_body(b)
    def add_system(self, s): return self.physics.add_system(s)
    # ... add_global_force, add_constraint, add_interaction_force, WORLD ...
    @property
    def bodies(self): return self.physics.bodies
    # delegace orchestrace
    def run(self, solver, duration, dt, ...): return self.sim.run(...)
    def integrate_to(self, solver, t_end, ...): return self.sim.integrate_to(...)
    def enable_logging(self, name=None): ...   # vytvoří/napojí OutputManager
    def save_plots(self, ...): return self.output.save_plots(self.physics, ...)
    @property
    def t(self): return self.sim.t
    @property
    def t_touchdown(self): return self.sim.t_touchdown
```

Atributy, na které sahají testy/examples (`world.bodies`, `world.t`,
`world.logger`, `world.output_path`, `world.constraints`, …), se vystaví jako
delegující properties.

### Sekce E — Per-komponentní `atol` (volitelné)

`Simulator.integrate_to` složí pole `atol` per stavový slice: default skalární
`atol`, ale provider může nabídnout `state_atol() -> np.ndarray` (délka
`num_states()`). Smíšený vektor (pozice ~10³ vs. `V_air` ~10⁻³) tak řešič váží
správně. Pokud žádný provider škálu nenabídne, chování je beze změny (skalární).

---

## Datový tok (cílový stav)

**Integrace (IVP):** `Simulator.integrate_to` → `physics.build_layout()` →
`rhs = compute_snapshot(physics, …)` → `solve_ivp` (touchdown event přes slice) →
`physics.unpack_global_state(sol.y[:,-1])`, `sim.t = sol.t[-1]`.

**Logování:** pro každý `tk, yk` v `sol` → `compute_snapshot(physics, tk, yk, …)`
(naplní force_categories + vazbové síly, bez mutace) → `output.log(physics, tk)`.

---

## Testovací strategie (TDD)

| Test | Co chrání |
|---|---|
| `compute_snapshot` čistota | 2× volání se stejným (t,y) → identický ydot + force_categories, žádná mutace |
| `compute_snapshot` == solver rhs | ydot z `compute_snapshot` == ydot z integrace (konzistence) |
| Simulator free-fall | stejný výsledek jako dnešní `World.run`/`integrate_to` (regrese) |
| OutputManager složky | vytvoří `logs/`, `plots/`, CSV existuje a má řádky |
| World fasáda API | `world.bodies`, `world.t`, `world.run`, `world.enable_logging`, `world.save_plots` fungují beze změny |
| Logging bez korupce | čistá síla logovaná replayem dá stejné hodnoty jako při integraci |
| Per-component atol | vektor atol má správné per-slice hodnoty; bez override skalární |

Regrese: celá stávající sada testů (`tests/`) musí projít beze změny — to je hlavní
pojistka, že fasáda zachovala chování.

---

## Odsouhlasená rozhodnutí

- **Osud World:** tenká zpětně kompatibilní fasáda nad Simulator + OutputManager.
- **Logging:** pure replay přes `compute_snapshot` (ne cachování při akceptovaných
  krocích).

## Otevřené body pro implementační plán

- Přesné rozhraní `state_atol()` (volitelná metoda protokolu) — nebo odložit, pokud
  by to nafukovalo rozsah.
- Zda `force_breakdown` (dnešní `World` atribut pro fixed-step) zůstává na fasádě
  nebo se přesouvá na `Simulator`.
