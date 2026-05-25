# Architektonický refaktoring: třída `World` a čisté logování (IVP)

**Kontext:** Tento dokument popisuje řešení dvou kritických architektonických
problémů v enginu AerisLab:

1. Přílišnou komplexitu třídy `World` (*God Object* anti-pattern).
2. Dvojité počítání fyziky s mutací vnitřního stavu během logování adaptivního
   IVP řešiče (chyba **CRIT-3**).

**Cíl:** Vytvořit testovatelný, modulární kód a zajistit, aby zpětná
re-evaluace fyziky pro zápis do CSV nekorumpovala paměť ML a pokročilých
padákových modelů. Oba problémy úzce souvisejí s budoucí integrací ML modelů
a adaptivních řešičů (SciPy Radau).

---

## 1. Rozdělení třídy `World` (konec "God Objectu")

### 1.1 Identifikace problému

Současná třída `World` (`simulation.py`, ~760 řádků) nese příliš mnoho
zodpovědností (*God Object anti-pattern*). Stará se o:

- Správu těles a vazeb (fyzika).
- Držení času a ukončení (`t`, `t_touchdown`, `ground_z`).
- Řešení integrace (`run`, `integrate_to`).
- Vytváření složek na disku a logování (`self.logger`, `enable_logging`).
- Generování grafů (`save_plots`).

Tato provázanost znemožňuje snadné spouštění tisíců simulací v paměti (např. pro
trénink ML nebo Monte Carlo) bez zbytečného I/O overheadu a brání testování
i rozšiřování.

### 1.2 Cílová architektura (Separation of Concerns)

Logika se rozdělí do tří nezávislých tříd.

#### A. `PhysicsWorld` (čistý fyzikální kontejner)

Stará se POUZE o mechanický a aerodynamický stav. Nezná čas (ten řídí řešič),
nezná složky, nezná CSV. Obsahuje tělesa, systémy, vazby, atmosféru a metody pro
sbalení/rozbalení stavového vektoru.

```python
# aerislab/core/world.py
class PhysicsWorld:
    """Čistý kontejner pro fyzikální stav simulace."""

    def __init__(self, atmosphere: AtmosphereModel = None):
        self.bodies = []
        self.systems = []
        self.global_forces = []
        self.interaction_forces = []
        # Atmosféra je "služba" pro aerodynamické síly
        self.atmosphere = atmosphere if atmosphere else StandardAtmosphere()

    # Dynamický sběr stavů (State Vector Move z předchozího návrhu)
    def pack_global_state(self) -> np.ndarray:
        """Sbalí aktuální vnitřní stavy do 1D pole."""
        ...

    def unpack_global_state(self, y: np.ndarray) -> None: ...
```

#### B. `OutputManager` (správa dat a disku)

Vytváří adresářovou strukturu, inicializuje loggery (CSV/Parquet) a generuje
grafy.

```python
# aerislab/core/output.py
class OutputManager:
    """Spravuje ukládání logů, složek a grafů na disk."""

    def __init__(self, simulation_name: str, base_dir: Path = Path("output")):
        self.output_path = base_dir / f"{simulation_name}_{timestamp}"
        self.logs_dir = self.output_path / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = CSVLogger(self.logs_dir / "simulation.csv")

    def save_plots(self, world_state):
        # Generování grafů na základě logů
        ...
```

#### C. `Simulator` / `Runner` (orchestrátor času)

Propojuje `PhysicsWorld` s řešičem a `OutputManager`em. Stará se o integrační
smyčku a události (např. dopad na zem).

```python
# aerislab/core/simulator.py
class Simulator:
    """Orchestrátor, který posouvá PhysicsWorld v čase."""

    def __init__(self, world: PhysicsWorld, output: OutputManager = None):
        self.world = world
        self.output = output
        self.t = 0.0
        self.termination_callbacks = []

    def integrate_ivp(self, solver: HybridIVPSolver, t_end: float):
        # Předá solveru čistý world a řeší posun v čase
        ...
```

---

## 2. Odstranění "Double-RHS" a mutací při logování IVP

### 2.1 Identifikace problému (příčina CRIT-3)

SciPy `solve_ivp` vrací pouze pole časů `sol.t` a stavů `sol.y`. Protože kód
potřebuje do CSV zapsat i velikosti sil ($F$) a vazbových reakcí ($\lambda$),
bere současný `solver.py` výsledná pole ze SciPy a v dlouhém cyklu znovu
integruje stavy tělísek do světa a pouští evaluaci všech sil.

Tím, že modely jako `ADDED_MASS` používají `self.prev_time` nebo akumulují `dt`
(např. `self._state.air_mass_inside`), tyto zpětné rychlé průchody zcela zničí
jejich časové derivace a vedou k nesmyslným rázovým silám.

### 2.2 Řešení: koncept "Pure Snapshot Replay"

Základním pravidlem pro adaptivní řešiče je, že vyhodnocení silových funkcí
(RHS) **nesmí mutovat vnitřní historii modelů**. Vše, co se vyvíjí v čase
($m_{air}$, plocha), musí být součástí vektoru $y$. Výpočet fyziky se
centralizuje do jediné čisté funkce (*pure function*), která slouží jak pro
SciPy integrátor, tak pro logování.

#### Centrální vyhodnocovací funkce

Sjednotí logiku výpočtu sil a KKT. Tato funkce pouze "vyfotí" stav a spočítá
zrychlení/síly, ale nenapáchá žádné trvalé změny v historii modelů (žádné
`+= dt`).

```python
def compute_snapshot(world: PhysicsWorld, t: float, y: np.ndarray) -> np.ndarray:
    """
    ČISTÁ FUNKCE pro vyhodnocení fyziky v čase t a stavu y.
    Zajistí i naplnění pomocných cache (např. b.force_categories) pro Logger.
    Nesmí obsahovat jakoukoliv mutaci vnitřní historie (žádné ukládání dt).
    """
    # 1. Zápis stavů bez normalizace a bez ovlivnění logiky modelů
    world.unpack_global_state(y)
    for b in world.bodies:
        b.clear_forces()

    # 2. Pomocné derivace (např. ML model padáku, objem vzduchu) — čtení stavu y
    aux_derivs = []
    if hasattr(world, "aux_components"):
        for c in world.aux_components:
            aux_derivs.append(c.compute_derivatives(t))

    # 3. Aplikace fyzikálních sil
    for system in world.systems:
        system.apply_all_forces(t)
    for b in world.bodies:
        for fb in b.per_body_forces:
            fb.apply(b, t)
    # (plus globální a interakční síly...)

    # 4. KKT systém
    Minv, J, F, rhs_v = assemble_system(world.bodies, world.constraints, ...)
    a_MBD, lam = solve_kkt(Minv, J, F, rhs_v)

    # 5. Zápis vazbových sil do cache tělísek (pro logování)
    if len(J) > 0:
        F_constraint_all = J.T @ lam
        for i, b in enumerate(world.bodies):
            b.force_categories["constraint"] = F_constraint_all[6 * i : 6 * i + 3]

    # 6. Sestavení výsledných derivací pro SciPy
    body_derivs = extract_body_derivatives(world.bodies, a_MBD)
    return np.concatenate([body_derivs] + aux_derivs)
```

#### Zapojení do řešiče a loggeru

Díky jedné funkci `compute_snapshot` zmizí duplicitní kód.

```python
# 1. Využití pro SciPy (uvnitř integrate)
def rhs(t: float, y: np.ndarray) -> np.ndarray:
    return compute_snapshot(world, t, y)

sol = solve_ivp(rhs, ...)

# 2. Využití pro čisté logování (po skončení integrace)
if simulator.output is not None:
    for i, tk in enumerate(sol.t):
        # Tímto voláním se bezpečně naplní b.force_categories pro daný čas,
        # aniž by se zničila paměť modelů.
        compute_snapshot(world, float(tk), sol.y[:, i])

        # Teprve nyní logger zapíše řádek
        simulator.output.logger.log(world, tk)
```

---

## 3. Doporučený postup implementace

Aby se kód nerozbil, je nutné dodržet toto pořadí kroků:

1. **State Vector Move:** Nejprve proveď dynamické skládání stavů ($y$) —
   odstranění vnitřních `dt` iterací z padákových modelů a jejich přesun do
   vektoru $y$. Bez tohoto kroku nelze z modelů s vnitřním `dt` odstranit
   závislost na čase. Je to absolutní prerekvizita.
2. **Centralizace `compute_snapshot`:** Vytvoř tuto čistou funkci uvnitř (nebo
   vedle) `HybridIVPSolver.integrate`. Přesměruj do ní SciPy i po-výpočetní
   logování (odstranění duplicit a bezpečná replay smyčka).
3. **Rozdělení `World`:** Jako poslední krok abstrahuj `PhysicsWorld`,
   `OutputManager` a `Simulator`. Úklid architektury před zahájením Monte Carlo
   a ML experimentů dokončí transformaci kódu do plně testovatelného stavu.
