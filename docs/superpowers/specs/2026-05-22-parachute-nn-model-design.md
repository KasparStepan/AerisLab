# Návrh: Padákový model — NN backend, analytický baseline, slack tether

**Datum:** 2026-05-22
**Status:** Navrženo (čeká na review)
**Pod-projekt:** 3 ze 3 (Padákový model)
**Předpoklady:**
- Pod-projekt 1 (Jádro): `StateProvider`/`AuxDynamics`/`InertialProvider`,
  `RigidBody6DOF.set_added_mass`, `AddedMassFlux` síla, dynamický stavový vektor.
- Pod-projekt 2 (Orchestrace): `compute_snapshot` — od této chvíle **každá síla
  musí být čistá** (žádná mutace skryté historie v RHS). To je důvod, proč se staré
  stavové modely retirují právě zde.

---

## Kontext a motivace

Stávající `AdvancedParachute`
([parachute_models.py](../../../src/aerislab/models/aerodynamics/parachute_models.py))
obsahuje 6 inflačních modelů, které mutují skrytý stav uvnitř evaluace sil
(`air_mass_inside += dm_dt*dt` s natvrdo `dt=0.01`, konečné diference z
`prev_time`/`prev_velocity`). To je nekompatibilní s adaptivním řešičem i s
`compute_snapshot`. Cílem je nahradit je modelem postaveným na principech
pod-projektů 1–2:

1. **NN backend** (Neural ODE, ONNX) predikující `V̇_air` a `Cd`,
2. **jeden analytický baseline** se stejným rozhraním pro V&V,
3. **`SlackTether`** síla pro šňůry místo tuhé `DistanceConstraint`.

---

## Cíle a ne-cíle

**Cíle:**

- Jednotné rozhraní `ParachuteModel`, které je čisté (state-vektorové) a skládá se
  do `Parachute` komponenty přes kompozici.
- NN model (`V̇_air` + `Cd`) napojený na `set_added_mass` (LHS) a `AddedMassFlux`
  (`ṁ_added·v`, RHS) — jeden zdroj pravdy `V_air`.
- Analytický baseline implementující totéž rozhraní → swap NN ↔ baseline beze
  změny scénáře; baseline je referencí pro validaci NN.
- `SlackTether` (tah-only, progresivní tuhost + tlumení) pro padákové šňůry.
- Anizotropní tenzor přidané hmoty z geometrie vrchlíku (`V_air` → `m_axial`,
  `m_transverse`).

**Ne-cíle:**

- Trénink samotné sítě a generování FSI dat (mimo tento engine; spec definuje jen
  *rozhraní* a požadavky na síť).
- Vícetělesové sloshing/flexibilní vrchlík (lumped model stačí).

---

## Architektura

### Sekce A — Rozhraní `ParachuteModel`

Model je čistá komponenta, která drží stav `V_air` (a případně další) a vystavuje:

- `StateProvider`: `num_states`, `pack_state`, `unpack_state` (stav `V_air`, …).
- `AuxDynamics`: `compute_derivatives(t) → [V̇_air, …]` (z NN nebo analyticky).
- aerodynamickou sílu přes `Force` protokol (odpor z `Cd`, `q`, `A`),
- příspěvek do přidané hmoty: nastaví `body.set_added_mass(m_t, m_a)` a registruje
  `AddedMassFlux(mdot_func=…)` čtoucí `ṁ_added` z `V̇_air`.

```python
class ParachuteModel(StateProvider, AuxDynamics, Protocol):
    def drag_force(self) -> Force: ...          # Force objekt (Cd·q·A), čte aktuální stav
    def update_added_mass(self, body) -> None:  # set_added_mass z V_air + geometrie
    def added_mass_rate(self, t, body) -> float:# ṁ_added z V̇_air (pro AddedMassFlux)
```

**Klíčové:** `compute_derivatives` čte `V_air` ze stavu a vrací `V̇_air` —
**neakumuluje**, nemá `prev_time`. Tím je čistý a kompatibilní s `compute_snapshot`.

### Sekce B — NN backend (`NeuralParachute`)

```python
class NeuralParachute(ParachuteModel):
    # vstupy sítě: [v_body, rho, V_air]  (bezrozměrně / dynamický tlak — viz ISA)
    # výstupy sítě: [V̇_air, Cd]
    def compute_derivatives(self, t):
        vdot_air, cd = self._infer(...)   # ONNX inference
        self._cd_cache = cd               # cache pro drag_force
        return np.array([vdot_air])
```

- **Runtime:** `onnxruntime` (rychlá inference). Volitelná závislost (extra).
- **Vstupy:** lokální rychlost (`body.to_body(v)`), hustota
  (`atmosphere.density(altitude)`), `V_air`. Škálování na dynamický tlak / bezrozměrné
  veličiny (viz Gemini.MD 1.3), aby síť generalizovala přes výšku.
- **Výstupy:** `V̇_air` (derivace stavu) + `Cd` (koeficient pro `drag_force`).
- **Požadavek na síť (training-time, dokumentační):** hladké aktivace
  (GELU/Tanh/Swish), **zákaz ReLU/LeakyReLU** — stiff řešič potřebuje spojitou první
  derivaci pro Jacobián.
- **Optimalizace (do plánu):** dodat řešiči analytický Jacobián (`solve_ivp(jac=…)`)
  z autodiff sítě místo numerických diferencí skrz ONNX.

### Sekce C — Analytický baseline (`AnalyticParachute`)

Implementuje stejné rozhraní jako `NeuralParachute`, ale `V̇_air` a `Cd` počítá
z uzavřené formy (spojitá `tanh` inflace, převzatá z dnešního
`_force_continuous_inflation`, přepsaná jako derivace stavu místo akumulátoru).
Slouží jako:

- výchozí funkční model, dokud NN neexistuje,
- referenční řešení pro validaci NN (V&V),
- regresní kotva (deterministický, bez ONNX).

### Sekce D — Geometrie: `V_air` → plocha a anizotropní přidaná hmota

Skalární `V_air` → magnituda; tvar dodá geometrie vrchlíku:

- **Projekční plocha** `A(V_air)`: z objemu hemisférického vrchlíku
  `V ≈ k_vol·A^1.5/√π` → `A = f(V_air)` (pro `drag_force`).
- **Přidaná hmota (anizotropní):** `m_axial = k_axial·ρ·V_air`,
  `m_transverse = k_transverse·ρ·V_air` (`k_transverse ≈ 0.1–0.2·k_axial`, viz
  Gemini.MD 1.1). `update_added_mass` zavolá
  `body.set_added_mass(m_transverse, m_axial)` v lokálním rámu vrchlíku; rotaci do
  globálu řeší `set_added_mass`/matice hmotnosti z pod-projektu 1.
- **`ṁ_added`** = `k_axial·ρ·V̇_air` (resp. per-osa) → `added_mass_rate` pro
  `AddedMassFlux`.

### Sekce E — `SlackTether` (síla šňůr)

Nová síla v [forces.py](../../../src/aerislab/dynamics/forces.py), tah-only:

```python
class SlackTether(Force):  # mezi dvěma tělesy, jako Spring, ale jednostranná
    # L <= L0:  F = 0                      (lano je prověšené, netlačí)
    # L  > L0:  F = -(k·ΔL + k3·ΔL³)·d̂  -  c·v_rel_line·d̂
    #           ΔL = L - L0;  k3 = progresivní (nelineární) tuhost tkaniny
```

- aplikuje se v úchytových bodech (jako `Spring`) → generuje i momenty,
- strukturální tlumení `c` potlačuje vysokofrekvenční oscilace,
- nahrazuje `DistanceConstraint` v padákových scénářích; `DistanceConstraint`
  zůstává pro tuhé klouby. Tím mizí nefyzikální „snatch" impuls z diskrétního
  dosažení tuhé vazby (Gemini.MD 1.2).

### Sekce F — Kompozice do `Parachute` komponenty

`Parachute` ([components/parachute.py](../../../src/aerislab/components/parachute.py))
drží `body` + `ParachuteModel` (NN nebo analytický). Při registraci do světa:

- model se přidá jako `aux_provider` (StateProvider + AuxDynamics) → `V_air` je
  součástí globálního `y`,
- `model.drag_force()` se přidá jako per-body Force,
- `AddedMassFlux(model.added_mass_rate)` se přidá jako per-body Force,
- model každý krok zavolá `update_added_mass(body)` (přes `apply_all_forces` /
  snapshot, čistě — jen nastaví LHS hmotu z aktuálního `V_air`).

Staré `AdvancedParachute` a 6 inflačních modelů se odstraní (nebo přesunou do
`not_in_use/`), s ponecháním jednoho analytického baselinu (Sekce C).

---

## Datový tok (jeden krok `compute_snapshot`)

`unpack_global_state(y)` (naplní `V_air`) → `clear_forces` →
`model.update_added_mass(body)` (LHS `M_added` z `V_air`) → `apply_all_forces`
(drag z `Cd`,`A`; `AddedMassFlux` `ṁ·v`; SlackTether; gravitace) →
`assemble_system` (M_eff s added mass) → `solve_kkt` → `ydot`: tělesa z KKT,
`V̇_air` z `model.compute_derivatives(t)`. **Žádná mutace historie.**

---

## Testovací a V&V strategie

| Test | Co chrání |
|---|---|
| Model čistota | `compute_derivatives` 2× se stejným stavem → identické `V̇_air`, žádná mutace |
| Baseline vs analytika | `AnalyticParachute` `A(t)`/`V_air(t)` proti uzavřené formě |
| SlackTether tah-only | `L<=L0` → nulová síla; `L>L0` → tah dle zákona; spojitost v `L=L0` |
| Added-mass napojení | `update_added_mass` nastaví `set_added_mass` z `V_air`; `ṁ` z `V̇_air` |
| Opening-shock V&V | špička síly při otevření v realistickém rozsahu (proti referenci/FSI) |
| NN smoke (je-li ONNX) | inference vrátí `[V̇_air, Cd]` správného tvaru; bez ONNX se test přeskočí |
| Snatch eliminace | scénář se `SlackTether` nemá nefyzikální impuls jako s `DistanceConstraint` |

---

## Odsouhlasená rozhodnutí

- **NN výstupy:** `V̇_air` (derivace → AuxDynamics) + `Cd` (koeficient → Force);
  plocha a anizotropní tenzor se odvodí z `V_air` geometrií.
- **Šňůry:** přidat `SlackTether`, `DistanceConstraint` ponechat pro tuhé vazby;
  padákové scénáře přepnou na `SlackTether`.

## Otevřené body pro implementační plán (just-in-time)

- Přesný formát ONNX modelu (jména vstupů/výstupů) a balení `onnxruntime` jako extra.
- Hodnoty geometrických koeficientů `k_vol`, `k_axial`, `k_transverse` (z literatury/FSI).
- Konstitutivní konstanty `SlackTether` (`k`, `k3`, `c`) a jejich default.
- Zda dodávat analytický Jacobián NN do `solve_ivp(jac=…)` už v tomto pod-projektu,
  nebo jako follow-up optimalizaci.
- Migrace existujících padákových scénářů/examples na nový model + `SlackTether`.
