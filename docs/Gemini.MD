# Architektonický a Numerický Návrh: AerisLab ML/FSI Integrace

**Kontext:** Návrh refaktoringu a implementace pro integraci Neural ODE modelů nafukování padáků (trénovaných na FSI datech) do MBD (Multi-Body Dynamics) řešiče AerisLab.
**Cíl:** Fyzikálně a numericky korektní simulace padákových shozů, stabilní integrace pomocí adaptivních řešičů (Scipy Radau) a příprava robustního základu pro závěr doktorského studia.

---

## 1. Fyzikální modelování a Aerodynamika

### 1.1 Tenzor přidané hmotnosti (Added Mass Tensor)

Padák je *bluff body* (tupé těleso) a jeho přidaná hmota je vysoce anizotropní (směrově závislá). Modelování pomocí izotropní koule (skalární $m_{added}$) by zcela uměle utlumilo přirozené kyvadlové oscilace systému, což je kritický parametr padákových sestav.

* **Fyzikální realita:** V axiálním směru (osa Z padáku, směr letu) padák za sebou táhne obrovskou "bublinu" vzduchu (wake). Přidaná hmota je zde extrémní. V transverzálním směru (osy X a Y, pohyb do boku) vrchlík vzduchem klouže bokem a přidaná hmota je minimální (cca 10–20 % axiální).
* **Řešení v MBD:** ML model predikuje objem $V_{air}$ (nebo referenční axiální hmotu). V MBD řešiči se tento skalár převede na diagonální tenzor v **lokální souřadné soustavě vrchlíku**:

```python
# M_added v lokálním systému (Z je osa symetrie vrchlíku)
M_added_local = np.diag([m_transverse, m_transverse, m_axial])

# Převod do globálního systému pro KKT řešič pomocí rotační matice R
R = body.rotation_world()
M_added_world = R @ M_added_local @ R.T
```

* **Aplikace do matice hmotnosti:** Přidaná hmota se nesmí aplikovat jako post-hoc vnější síla, ale musí se přímo přičítat k matici hmotnosti tělesa ($M_{eff} = M_{body} + M_{added}$). O setrvačné síly vzniklé decelerací ($m_{added} \cdot \dot{v}$) se pak automaticky postará samotný řešič inverzí matice hmotnosti ($M_{eff}^{-1} \cdot F$). Tím získáš fyzikálně korektní opening shock i přesto, že trénovací FSI data byla generována při konstantní rychlosti.

### 1.2 Model závěsných šňůr (Zrušení DistanceConstraint)

Použití `DistanceConstraint` (holonomní tuhé vazby) pro spojení padáku a payloadu je pro simulaci shozu zásadní chyba. Když vzdálenost dosáhne délky šňůr, KKT řešič aplikuje v diskrétním čase nekonečný impuls, což vygeneruje nesmyslný pík rázové síly ("snatch force").

* **Řešení:** Šňůry musíš modelovat jako **Slack Tether** (volné lano) pomocí čisté síly (`Force`). Lano nepřenáší žádný tlak při stlačení ($L < L_0$), ale při natažení ($L > L_0$) aplikuje Hookův zákon s progresivní nelineární tuhostí (tkanina se natahuje) a strukturálním útlumem pro potlačení vysokofrekvenčních oscilací.

### 1.3 Atmosférický model (ISA)

Tvůj kód aktuálně používá konstantní hustotu vzduchu $\rho = 1.225 \, \text{kg/m}^3$. FSI simulace sice trénuješ při konstantní hustotě, ale reálné shozy mění výšku (např. otevírání ve 4000 m vs. dopad na úroveň moře), kde hustota klesá až o 40 %.

* **Řešení:** Zavést modul `models.atmosphere.ISA` (International Standard Atmosphere). Před odesláním dat do ML modelu se rychlost a hustota převedou na bezrozměrné veličiny nebo dynamický tlak ($q = 0.5 \cdot \rho \cdot v^2$), takže ML model správně škáluje predikce pro libovolnou výšku.

---

## 2. Architektura: "The State Vector Move"

Aby mohl Scipy Radau (stiff solver) korektně integrovat mechaniku těles i vnitřní stavy padáku současně, je nutné vyvázat řešič z hardcodovaných 13 stavů na těleso. Je potřeba zavést dynamické sestavování globálního stavového vektoru $y$. Všechny mutace vnitřních stavů v rámci explicitního kroku `dt` uvnitř RHS musí zmizet.

### 2.1 Rozhraní `StatefulComponent`

Vše, co se vyvíjí v čase (tuhé těleso, ML padák, ...), musí splňovat společný protokol komunikace:

```python
from typing import Protocol
import numpy as np


class StatefulComponent(Protocol):
    def get_num_states(self) -> int:
        """Vrátí počet stavových proměnných komponenty."""
        ...

    def pack_state(self) -> np.ndarray:
        """Sbalí aktuální vnitřní stavy do 1D pole."""
        ...

    def unpack_state(self, y: np.ndarray) -> None:
        """Zapíše stavy z 1D pole zpět do vnitřních proměnných."""
        ...

    def compute_derivatives(self, t: float) -> np.ndarray:
        """Vrátí časové derivace svých stavů (ydot)."""
        ...
```

### 2.2 Sestavení Neural ODE padáku

Padák si sám neintegruje nic, pouze poskytuje derivace do Scipy řešiče:

```python
class NeuralParachute(StatefulComponent):
    def __init__(self, ml_model):
        self.A = 0.001       # Aktuální průřez plocha [m²]
        self.A_side = 0.0    # Aktuální boční plocha [m²]
        self.V_air = 0.0     # Objem uzavřeného vzduchu [m³]
        self.ml_model = ml_model  # ONNX Session

    def get_num_states(self) -> int:
        return 3

    def pack_state(self) -> np.ndarray:
        return np.array([self.A, self.A_side, self.V_air])

    def unpack_state(self, y: np.ndarray) -> None:
        self.A, self.A_side, self.V_air = y[0], y[1], y[2]

    def compute_derivatives(self, t: float, v_rel: np.ndarray, rho: float) -> np.ndarray:
        # Volání rychlé ML inference v lokálním rámu padáku
        # Vstupy: [v_body, rho, A, A_side, V_air]
        # Výstupy ze sítě: [dA_dt, dA_side_dt, dV_air_dt, Cd]
        dA_dt, dA_side_dt, dV_air_dt, Cd = self.ml_model.predict(
            v_rel, rho, self.A, self.A_side, self.V_air
        )

        self.current_Cd = Cd  # Uložíme do cache pro výpočet vnější síly
        return np.array([dA_dt, dA_side_dt, dV_air_dt])
```

### 2.3 Matematicky čistá funkce `rhs` v Solveru

Díky tomuto rozhraní se funkce `rhs(t, y)` pro Scipy kompletně unifikuje. Celá fyzika (MBD + aerodynamika z ML) se řeší jako jeden plně provázaný systém diferenciálních rovnic:

```python
def rhs(t: float, y: np.ndarray) -> np.ndarray:
    # 1. Rozbalení globálního vektoru y ze Scipy do všech těles a ML komponent
    world.unpack_global_state(y)

    # 2. Vynulování silových akumulátorů tělísek
    for b in world.bodies:
        b.clear_forces()

    # 3. Výpočet derivací skrytých stavů z ML modelu a uložení Cd do cache
    aux_derivatives = []
    for comp in world.ml_components:
        v_rel = comp.body.to_body(comp.body.v)  # Transformace do lokálního rámu padáku
        rho = world.atmosphere.density(comp.body.p[2])
        aux_derivatives.append(comp.compute_derivatives(t, v_rel, rho))

    # 4. Aplikace fyzikálních sil (aerodynamický odpor z cache Cd, gravitace, šňůry...)
    world.apply_all_forces(t)

    # 5. Sestavení a vyřešení KKT systému (zahrnuje dynamicky updatovanou přidanou hmotu)
    Minv, J, F, rhs_v = assemble_system(world.bodies, world.constraints, ...)
    a_MBD, lam = solve_kkt(Minv, J, F, rhs_v)

    # 6. Extrakce časových derivací tělísek (dp/dt = v, dq/dt = qdot, dv/dt = a_lin, dw/dt = a_ang)
    body_derivatives = extract_body_derivatives(world.bodies, a_MBD)

    # 7. Složení finálního vektoru derivací ydot pro Scipy
    return np.concatenate([body_derivatives] + aux_derivatives)
```

---

## 3. Numerická optimalizace a KKT Řešič

### 3.1 Odstranění nespojitosti kvaternionů z RHS

Volání `b.q[:] = quat_normalize(b.q)` přímo uprostřed funkce `rhs` (v nynějším `solver.py`) vnáší do numeriky skokovou nespojitost. Když Radau zkouší dělat drobné perturbace pro výpočet Jacobiánu pomocí konečných diferencí, tato normalizace mu odhad zkorumpuje. Řešič pak drasticky zkracuje časový krok nebo kolabuje.

**Řešení:** Uvnitř `rhs` nechat kvaternion volně vyvíjet (integrovat surový $q_{dot}$). Normalizaci provádět zásadně až po kompletně akceptovaném kroku solveru, nebo čistěji přidat vazebnou podmínku $|q|^2 - 1 = 0$ jako další řádek holonomních vazeb přímo do KKT matice $J$.

### 3.2 Rychlá analytická inverze matice hmotnosti

Protože se přidaná hmota ($m_{added}$) mění spojitě v čase podle predikovaného $V_{air}$, matice hmotnosti $M$ se neustále mění a musí se přepočítávat v každém sub-kroku. Volání plné inverze $6 \times 6$ matice pomocí `np.linalg.inv()` je výpočetně nepřípustné.

**Řešení:** Využít blokově diagonální struktury $M$ a invertovat bloky analyticky, což je výpočetně okamžité:

```python
# Translační část: Pouhé převrácené číslo celkové hmoty (skalár)
inv_m = 1.0 / (body.mass + m_added_skalar)

# Rotační část: Inverze diagonální lokální matice a rotace do globálu
# I_total_local = I_body_base + I_added_from_FSI
I_inv_local = np.diag(1.0 / np.diag(I_total_local))
I_inv_world = R @ I_inv_local @ R.T
```

### 3.3 Hladké aktivace ML modelů (GELU / Tanh)

Stiff řešiče (Radau, BDF) bytostně vyžadují hladkost derivací.

**Pravidlo:** V architektuře tvé neuronové sítě (PyTorch) je přísně zakázáno použít aktivace typu ReLU nebo LeakyReLU. Tyto funkce mají nespojitou první derivaci v nule, což při výpočtu Jacobiánu vede k explozím nebo zamrzání řešiče. Používej výhradně hladké aktivace jako GELU, Swish nebo Tanh.

---

## 4. Cílová Roadmapa pro dokončení doktorátu

```text
 ┌──────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
 │   KROK 1: SLACK TETHER    │ ───> │ KROK 2: STATE VECTOR MOVE │ ───> │  KROK 3: ATMOSFÉRA (ISA)  │
 │ Výměna tuhé vazby šňůr    │      │ Abstrakce solveru od těl  │      │ Zavedení výškového        │
 │ za nelineární sílu v tahu │      │ do unifikovaného y vektoru│      │ profilu hustoty vzduchu   │
 └──────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘
                                                                                   │
                                                                                   ▼
 ┌──────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
 │  KROK 6: FINÁLNÍ VALIDACE │ <─── │  KROK 5: ONNX INTEGRACE   │ <─── │ KROK 4: TRÉNINK SURROGATE │
 │ Ověření rázových sil      │      │ Nasazení onnxruntime      │      │ Hladká síť trénovaná      │
 │ shozu proti referenčnímu  │      │ a lokálních transformací  │      │ na FSI (V_air, Cd)        │
 └──────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘
```
