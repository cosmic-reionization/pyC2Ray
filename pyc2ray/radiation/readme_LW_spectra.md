# Hardcoded Stellar Spectrum Parameters in pyc2ray LW Module
## Analysis, Physical Relationships, and Automation Strategy

---

## 1. Complete Inventory of Hardcoded Values

When you change the assumed blackbody temperature and/or Pop III stellar mass, the following values — spread across three files — all need to be updated consistently. They fall into five logical groups.

---

### Group A: BB temperature inputs (in `run_LW_2BB.py`)

These are the most visible inputs, and the ones you change first — but they are *not* the only ones.

| Location | Variable | Current value | Physical meaning |
|---|---|---|---|
| `run_LW_2BB.py` L~252 | `bb_Teff` (minihalo call) | `10**(4.850)` ≈ 70,795 K | Effective BB temperature used to build the H-ionisation cross-section weighting table for **subgrid (Pop III/minihalo) sources** |
| `run_LW_2BB.py` L~255 | `bb_Teff` (ACH call) | `1e5` = 100,000 K | Same, for **atomic-cooling halo (ACH) sources** |

These feed into `update_tables()` → `BlackBodySource.make_photo_table()`, which controls how the ray-tracer weights the ionising photon cross-section across frequency. **Changing these alone is insufficient** — the LW emissivity and ionising photon rate constants (below) also depend on the assumed spectrum and must match.

---

### Group B: LW emissivities (in `c2ray_cubep3m_LW_2spectra.py __init__`)

```python
self.emis00  = 1.67e21   # erg s⁻¹ Hz⁻¹ M☉⁻¹  — HMACH sources
self.emis01  = 3e21      # erg s⁻¹ Hz⁻¹ M☉⁻¹  — LMACH sources
self.emissub = 3e21      # erg s⁻¹ Hz⁻¹ M☉⁻¹  — subgrid/minihalo sources
```

**Physical meaning:** The mean specific luminosity *in the Lyman-Werner band* (11.2–13.6 eV, λ = 912–1107 Å) per unit stellar mass. Used in `get_srclumK()` to build the 3-D source emissivity field that feeds the LW Green's function convolution.

**How they enter the calculation:**
```python
coeff00 = emis00 × CC00 × fstar[0] × (Ω_b/Ω_m)
srclum[i,j,k] += M_halo_msun[i] × coeff00          # → erg/s/Hz per cell
```

**Dependence on T_eff:** These *are* derivable from T_eff, given `QH_M_real` (see Section 2.1). The relationship is:

```
emis_LW = QH_M_real × h × <B_ν(T)>_LW / ∫_{ν_ion}^{∞} B_ν(T)/ν dν
```

where `<B_ν>_LW` is the mean Planck function over the LW band and the denominator is the ionising photon kernel. The stellar radius cancels, making this a **pure ratio of BB integrals** — fully computable at runtime.

Numerically, for the current hardcoded temperatures:

| Source | T_eff (K) | QH_M_real | emis (hardcoded) | emis (BB formula) | Ratio |
|---|---|---|---|---|---|
| HMACH (00) | 70,795 | 6.31×10⁴⁶ | 1.67×10²¹ | 4.1×10²⁰ | 4.0× |
| LMACH (01) | 100,000 | 1.20×10⁴⁸ | 3.00×10²¹ | 4.4×10²¹ | 0.69× |
| Sub/MH | 100,000 | 1.20×10⁴⁸ | 3.00×10²¹ | 4.4×10²¹ | 0.69× |

The LMACH/Sub values are ~30% from the pure BB formula; the HMACH is 4× off. This means the hardcoded values were **not taken directly from a pure BB** — they likely originate from stellar population synthesis models (e.g., Leitherer et al. 1999/Starburst99 for Pop II ACHs, Schaerer 2002 for Pop III). However, the BB formula is a good first-order estimate and, crucially, gives the **correct scaling direction and magnitude** when T_eff changes.

---

### Group C: Real ionising photon rates (in `c2ray_cubep3m_LW_2spectra.py __init__`)

```python
self.QH_M_real00       = 6.309573445e46   # photons s⁻¹ M☉⁻¹  — HMACH
self.QH_M_real01       = 1.2e48           # photons s⁻¹ M☉⁻¹  — LMACH
self.QH_M_real_sub     = (from YAML)      # photons s⁻¹ M☉⁻¹  — subgrid (same formula)
```

**Physical meaning:** The ionising photon production rate *per solar mass of stars*. This is the "real" (astrophysically motivated) normalization for each source population, as opposed to the C2Ray simulation-unit rate computed from `Ni`.

**How they enter the calculation:** Used in `get_srclumK()` via the correction coefficient:

```python
CC = QH_M_C2ray / QH_M_real   # dimensionless correction factor
coeff = emis × CC × fstar × (Ω_b/Ω_m)
```

`CC` bridges the C2Ray photon-counting scheme (which works in units of `Ni × M☉/m_p / Δt`) to the physical ionising photon rate. If `emis` and `QH_M_real` are both updated consistently, `CC` takes care of the normalisation automatically.

**Dependence on T_eff:** `QH_M_real` **cannot** be derived from T_eff alone — you also need either the stellar radius `R_*` or the total luminosity `L_bol`, which requires a mass–luminosity relation. In practice:

- **Pop III (subgrid):** Given `M_PIIIstar_msun`, use Schaerer (2002) Table 3 (zero metallicity) to look up Q_H and then divide by `M_PIIIstar_msun`.
- **Pop II (ACHs):** Integrate Q_H over a Salpeter IMF, or use a Starburst99 table value for the assumed metallicity.

Typical literature values for reference:

| Stellar type | T_eff range | QH_M_real (phot/s/M☉) | Source |
|---|---|---|---|
| Pop II, low-Z | 40,000–60,000 K | ~10⁴⁶–10⁴⁷ | Leitherer+ 1999 |
| Pop II, near-zero Z | ~70,000 K | ~6×10⁴⁶ | matches `QH_M_real00` |
| Pop III, 25 M☉ | ~90,000 K | ~5×10⁴⁷ | Schaerer 2002 |
| Pop III, 100 M☉ | ~100,000 K | ~1.5×10⁴⁸ | Schaerer 2002 |
| Pop III, 500 M☉ | ~107,000 K | ~3×10⁴⁸ | Schaerer 2002 |

---

### Group D: Ionising photon budget per proton `Ni` (in `c2ray_cubep3m_LW_2spectra.py __init__`)

```python
self.Ni = np.array([ 6000/6,   50000/6,   self.Ni_MH ])
                  # = 1000      = 8333      (from YAML)
```

**Physical meaning:** The number of ionising photons produced per hydrogen atom (proton) in the stellar mass, integrated over the C2Ray timestep. Used in `get_srclumK()` to compute `QH_M_C2ray`:

```python
QH_M_C2ray = Ni × (M☉/m_p) / C2ray_lifetime   # [photons/s/M☉]
CC = QH_M_C2ray / QH_M_real
```

**The key identity:**

```
Ni = QH_M_real × (m_p / M☉) × t_star
```

where `t_star` is the effective stellar lifetime over which the ionising budget is emitted. Inverting:

```
t_star = Ni × M☉ / (QH_M_real × m_p)
```

For current values:
- HMACH: Ni=1000, t_star ≈ **0.60 Myr**
- LMACH: Ni=8333, t_star ≈ **0.26 Myr**

These are physically reasonable main-sequence lifetimes for massive stars. The important implication: **if you change T_eff and therefore QH_M_real, you must also update Ni** to keep t_star physically consistent (or explicitly re-set t_star and let the code derive Ni).

**Dependence on T_eff:** Ni is derivable from T_eff if you additionally know `t_star` (stellar lifetime, itself a function of stellar mass from stellar evolution models).

---

### Group E: Astrophysical parameters that are NOT derivable from T_eff

These must remain as explicit user inputs regardless of automation, because they encode astrophysical model choices independent of the stellar SED:

```python
self.phot_per_atom = np.array([10/6,  150/6,   self.phot_per_atom_MH])
#                             = 1.67   = 25      (from YAML)
self.fstar         = np.array([0.008,  0.015,   self.fstar_MH])
#                             HMACH   LMACH      (from YAML)
```

**`fstar` (star formation efficiency):** Fraction of halo baryons converted to stars. Set by ISM physics / feedback, not the SED.

**`phot_per_atom` (photons per baryon):** This is `Ni × f_esc` — the *escaping* ionising photons per baryon. It encodes the poorly-constrained escape fraction `f_esc`, which is an independent model choice:

```
phot_per_atom = Ni × f_esc
f_esc = phot_per_atom / Ni = (10/6) / 1000 ≈ 0.0017    [HMACH]
                           = (150/6) / 8333 ≈ 0.003     [LMACH]
```

**From YAML (always user-set):**

| YAML key | Physical meaning | Cannot be computed from |
|---|---|---|
| `M_PIIIstar_msun` | Characteristic Pop III stellar mass | T_eff alone (this is the PRIMARY input for subgrid sources) |
| `Ni_MH` | Ionising photons per proton for minihalos | `QH_M_real_sub` + `t_star` needed |
| `QH_M_real_MH` | Ionising photon rate for minihalos | Schaerer (2002) table + `M_PIIIstar_msun` needed |
| `phot_per_atom_MH` | Photons per atom for minihalos | Encodes f_esc — astrophysical input |
| `fstar_MH` | Star formation efficiency for minihalos | Astrophysical input |

---

## 2. Dependency Graph

```
                    ┌──────────────────────────────┐
USER INPUTS  ──────▶│  T_eff  │  M_PIIIstar  │ IMF │
                    └──────┬───────────┬────────────┘
                           │           │
                    ┌──────▼─────┐  ┌──▼──────────────────┐
                    │ BB integrals│  │Schaerer/SB99 table  │
                    │(analytically│  │(lookup or fitting   │
                    │ computable) │  │ formula)            │
                    └──────┬──────┘  └──────┬──────────────┘
                           │                │
           ┌───────────────▼──┐    ┌────────▼───────┐
           │ LW/Q_H ratio     │    │ QH_M_real      │  ← also needed for emis
           │ (pure BB ratio)  │    │[phot/s/M☉]     │
           └──────────────────┘    └────────┬───────┘
                   │                        │
           ┌───────▼────────────────────────▼───────┐
           │   emis_LW = QH_M_real × h × LW/Q_H     │  COMPUTABLE
           │   [erg/s/Hz/M☉]                         │
           └────────────────────────────────────────┘
                                    │
                               ┌────▼──────────────┐
                               │ t_star (from M_*)  │  stellar evo models
                               └────┬──────────────┘
                                    │
                           ┌────────▼──────────────┐
                           │ Ni = QH_M_real × (m_p/M☉) × t_star  │  COMPUTABLE
                           └──────────────────────────────────────┘

USER MUST ALWAYS PROVIDE:  fstar,  phot_per_atom (encodes f_esc)