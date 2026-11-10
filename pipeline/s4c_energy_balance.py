# -*- coding: utf-8 -*-
"""
Lake-surface energy balance: heat lost by the lake surface per unit area.

    Q_total = Q_evap + Q_sens + Q_rad − Q_solar_net        [W m-2]  (net surface heat loss)

A lake-temperature difference (K) is not directly comparable between lakes;
solving the surface energy balance converts it into a heat flux. At 40–60 °C
lake temperatures, radiation is only ~7–11% of the total heat loss
(Stevenson 1992 PhD Table 2.5, at 50 °C: evaporation 0.76 / sensible 0.16 /
radiation 0.08; Pailot-Bonnétat et al. 2025 field data: 0.58 / 0.31 / 0.11);
the dominant term is latent heat of evaporation.

Contents
--------
  · Physical coefficients, each with its primary source.
  · Pure flux functions (saturation vapour pressure, virtual temperature,
    `surface_fluxes`) following Terada & Hashimoto (2017) Appendix A, with
    ERA5-Land downward radiation.

The surface heat loss computed from lake surface temperature and meteorology is
close to observation; it is not a geothermal input, which would require
unobservable terms (rain, seepage, inflow enthalpy, storage).
analysis/validation_energy.py calls `surface_fluxes` to convert the event
contrasts into heat-flux scales.
"""

from __future__ import annotations


import numpy as np


import matplotlib
matplotlib.use("Agg")                       # batch run, no display
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ===========================================================================
# 1. Physical coefficients, each with its primary source
#
# These values set the absolute level of the fluxes. Each is given with its
# source equation and unit.
# ===========================================================================

# --- Evaporation (latent heat), Ryan, Harleman & Stolzenbach (1974) form -----
# Terada & Hashimoto (2017) JGR Solid Earth 122:6108-6130, doi:10.1002/2017JB014387
# Appendix A eq. (A2). Free and forced convection are added linearly (not as a vector sum).
LAMBDA0 = 0.027        # W m-2 Pa-1 K^(-1/3)  free-convection coefficient
#   Cross-check: Ryan & Harleman (1973) MIT Parsons Lab Report 161 eq. (2.35)
#   22.4 BTU ft-2 day-1 mmHg-1 (degF^-1/3) → SI 0.0269. Helfrich (1982) 0.026,
#   Pasternack & Varekamp (1997) 0.027. All four agree within rounding.
B0 = 0.032             # W m-2 Pa-1 (m/s)-1   forced-convection coefficient
#   Cross-check: Ryan & Harleman (1973) 14 → SI 0.0309. Helfrich 0.031, P&V 0.032.

# --- Sensible heat, Bowen ratio ---------------------------------------------
BOWEN_C = 61.0         # Pa K-1
#   Terada & Hashimoto (2017) eqs. (A6)(A7): Bo = C (T_s - T_a)/(e_s - e_a).
#   Matches Ryan & Harleman (1973) §2.5 C1 = 0.255 mmHg/degF = 61.2 Pa K-1,
#   and 0.61 mbar K-1 used by P&V (1997) and Pailot-Bonnétat (2025).

# --- Radiation ----------------------------------------------------------------
SIGMA_SB = 5.67e-8     # W m-2 K-4  Stefan-Boltzmann
#   Same in Aveni et al. (2025), TIRVolcH (2024), Terada & Hashimoto (2017), pumahu.
EPS_W = 0.97           # broadband emissivity of water (dimensionless)
#   Jimenez-Munoz & Sobrino (2006): 0.4-0.5 K per 0.01 in humid atmospheres (0.7-1.0 K dry).
ALBEDO_W = 0.10        # shortwave albedo of water (dimensionless)
#   Stevenson (1992) PhD thesis. Terada/pumahu/P&V have no albedo term (they
#   prescribe absorbed solar flux); with ERA5 downward shortwave an albedo is needed.

# --- Water properties ---------------------------------------------------------
RHO_W = 1000.0         # kg m-3      water density             (pumahu)
L_VAP = 2.38e6         # J kg-1      latent heat of vaporisation @50 degC  (Terada & Hashimoto 2017 A3)
R_V = 461.0            # J kg-1 K-1  gas constant of water vapour          (Terada & Hashimoto 2017 A3)
EPSILON_MW = 0.622     # -           Mw/Md, mixing-ratio definition (A5)

#   Wind speed over the lake is the largest single uncertainty of the fluxes.
#   Stevenson (1992) argues the log profile fails over hot lakes and recommends a
#   power law. Crater rims shelter the lake: P&V (1997) fixed 2.5 m/s, and a buoy
#   on Ruapehu Crater Lake measured 5 m/s (Hurst et al. 2012).

G_ACC, R_DRY = 9.80665, 287.05


# ===========================================================================
# 2. Physics: pure functions with no dependence on files or global state
#    (so they can be tested in isolation).
# ===========================================================================

def esat_magnus_pa(T_K):
    """Saturation vapour pressure [Pa]. Alduchov & Eskridge (1996) AERK."""
    Tc = np.asarray(T_K, dtype=float) - 273.15
    return 610.94 * np.exp(17.625 * Tc / (Tc + 243.04))


def esat_terada_pa(T_K):
    """Saturation vapour pressure [Pa]. Terada & Hashimoto (2017) eq. (A3).

    Constant-L Clausius-Clapeyron: exactly 611 Pa at 0 degC, increasingly low at
    higher temperature (−6.5% at 30 degC, −7.8% at 50 degC). Comparison only.
    """
    T = np.asarray(T_K, dtype=float)
    return 611.0 * np.exp((L_VAP / R_V) * (1.0 / 273.15 - 1.0 / T))


def esat_pa(T_K, model: str = "magnus"):
    return esat_terada_pa(T_K) if model == "terada_a3" else esat_magnus_pa(T_K)


def virtual_temperature(T_K, e_Pa, P_Pa):
    """Virtual temperature T_v [K]. Terada & Hashimoto (2017) eqs. (A4)(A5).

    Free convection is driven by density, not temperature, differences. Water
    vapour is lighter than dry air, so over a 40-60 degC lake surface ignoring
    the vapour contribution strongly underestimates free convection.
    """
    e = np.clip(np.asarray(e_Pa, float), 0.0, None)
    P = np.asarray(P_Pa, float)
    w = EPSILON_MW * e / np.maximum(P - e, 1.0)          # mixing ratio (A5)
    return np.asarray(T_K, float) * (1.0 + w / EPSILON_MW) / (1.0 + w)


def surface_fluxes(T_s_K, Ta_K, RH, Psfc_Pa, u2_ms, SWdown_Wm2, LWdown_Wm2,
                   eps_w=EPS_W, albedo=ALBEDO_W, lam0=LAMBDA0, b0=B0,
                   bowen_C=BOWEN_C, esat_model="magnus"):
    """Heat fluxes per unit lake area [W m-2]. **Sign: positive = heat lost by the lake.**

    Follows Terada & Hashimoto (2017) Appendix A, with the radiative terms
    taken from ERA5-Land downward radiation (the original uses a constant
    R_sun = 100 W/m2 and a Linacre (1992) parameterisation of downward longwave).

        phi_e = [ lam0 (T_vs - T_va)^(1/3) + b0 u2 ] (e_s - e_a)          (A2)
        phi_c = Bo * phi_e ,  Bo = C (T_s - T_a)/(e_s - e_a)              (A6,A7)
              = C (T_s - T_a) [ lam0 (dTv)^(1/3) + b0 u2 ]   ← (e_s-e_a) cancels;
                computing it in this form avoids 0/0 when e_s ≈ e_a.
        phi_r = eps sigma T_s^4 - eps LW↓        (net longwave loss)
        phi_sw= (1-alpha) SW↓                    (absorbed shortwave, a gain)

    Returns a dict of ndarrays.
    """
    T_s = np.asarray(T_s_K, float)
    T_a = np.asarray(Ta_K, float)
    P = np.asarray(Psfc_Pa, float)

    # Vapour pressures. e_a = RH·e_sat(Ta) with the **same formula** as e_s, so
    # that (e_s - e_a) is not contaminated by formula differences.
    e_s = esat_pa(T_s, esat_model)
    e_a = np.clip(np.asarray(RH, float), 0.0, 1.0) * esat_pa(T_a, esat_model)
    de = e_s - e_a

    # Free-convection driver. With a stable layer (air above the surface
    # denser) there is no free convection, so clip at 0 (also keeps the cube
    # root real).
    T_vs = virtual_temperature(T_s, e_s, P)
    T_va = virtual_temperature(T_a, e_a, P)
    dTv = np.maximum(T_vs - T_va, 0.0)

    h_free = lam0 * np.cbrt(dTv)                 # W m-2 Pa-1
    h_forced = b0 * np.maximum(np.asarray(u2_ms, float), 0.0)
    h_tot = h_free + h_forced

    q_evap = h_tot * de                          # de<0 means condensation; sign kept
    q_sens = bowen_C * (T_s - T_a) * h_tot
    with np.errstate(divide="ignore", invalid="ignore"):
        bowen = np.where(np.abs(de) > 1e-6, bowen_C * (T_s - T_a) / de, np.nan)

    eps = np.asarray(eps_w, float)
    q_rad = eps * SIGMA_SB * T_s ** 4 - eps * np.asarray(LWdown_Wm2, float)
    q_solar = (1.0 - albedo) * np.asarray(SWdown_Wm2, float)

    q_total = q_evap + q_sens + q_rad - q_solar

    return dict(Q_evap_Wm2=q_evap, Q_sens_Wm2=q_sens, Q_rad_Wm2=q_rad,
                Q_solar_net_Wm2=q_solar, Q_total_Wm2=q_total,
                bowen_ratio=bowen, e_s_Pa=e_s, e_a_Pa=e_a,
                T_vs_K=T_vs, T_va_K=T_va)


AREA_CLIP_LO, AREA_CLIP_HI = 0.25, 2.5   # allowed range of lake area relative to the AOI polygon area


