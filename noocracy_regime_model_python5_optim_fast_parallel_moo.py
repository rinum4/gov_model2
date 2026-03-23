from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable, Tuple
import math
import random

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


import numpy as np

try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.termination import get_termination
    from pymoo.optimize import minimize
except Exception:  # pragma: no cover
    ElementwiseProblem = None
    GA = None
    FloatRandomSampling = None
    SBX = None
    PM = None
    get_termination = None
    minimize = None


import numpy as np
from multiprocessing import Pool

from pymoo.parallelization.starmap import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from pymoo.algorithms.moo.nsga2 import NSGA2

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return a / (b if abs(b) > eps else eps)


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

@dataclass
class Params:
    # Simulation
    dt: float = 0.25
    final_time: float = 100.0

    # Core constants / stabilizers
    eps: float = 0.001
    A12: float = 1.0
    H12: float = 0.7
    Q12: float = 0.8
    EH12: float = 0.1
    ER12: float = 0.1
    # ER12: float = 0.08
    L12: float = 0.15
    P12: float = 0.8
    T12: float = 0.35
    G12: float = 0.35
    SEC0: float = 0.11 
    SEC12: float = 0.08
    Ineq12: float = 3.0
    cb12: float = 0.8
    fc12: float = 0.35
    fd12: float = 0.3
    F12: float = 0.2
    Ed12: float = 0.08
    Eag12: float = 0.08
    ff12: float = 0.35

    # Trust weights
    a1: float = 0.2
    a2: float = 0.15
    a3: float = 0.15
    a4: float = 0.25
    a5: float = 0.15
    a6: float = 0.1

    # Conflict
    b1: float = 0.4
    b2: float = 0.45
    b3: float = 0.25

    # Production exponents / energy
    alphaA: float = 1.0
    alphaR: float = 1.1
    beta: float = 1.05
    phi: float = 0.7
    psi: float = 0.6
    chi: float = 1.5 / 2.0
    lambda_: float = 0.6
    nu: float = 0.9

    # Capital share decomposition
    aplha0: float = 0.35
    aplhaA: float = 0.15
    aplhaH: float = 0.2
    aplhaQ: float = 0.1

    # Birth / death
    b0: float = 0.02
    xi_b: float = 1.8
    d0: float = 0.01
    delta_f: float = 1.2
    delta_E: float = 0.6
    delta_C: float = 0.8

    # Dynamic rates
    dC: float = 0.04
    dG: float = 0.022
    dT: float = 0.025
    rC: float = 0.03
    rG: float = 0.065
    rT: float = 0.025
    r_soil: float = 0.01
    d_soil: float = 0.02 

    # Resource / EROI
    e0: float = 0.22
    eP: float = 0.08
    eY: float = 0.1 / 2.0
    eta0: float = 12.0
    eta_min: float = 0.05
    q_min: float = 0.05
    gamma: float = 0.65
    kR: float = 0.1
    R0: float = 3 

    # Energy efficiency augmentation
    weA: float = 0.35
    weH: float = 0.25
    wEQ: float = 0.40
    s_eta: float = 0.2
    # s_eta: float = 0.0

    # Maintenance and security costs
    cA: float = 0.03
    cH: float = 0.05
    cQ: float = 0.02
    csec: float = 0.02

    # Energy-tech coefficients
    epsA: float = 0.015
    epsH: float = 0.02
    epsQ: float = 0.008

    # Capital multipliers
    kappaA: float = 0.35
    kappaH: float = 0.6
    kappaE: float = 1.0
    kappaP: float = 1.0
    kappa_pA: float = 0.65
    kappa_pH: float = 1.5
    kappaC: float = 0.8

    # Land / solar
    L0: float = 0.2
    L_other: float = 0.05
    gamma_sol: float = 0.03
    tau_sol: float = 40.0

    # Social / control
    l1: float = 0.7
    l2: float = 0.4
    l3: float = 0.6
    l4: float = 0.2
    omegaG: float = 0.5
    omegaT: float = 0.4
    f_stab_min: float = 0.35
    Tmax: float = 1.0
    Gmax: float = 1.0

    # Employment / automation
    kA: float = 0.9
    kH: float = 2.2
    kQ: float = 1.6
    s0A: float = 0.35
    s0W: float = 0.6
    sK: float = 0.01

    # Budget / taxation / spending
    # thetaA: float = 0.4
    # thetaH: float = 0.3
    # thetaQ: float = 0.3
    
    # Misc production / agriculture
    p0: float = 0.03
    ro_fert: float = 0.2
    Yld0: float = 1 * 50
    waste: float = 0.2
    yyA: float = 0.45
    yyR: float = 1.0

    # Welfare / elite utility
    wy: float = 0.45
    ws: float = 0.2
    wc: float = 0.1
    wsec: float = 0.05
    wr: float = 0.2
    ww_cm: float = 0.35
    ww_f: float = 0.3
    ww_t: float = 0.2
    ww_c: float = 0.15
    ww_cr: float = 0.25

    # Norm constants
    cm0: float = 0.268     # 0.147
    food_pc0: float = 1.39 # 0.53
    yy_elit0: float = 3.72 # 6.06 
    
    c_star: float = 1.0

    # Dividend / enclosure
    gamma_priv: float = 0.0

    # Post-employment / absorptive sector
    mmin: float = 0.3
    
    # State decay of tech capital
    tauA: float = 28.0
    tauH: float = 16.0
    tauQ: float = 12.0
    
    # Private elite reinvestment
    theta_priv_inv: float = 0.2 #!
    theta_priv_A: float = 0.8  # 0.45
    theta_priv_H: float = 0.1  # 0.25
    theta_priv_Q: float = 0.1  # 0.30
    
    # regime spec!
    m0: float = 0.55
    mu_sec: float = 0.2
    muU: float = 1.0
    mmax: float = 0.95
    
    # Energy shares
    sE_ag: float = 0.36
    sE_R: float = 0.38
    sE_sec: float = 0.06
    sE_house: float = 0.2
    
    taxK_rate: float = 0.55
    taxW_rate: float = 0.6
    
    thetaU: float = 0.35    
    thetaSVC: float = 0.49
    theta_sec: float = 0.06
    thta_inv: float = 0.1

@dataclass
class State:
    A: float = 0.6
    A_pop: float = 0.12 #+
    Bg: float = 0.215     # 0.1
    C: float = 0.3
    Fs: float = 0.85
    G: float = 0.65
    H: float = 0.03
    L_arb: float = 1.0
    L_sol: float = 0.05
    P: float = 1.0
    Ph: float = 2.0
    Q: float = 0.03        # 0.1
    R: float = 3
    T: float = 0.6
    U_elit: float = 0.0
    U_pop: float = 0.0

# ------------------------------------------------------------
# Core model
# ------------------------------------------------------------

class RegimeModel:
    def __init__(self, params: Params, state: Optional[State] = None):
        self.p = params
        self.s = state or State(R=params.R0)

    def aux(self) -> Dict[str, float]:
        p = self.p
        s = self.s

        # Bounded stocks
        Ceff = clamp(s.C, 0.0, 1.0)
        Geff = clamp(s.G, 0.0, 1.0)
        Teff = clamp(s.T, 0.0, 1.0)
        Peff = max(p.eps, s.P)
        
        A_tot = s.A + s.A_pop #+

        # Technology / capital share
        aplha = (
            p.aplha0
            # + p.aplhaA * safe_div(s.A, s.A + p.A12, p.eps) #-
            + p.aplhaA * safe_div(A_tot, A_tot + p.A12, p.eps) #+
            + p.aplhaH * safe_div(s.H, s.H + p.H12, p.eps)
            + p.aplhaQ * safe_div(s.Q, s.Q + p.Q12, p.eps)
        )

        # Employment structure
        s_task_A = math.exp(-p.kQ * s.Q)
        # s_task_R = math.exp(-p.kA * s.A - p.kH * s.H)
        s_task_R = math.exp(-p.kA * A_tot - p.kH * s.H) #+
        PK = p.sK * s.P
        PA = p.s0A * s.P * s_task_A
        PW = p.s0W * s.P * s_task_R
        PU = max(0.0, s.P - PK - PW - PA)
        U = safe_div(PU, max(s.P, p.eps), p.eps)

        # Resource quality and efficiency
        R_frac = max(0.0, safe_div(s.R, p.R0, p.eps))
        qR = p.q_min + (1.0 - p.q_min) * (R_frac ** p.gamma)
        Eff_gain = (
            # p.weA * safe_div(s.A, s.A + p.A12, p.eps)   #-
            p.weA * safe_div(A_tot, A_tot + p.A12, p.eps) #+
            + p.weH * safe_div(s.H, s.H + p.H12, p.eps)
            + p.wEQ * safe_div(s.Q, s.Q + p.Q12, p.eps)
        )
        eta = max(1.0 + 1e-6, p.eta_min + (p.eta0 - p.eta_min) * (qR ** p.beta) * (1.0 + p.s_eta * Eff_gain))

        # Energy system
        # KE = 1.0 + p.kappaA * s.A + p.kappaH * s.H #-
        KE = 1.0 + p.kappaA * A_tot + p.kappaH * s.H #+ 
        Eg = p.e0 * (KE ** p.phi) * qR
        Enet = Eg * (1.0 - 1.0 / eta)
        # EA = p.epsA * s.A #-
        EA = p.epsA * A_tot #+  
        EH = p.epsH * s.H
        EQ = p.epsQ * s.Q
        Etech = EA + EH + EQ
        Eavail = max(0.0, Enet - Etech)
        Eag = p.sE_ag * Eavail
        Ehouse = p.sE_house * Eavail
        ER = p.sE_R * Eavail
        Esec = p.sE_sec * Eavail
        Eneed = p.eP * s.P + p.eY * (0.0)  # temporary, updated after Y

        # Agriculture / land
        # Kp = 1.0 + p.kappa_pA * s.A + p.kappa_pH * s.H  #-
        Kp = 1.0 + p.kappa_pA * A_tot + p.kappa_pH * s.H  #+ 
        Pext = p.p0 * (Kp ** p.psi) * safe_div(s.Ph, s.Ph + p.P12, p.eps)
        Efert = p.ro_fert * Eag
        Fprod = min(p.kappaE * Efert, p.kappaP * Pext)
        fEag = safe_div(Eag, Eag + p.Eag12, p.eps) ** p.chi
        ffert = safe_div(Fprod, Fprod + p.F12, p.eps)
        Yld = p.Yld0 * fEag * ffert * s.Fs
        Lfood = max(0.0, s.L_arb - s.L_sol - p.L_other)
        Food = Lfood * Yld
        food_pc = Food * (1.0 - p.waste) / Peff
        food_pc_norm = safe_div(food_pc, p.food_pc0, p.eps)

        # Social / governance stabilization
        Ineq = None  # set after income block
        Ineq_eff = None
        Gap = None
        m = clamp(p.m0 + p.muU * U, p.mmin, p.mmax)
        f_stab = None
        Lctrl = None
        CollapseRisk = None

        # Production / incomes (depends on food and stabilization)
        # temporary placeholders for first pass using current T/G/C only
        f_stab = p.f_stab_min + (1.0 - p.f_stab_min) * (
            (safe_div(Teff, Teff + p.T12, p.eps) ** p.omegaT)
            * (safe_div(Geff, Geff + p.G12, p.eps) ** p.omegaG)
            * (1.0 / (1.0 + p.kappaC * Ceff))
        )
        YA = p.yyA * m * (safe_div(Ehouse, Ehouse + p.EH12, p.eps) ** p.alphaA) * (1.0 / (1.0 + p.kappaC * Ceff))
        YR = p.yyR * (safe_div(ER, ER + p.ER12, p.eps) ** p.alphaR) * (safe_div(Lfood, Lfood + p.L12, p.eps) ** p.lambda_) * f_stab
        Y = YR + YA
        Eneed = p.eP * s.P + p.eY * Y

        # Budget / income / inequality
        Cost_maint = p.cA * s.A + p.cH * s.H + p.cQ * s.Q
        # Cost_sec = p.csec * Ceff #нужно только в режиме fortless
        Cost_sec = 0.0
        kA_priv = safe_div(s.A, A_tot, p.eps) #+
        Cost_maint_pop = p.cA * s.A_pop
        # Rent = max(0.0, aplha * Y - Cost_maint - Cost_sec)         #-
        Rent = max(0.0, aplha * Y * kA_priv - Cost_maint - Cost_sec) #+
        Rent_pop = max(0.0, aplha * Y * (1.0 - kA_priv) - Cost_maint_pop) #+
        
        Div_priv = max(0.0, p.gamma_priv * Rent)
        TaxK = p.taxK_rate * Rent
        wagePool = (1.0 - aplha) * Y
        TaxW = p.taxW_rate * wagePool
        wagePool_net = max(0.0, wagePool - TaxW)
        Bg_plus = max(0.0, s.Bg)
        Itot = p.thta_inv * Bg_plus
        SEC = p.theta_sec * Bg_plus
        SVC = p.thetaSVC * Bg_plus
        TU = p.thetaU * Bg_plus
        
        # private reinvestment loop
        # Rent_net = Rent - TaxK - Div_priv
        Rent_net_pre_priv = Rent - TaxK - Div_priv
        I_priv = p.theta_priv_inv * max(0.0, Rent_net_pre_priv)
        I_priv_A = p.theta_priv_A * I_priv
        I_priv_H = p.theta_priv_H * I_priv
        I_priv_Q = p.theta_priv_Q * I_priv
        Rent_net = Rent_net_pre_priv - I_priv
        
        cm = min((max(0.0, wagePool_net) + TU + SVC + Div_priv) / Peff, 1.0)
        cm_norm = safe_div(cm, p.cm0, p.eps)
        Gap = max(0.0, (p.c_star - cm) / p.c_star)
        yyK = Rent_net / (PK + p.eps)
        yyW = wagePool_net / (PW + PA + p.eps)
        yy_elit = Rent_net / (PK + p.eps)
        yy_elit_norm = safe_div(yy_elit, p.yy_elit0, p.eps)
        Ineq = yyK / (yyW + p.eps)
        Ineq_eff = Ineq / (Ineq + p.Ineq12)

        # Updated higher-level controls
        Lctrl = p.L0 + p.l1 * Gap + p.l2 * Ineq_eff + p.l3 * U + p.l4 * m
        # CollapseRisk = 0.4 * 0.5 / (food_pc + 0.5) + 0.4 * 0.4 / (s.G + 0.4) + 0.2 * s.C / (s.C + 0.35)
        CollapseRisk = 0.4 * 0.5 / (food_pc_norm + 0.5) + 0.4 * 0.4 / (s.G + 0.4) + 0.2 * s.C / (s.C + 0.35) # dim correction
        SEC_core = SEC / (SEC + p.SEC12)
        SEC_norm = safe_div(SEC , p.SEC0, p.eps) # dim correction
        b_cm = p.b0 * (1.0 / (1.0 + (min(max(0.0, cm) / p.cb12, 1000.0) ** p.xi_b)))
        d = p.d0 * (1.0 + p.delta_f * p.fd12 / (food_pc + p.fd12) + p.delta_E * p.Ed12 / (Ehouse + p.Ed12) + p.delta_C * Ceff)
        Birth = b_cm * s.P
        Deaths = d * s.P

        welfare_flow = p.ww_cm * cm_norm + p.ww_f * food_pc_norm + p.ww_t * s.T + p.ww_c * (1.0 - s.C) - p.ww_cr * CollapseRisk
        # elite_flow = p.wy * yy_elit_norm + p.ws * SEC_core - p.wc * s.C - p.wsec * SEC - p.wr * CollapseRisk
        elite_flow = p.wy * yy_elit_norm + p.ws * SEC_core - p.wc * s.C - p.wsec * SEC_norm - p.wr * CollapseRisk # dim correction

        return {
            "A": s.A,
            "A_pop": s.A_pop, #+
            "A_tot": A_tot,   #+
            "Bg": s.Bg,
            "Ceff": Ceff,
            "Geff": Geff,
            "Teff": Teff,
            "Peff": Peff,
            "aplha": aplha,
            "PK": PK,
            "PA": PA,
            "PW": PW,
            "PU": PU,
            "U": U,
            "R_frac": R_frac,
            "qR": qR,
            "Eff_gain": Eff_gain,
            "eta": eta,
            "KE": KE,
            "Eg": Eg,
            "Enet": Enet,
            "EA": EA,
            "EH": EH,
            "EQ": EQ,
            "Etech": Etech,
            "Eavail": Eavail,
            "Eag": Eag,
            "Ehouse": Ehouse,
            "ER": ER,
            "Esec": Esec,
            "Eneed": Eneed,
            "Kp": Kp,
            "Pext": Pext,
            "Efert": Efert,
            "Fprod": Fprod,
            "fEag": fEag,
            "ffert": ffert,
            "Yld": Yld,
            "Lfood": Lfood,
            "Food": Food,
            "food_pc": food_pc,
            "food_pc_norm": food_pc_norm,
            "m": m,
            "f_stab": f_stab,
            "YA": YA,
            "YR": YR,
            "Y": Y,
            "kA_priv": kA_priv, #+
            "Cost_maint": Cost_maint,
            "Cost_maint_pop": Cost_maint_pop, #+
            "Cost_sec": Cost_sec,
            "Rent": Rent,
            "Rent_pop": Rent_pop, #+
            "Div_priv": Div_priv,
            "TaxK": TaxK,
            "TaxW": TaxW,
            "wagePool": wagePool,
            "wagePool_net": wagePool_net,
            "Bg_plus": Bg_plus,
            "Itot": Itot,
            "SEC": SEC,
            "SVC": SVC,
            "TU": TU,
            "Rent_net": Rent_net,
            "Rent_net_pre_priv": Rent_net_pre_priv,
            "I_priv": I_priv,
            "I_priv_A": I_priv_A,
            "I_priv_H": I_priv_H,
            "I_priv_Q": I_priv_Q,
            "cm": cm,
            "cm_norm": cm_norm,
            "Gap": Gap,
            "yyK": yyK,
            "yyW": yyW,
            "yy_elit": yy_elit,
            "yy_elit_norm": yy_elit_norm,
            "Ineq": Ineq,
            "Ineq_eff": Ineq_eff,
            "Lctrl": Lctrl,
            "CollapseRisk": CollapseRisk,
            "SEC_core": SEC_core,
            "b_cm": b_cm,
            "d": d,
            "Birth": Birth,
            "Deaths": Deaths,
            "welfare_flow": welfare_flow,
            "elite_flow": elite_flow,
        }

    def step(self) -> Dict[str, float]:
        p = self.p
        s = self.s
        a = self.aux()
        dt = p.dt

        # Differential equations
        # dA = a["Itot"] * p.thetaA + a["I_priv_A"] - s.A / p.tauA #-
        dA = a["I_priv_A"] - s.A / p.tauA #+
        dA_pop = a["Itot"] - s.A_pop / p.tauA #+
        # dH = a["Itot"] * p.thetaH + a["I_priv_H"] - s.H / p.tauH #-
        dH = a["I_priv_H"] - s.H / p.tauH #+
        # dQ = a["Itot"] * p.thetaQ + a["I_priv_Q"] - s.Q / p.tauQ #-
        dQ = a["I_priv_Q"] - s.Q / p.tauQ #+
        
        # dBg = a["TaxK"] + a["TaxW"] - (a["TU"] + a["SVC"] + a["SEC"] + a["Itot"]) #-
        dBg_down = 0.0 #+
        if s.Bg > 0.0: #+
            dBg_down = a["TU"] + a["SVC"] + a["SEC"] + a["Itot"]    #+ 
            
        dBg = (a["TaxK"]
               + a["TaxW"]
               + a["Rent_pop"]
               - dBg_down #+
            )

        conflict_up = 0.0
        if s.C < 1.0:
            # conflict_up = p.rC * (p.b1 * a["Gap"] + p.b2 * p.fc12 / (a["food_pc"] + p.fc12) + p.b3 * a["Ineq_eff"]) * (1.0 - s.C)
            conflict_up = p.rC * (p.b1 * a["Gap"] + p.b2 * p.fc12 / (a["food_pc_norm"] + p.fc12) + p.b3 * a["Ineq_eff"]) * (1.0 - s.C) # dim correction
        conflict_down = 0.0
        if s.C > 0.0:
            conflict_down = p.dC * s.C * (1.0 + p.mu_sec * a["SEC_core"] ) # dim correction
        dC = conflict_up - conflict_down

        dFs = p.r_soil * (1.0 - s.Fs) - p.d_soil * a["Fprod"]

        dG_up = p.rG * (p.Gmax - s.G) if s.G < 1.0 else 0.0
        dG_down = p.dG * a["Lctrl"] / (1.0 + a["Teff"]) if s.G > 0.0 else 0.0
        dG = dG_up - dG_down

        dL_arb = -s.L_arb * 0.002
        dL_sol = p.gamma_sol * max(0.0, a["Eneed"] - a["Enet"]) - s.L_sol / p.tau_sol
        dP = a["Birth"] - a["Deaths"]
        dPh = -a["Pext"]
        dR = -p.kR * a["Enet"]

        dT_up = p.rT * (p.Tmax - s.T) if s.T < 1.0 else 0.0
        dT_down = 0.0
        if s.T > 0.0:
            dT_down = p.dT * (
                p.a1 * a["Gap"]
                + p.a2 * a["Ineq_eff"]
                + p.a3 * a["Ceff"]
                # + p.a4 * (p.fc12 / (a["food_pc"] + p.fc12))
                + p.a4 * (p.fc12 / (a["food_pc_norm"] + p.fc12)) # dim correction
                + p.a5 * (1.0 - a["Geff"])
                + p.a6 * a["CollapseRisk"]
            )
        dT = dT_up - dT_down

        dU_elit = a["elite_flow"]
        dU_pop = a["welfare_flow"]

        # Euler update
        s.A += dt * dA
        s.A_pop += dt * dA_pop  #+
        s.H += dt * dH
        s.Q += dt * dQ
        s.Bg += dt * dBg
        s.C += dt * dC
        s.Fs += dt * dFs
        s.G += dt * dG
        s.L_arb += dt * dL_arb
        s.L_sol += dt * dL_sol
        s.P += dt * dP
        s.Ph += dt * dPh
        s.R += dt * dR
        s.T += dt * dT
        s.U_elit += dt * dU_elit
        s.U_pop += dt * dU_pop

        # Soft bounds for physical sense
        s.A = max(0.0, s.A)
        s.A_pop = max(0.0, s.A_pop) #+
        s.Bg = max(0.0, s.Bg) #+
        s.H = max(0.0, s.H)
        s.Q = max(0.0, s.Q)
        s.Fs = max(0.0, s.Fs)
        s.L_arb = max(0.0, s.L_arb)
        s.L_sol = max(0.0, s.L_sol)
        s.P = max(p.eps, s.P)
        s.Ph = max(0.0, s.Ph)
        s.R = max(0.0, s.R)
        s.T = max(0.0, s.T)
        s.G = max(0.0, s.G)
        s.C = max(0.0, s.C)

        out = {"time": None, **a, **asdict(s)}
        return out

    def run(self, save_every: int = 1, sample_years: Optional[Iterable[float]] = None) -> "pd.DataFrame | List[Dict[str, float]]":
        steps = int(round(self.p.final_time / self.p.dt))
        rows: List[Dict[str, float]] = []
        sample_set = None if sample_years is None else {round(float(x), 10) for x in sample_years}
        for i in range(steps + 1):
            t = round(i * self.p.dt, 10)
            a = self.aux()
            row = {"time": t, **a, **asdict(self.s)}
            should_save = False
            if sample_set is not None:
                should_save = t in sample_set
            else:
                should_save = (i % save_every == 0)
            if should_save:
                rows.append(row)
            if i < steps:
                self.step()
        if pd is not None:
            return pd.DataFrame(rows)
        return rows
    
    def run_fast(
        self,
        track_thresholds: bool = True,
        thresholds: Optional[Dict[str, Tuple[str, float, str]]] = None,
    ) -> Dict[str, float]:
        """
        Fast simulation mode for optimization:
        - no DataFrame
        - no full trajectory storage
        - optional threshold tracking
        """
    
        if thresholds is None:
            thresholds = {
                "T_Y20": ("Y", 0.20, "lt"),
                "T_Y10": ("Y", 0.10, "lt"),
                "T_F20": ("food_pc", 0.20, "lt"),
                "T_F10": ("food_pc", 0.10, "lt"),
                "T_G35": ("G", 0.35, "lt"),
                "T_P80": ("P", 0.80, "lt"),
                "T_P60": ("P", 0.60, "lt"),
            }
    
        steps = int(round(self.p.final_time / self.p.dt))
    
        thr_times = None
        if track_thresholds:
            thr_times = {name: float("inf") for name in thresholds.keys()}
    
        def _hit(value: float, cutoff: float, mode: str) -> bool:
            if mode == "lt":
                return value < cutoff
            elif mode == "le":
                return value <= cutoff
            elif mode == "gt":
                return value > cutoff
            elif mode == "ge":
                return value >= cutoff
            else:
                raise ValueError(f"Unknown threshold mode: {mode}")
    
        for i in range(steps + 1):
            t = round(i * self.p.dt, 10)
            a = self.aux()
    
            if track_thresholds:
                for name, (var_name, cutoff, mode) in thresholds.items():
                    if math.isinf(thr_times[name]):
                        if var_name in a:
                            v = float(a[var_name])
                        elif hasattr(self.s, var_name):
                            v = float(getattr(self.s, var_name))
                        else:
                            raise KeyError(f"Variable '{var_name}' not found in aux() or state.")
    
                        if _hit(v, cutoff, mode):
                            thr_times[name] = t
    
            if i < steps:
                self.step()
    
        a_final = self.aux()
    
        result = {
            "time": float(self.p.final_time),
            "Y_100": float(a_final["Y"]) if "Y" in a_final else math.nan,
            "Y_R_100": float(a_final["YR"]) if "YR" in a_final else math.nan,
            "P_100": float(a_final["P"]) if "P" in a_final else float(getattr(self.s, "P", math.nan)),
            "G_100": float(a_final["G"]) if "G" in a_final else float(getattr(self.s, "G", math.nan)),
            "T_100": float(a_final["T"]) if "T" in a_final else float(getattr(self.s, "T", math.nan)),
            "food_pc_100": float(a_final["food_pc"]) if "food_pc" in a_final else math.nan,
            "U_pop_100": float(a_final["U_pop"]) if "U_pop" in a_final else float(getattr(self.s, "U_pop", math.nan)),
            "U_elit_100": float(a_final["U_elit"]) if "U_elit" in a_final else float(getattr(self.s, "U_elit", math.nan)),
        }
    
        if track_thresholds:
            result.update({k: float(v) for k, v in thr_times.items()})
    
        return result

# ------------------------------------------------------------
# Threshold utilities
# ------------------------------------------------------------

def first_below(series, threshold: float, time_series=None):
    if pd is not None and hasattr(series, "iloc"):
        mask = series < threshold
        if mask.any():
            idx = mask.idxmax() if mask.iloc[0] else mask[mask].index[0]
            if time_series is None:
                return idx
            return float(time_series.loc[idx])
        return math.inf
    # list fallback
    if time_series is None:
        time_series = list(range(len(series)))
    for t, x in zip(time_series, series):
        if x < threshold:
            return t
    return math.inf


def summarize_thresholds(df) -> Dict[str, float]:
    return {
        "T_Y20": first_below(df["Y"], 0.2, df["time"]),
        "T_Y10": first_below(df["Y"], 0.1, df["time"]),
        "T_F20": first_below(df["food_pc"], 0.2, df["time"]),
        "T_F10": first_below(df["food_pc"], 0.1, df["time"]),
        "T_G35": first_below(df["G"], 0.35, df["time"]),
        "T_P80": first_below(df["P"], 0.8, df["time"]),
        "T_P60": first_below(df["P"], 0.6, df["time"]),
    }


# ------------------------------------------------------------
# Regime profiles and comparison helpers
# ------------------------------------------------------------

# ------------------------------------------------------------
# Regime profiles and comparison helpers
# ------------------------------------------------------------

def make_noocracy() -> Params:
    return Params()

def make_noocracy_privinv(theta_priv_inv: float = 0.2) -> Params:
    p = make_noocracy()
    p.theta_priv_inv = theta_priv_inv
    return p

def make_noocracy_opt() -> Params:
    """
    Welfare-optimized synthetic regime found by GA
    (soft-penalty robust optimum, version 1).
    """
    p = Params()
    # 
    p.m0 = 0.869727 #         0.674958
    p.muU = 1.984067 #        1.894645
    p.mu_sec = 0.973617 #     0.872403
    p.mmax = 1.157685 #       1.391355

    p.sE_ag = 0.465179 #      0.607360
    p.sE_R = 0.408049 #       0.268761
    p.sE_sec = 0.031551 #     0.025251
    p.sE_house = 0.095221 #   0.098629

    p.taxK_rate = 0.795847 #  0.791472
    p.taxW_rate = 0.056322 #  0.071974

    
    p.thetaU = 0.336185 #     0.275529
    p.thetaSVC = 0.528415 #   0.502769 
    p.theta_sec = 0.045217 #  0.101401
    p.thta_inv = 0.090182 #   0.120301

    return p

def make_noocracy_opt_privinv(theta_priv_inv: float = 0.2) -> Params:
    p = make_noocracy_opt()
    p.theta_priv_inv = theta_priv_inv
    return p

def make_noocracy_opt_multi() -> Params:
    """
    Welfare-optimized synthetic regime found by GA
    (soft-penalty robust optimum + moo, version 2).
    """
    p = Params()

    p.m0 = 0.899737
    p.muU = 1.990414
    p.mu_sec = 0.913350
    p.mmax = 1.390736

    p.sE_ag = 0.797605
    p.sE_R = 0.105823
    p.sE_sec = 0.027023
    p.sE_house = 0.069547

    p.taxK_rate = 0.050936
    p.taxW_rate = 0.052420
    
    p.thetaU = 0.269393
    p.thetaSVC = 0.363909
    p.theta_sec = 0.293033
    p.thta_inv = 0.073663

    return p

def make_noocracy_opt_multi_privinv(theta_priv_inv: float = 0.2) -> Params:
    p = make_noocracy_opt_multi()
    p.theta_priv_inv = theta_priv_inv
    return p

def make_world3_reference() -> Params:
    """
    Reference / calibration regime:
    muted institutional design, closer to a classical Limits-to-Growth / World3 baseline.
    This is not a normative regime, but a reference scenario for comparison.
    """
    p = Params()

    # institutional / distributive defaults from the baseline Vensim specification
    p.m0 = 0.45
    p.muU = 1.60
    p.mu_sec = 0.80
    p.mmax = 1.20

    p.dG = 0.03
    p.rG = 0.05
    p.rT = 0.04

    p.sE_ag = 0.30
    p.sE_R = 0.35
    p.sE_sec = 0.10
    p.sE_house = 0.25

    p.taxK_rate = 0.50
    p.taxW_rate = 0.30

    p.thetaU = 0.25
    p.thetaSVC = 0.25
    p.theta_sec = 0.15
    p.thta_inv = 0.35

    return p

def make_world3_reference_privinv(theta_priv_inv: float = 0.2) -> Params:
    p = make_world3_reference()
    p.theta_priv_inv = theta_priv_inv
    return p

def make_cyberpunk() -> Params:
    p = Params()
    
    p.b3 = 0.35
    
    p.m0 = 0.85
    p.muU = 1.8
    p.mu_sec = 0.6
    p.mmax = 1.35
    
    p.dG = 0.03
    p.rG = 0.05
    
    p.sE_ag = 0.22
    p.sE_R = 0.33
    p.sE_sec = 0.12
    p.sE_house = 0.33
    
    p.taxK_rate = 0.2
    p.taxW_rate = 0.45
    
    p.thetaU = 0.20
    p.thetaSVC = 0.20
    p.theta_sec = 0.50
    p.thta_inv = 0.10
    
    return p

def make_cyberpunk_privinv(theta_priv_inv: float = 0.2) -> Params:
    p = make_cyberpunk()
    p.theta_priv_inv = theta_priv_inv
    return p

def make_fortress_elites() -> Params:
    p = make_cyberpunk()
    
    p.m0 = 0.45
    p.muU = 0.7
    p.mu_sec = 1.2
    p.mmax = 1.35
    
    p.sE_sec = 0.22
    p.sE_house = 0.26
    p.sE_R = 0.32
    p.sE_ag = 0.2
    
    p.taxK_rate = 0.15
    p.taxW_rate = 0.2
    
    p.thetaU = 0.18
    p.thetaSVC = 0.22
    p.theta_sec = 0.5
    p.thta_inv = 0.1
    
    return p

def make_fortress_elites_privinv(theta_priv_inv: float = 0.05) -> Params:
    p = make_fortress_elites()
    p.theta_priv_inv = theta_priv_inv
    return p

def make_neo_feudalism() -> Params:
    p = Params()
    
    p.b3 = 0.35
    
    p.m0 = 0.45
    p.muU = 0.9
    p.mu_sec = 0.9
    p.mmax = 0.85
    
    p.dG = 0.03
    p.rG = 0.05
    
    p.sE_ag = 0.28
    p.sE_R = 0.34
    p.sE_sec = 0.18
    p.sE_house = 0.2
    
    p.taxK_rate = 0.25
    p.taxW_rate = 0.6
    
    p.thetaU = 0.2
    p.thetaSVC = 0.24
    p.theta_sec = 0.46
    p.thta_inv = 0.1
    
    return p

def make_neo_feudalism_privinv(theta_priv_inv: float = 0.2) -> Params:
    p = make_neo_feudalism()
    p.theta_priv_inv = theta_priv_inv
    return p

def make_techno_communism() -> Params:
    p = Params()
    
    p.m0 = 0.45
    p.muU = 0.8
    p.mu_sec = 0.6
    p.mmax = 0.7
    
    p.dG = 0.03
    p.rG = 0.05
    
    p.sE_ag = 0.34
    p.sE_R = 0.42
    p.sE_sec = 0.12
    p.sE_house = 0.12
    
    p.taxK_rate = 0.65
    p.taxW_rate = 0.5
    
    p.thetaU = 0.31
    p.thetaSVC = 0.41
    p.theta_sec = 0.18
    p.thta_inv = 0.1
    
    return p

def make_techno_communism_privinv(theta_priv_inv: float = 0.2) -> Params:
    p = make_techno_communism()
    p.theta_priv_inv = theta_priv_inv
    return p

def make_techno_socialism() -> Params:
    p = Params()
    
    p.m0 = 0.8
    p.muU = 1.4
    p.mu_sec = 0.4
    p.mmax = 1.25
    
    p.dG = 0.03
    p.rG = 0.05
    
    p.sE_ag = 0.3
    p.sE_R = 0.32
    p.sE_sec = 0.08
    p.sE_house = 0.3
    
    p.taxK_rate = 0.6
    p.taxW_rate = 0.6
    
    p.thetaU = 0.29
    p.thetaSVC = 0.45
    p.theta_sec = 0.16
    p.thta_inv = 0.1
    
    return p

def make_techno_socialism_privinv(theta_priv_inv: float = 0.2) -> Params:
    p = make_techno_socialism()
    p.theta_priv_inv = theta_priv_inv
    return p


REGIME_BUILDERS = {
    "Cyberpunk": make_cyberpunk,
    "Fortress elites": make_fortress_elites,
    "Neo-feudalism": make_neo_feudalism,
    "Techno-communism": make_techno_communism,
    "Techno-socialism": make_techno_socialism,
    "Noocracy": make_noocracy,
    "Noocracy-opt": make_noocracy_opt,
    "Noocracy-opt-moo": make_noocracy_opt_multi,
    "World3-ref": make_world3_reference,
}

REGIME_BUILDERS_privinv = {
    "Cyberpunk": make_cyberpunk_privinv,
    "Fortress elites": make_fortress_elites_privinv,
    "Neo-feudalism": make_neo_feudalism_privinv,
    "Techno-communism": make_techno_communism_privinv,
    "Techno-socialism": make_techno_socialism_privinv,
    "Noocracy": make_noocracy_privinv,
    "Noocracy-opt": make_noocracy_opt_privinv,
    "Noocracy-opt-moo": make_noocracy_opt_multi_privinv,
    "World3-ref": make_world3_reference_privinv,
}


def compare_regimes(sample_years: Optional[Iterable[float]] = None, save_excel: Optional[str] = None):
    if sample_years is None:
        sample_years = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    outputs = {}
    summary_rows = []
    threshold_rows = []
    # for regime_name, builder in REGIME_BUILDERS.items():
    for regime_name, builder in REGIME_BUILDERS_privinv.items():     
        model = RegimeModel(builder())
        df = model.run(sample_years=sample_years)
        if pd is not None:
            out = df[["time", "Y", "YR", "Enet", "eta", "food_pc", "P", "PU", "T", "G", "Ceff", "Ineq_eff", "A", "H", "Q", "U_pop", "U_elit"]].copy()
            out = out.rename(columns={"YR": "Y_R", "Enet": "E_net", "PU": "P_U", "Ceff": "C"})
            outputs[regime_name] = out
            final = out.iloc[-1].to_dict()
            final = {f"{k}_100": v for k, v in final.items() if k != "time"}
            final["Regime"] = regime_name
            summary_rows.append(final)
            thr = summarize_thresholds(df)
            thr["Regime"] = regime_name
            threshold_rows.append(thr)
        else:
            outputs[regime_name] = df
    if pd is not None:
        summary_df = pd.DataFrame(summary_rows)
        thresholds_df = pd.DataFrame(threshold_rows)
        if save_excel:
            with pd.ExcelWriter(save_excel) as writer:
                for regime_name, out in outputs.items():
                    sheet = regime_name[:31]
                    out.to_excel(writer, sheet_name=sheet, index=False)
                summary_df.to_excel(writer, sheet_name="summary_100", index=False)
                thresholds_df.to_excel(writer, sheet_name="thresholds", index=False)
        return outputs, summary_df, thresholds_df
    return outputs, summary_rows, threshold_rows


# ------------------------------------------------------------
# Monte Carlo / sensitivity scaffolding
# ------------------------------------------------------------

def sample_params(base: Params, ranges: Dict[str, Tuple[float, float]], rng: random.Random) -> Params:
    data = asdict(base)
    for k, (lo, hi) in ranges.items():
        if k not in data:
            raise KeyError(f"Unknown parameter: {k}")
        data[k] = rng.uniform(lo, hi)
    return Params(**data)


def run_monte_carlo(
    base: Params,
    ranges: Dict[str, Tuple[float, float]],
    n: int = 100,
    seed: int = 42,
) -> "pd.DataFrame | List[Dict[str, float]]":
    rng = random.Random(seed)
    rows: List[Dict[str, float]] = []
    for i in range(n):
        p = sample_params(base, ranges, rng)
        model = RegimeModel(p)
        df = model.run(save_every=1)
        final = df.iloc[-1].to_dict() if pd is not None else df[-1]
        thr = summarize_thresholds(df)
        row = {
            "run": i,
            **{k: getattr(p, k) for k in ranges.keys()},
            **thr,
            "Y_100": final["Y"],
            "YR_100": final["YR"],
            "P_100": final["P"],
            "G_100": final["G"],
            "T_100": final["T"],
            "food_100": final["food_pc"],
            "U_pop_100": final["U_pop"],
            "U_elit_100": final["U_elit"],
        }
        rows.append(row)
    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def run_world_monte_carlo(
    n: int = 100,
    seed: int = 42,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    save_excel: Optional[str] = None,
):
    if ranges is None:
        ranges = {
            "gamma": (0.55, 0.75),
            "beta": (0.90, 1.20),
            "kR": (0.08, 0.12),
            "Yld0": (28.0, 42.0),
            "d_soil": (0.02, 0.04),
            "dT": (0.020, 0.035),
            "a4": (0.18, 0.32),
            "dG": (0.022, 0.038),
            "waste": (0.15, 0.25),
        }
    rng = random.Random(seed)
    all_rows: List[Dict[str, float]] = []
    for run in range(n):
        sampled = sample_params(Params(), ranges, rng)
        sampled_dict = {k: getattr(sampled, k) for k in ranges.keys()}
        
        # for regime_name, builder in REGIME_BUILDERS.items():
        for regime_name, builder in REGIME_BUILDERS_privinv.items():
            p = builder()
            for k in ranges.keys():
                setattr(p, k, getattr(sampled, k))
            model = RegimeModel(p)
            df = model.run(save_every=1)
            final = df.iloc[-1].to_dict() if pd is not None else df[-1]
            thr = summarize_thresholds(df)
            row = {
                "run": run,
                "Regime": regime_name,
                **sampled_dict,
                **thr,
                "Y_100": final["Y"],
                "Y_R_100": final["YR"],
                "P_100": final["P"],
                "G_100": final["G"],
                "T_100": final["T"],
                "food_pc_100": final["food_pc"],
                "U_pop_100": final["U_pop"],
                "U_elit_100": final["U_elit"],
            }
            all_rows.append(row)
    if pd is not None:
        out = pd.DataFrame(all_rows)
        if save_excel:
            with pd.ExcelWriter(save_excel) as writer:
                out.to_excel(writer, sheet_name="mc_runs", index=False)
                out.groupby("Regime")[["Y_R_100", "P_100", "G_100", "T_100", "food_pc_100", "U_pop_100", "U_elit_100", "T_Y20", "T_F20", "T_G35", "T_P80"]].agg(["mean", "std", "median"]).to_excel(writer, sheet_name="mc_summary")
        return out
    return all_rows

def build_mc_publication_summary(
    mc_runs,
    horizon: float = 100.0,
    regime_col: str = "Regime",
    run_col: str = "run",
    save_excel: Optional[str] = None,
):
    """
    Post-process output of run_world_monte_carlo(...) into publication-ready tables.

    Parameters
    ----------
    mc_runs : pd.DataFrame
        DataFrame returned by run_world_monte_carlo(...).
    horizon : float
        Simulation horizon used for right-censoring threshold metrics.
    regime_col : str
        Column with regime names.
    run_col : str
        Column identifying Monte Carlo world/run id.
    save_excel : Optional[str]
        If provided, save all summary tables to an Excel workbook.

    Returns
    -------
    dict[str, pd.DataFrame]
        {
            "mc_enriched": ...,
            "summary_main": ...,
            "thresholds_summary": ...,
            "wins": ...,
            "ranks": ...,
        }
    """
    if pd is None:
        raise RuntimeError("build_mc_publication_summary requires pandas.")

    df = mc_runs.copy()

    # -----------------------------
    # Config
    # -----------------------------
    final_metrics = [
        "Y_100",
        "Y_R_100",
        "P_100",
        "G_100",
        "T_100",
        "food_pc_100",
        "U_pop_100",
        "U_elit_100",
    ]
    threshold_metrics = [
        "T_Y20",
        "T_Y10",
        "T_F20",
        "T_F10",
        "T_G35",
        "T_P80",
        "T_P60",
    ]

    for col in [regime_col, run_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in mc_runs.")

    for col in final_metrics + threshold_metrics:
        if col not in df.columns:
            # threshold columns may be partially absent in some variants
            # so we tolerate that and only use existing ones
            pass

    final_metrics = [c for c in final_metrics if c in df.columns]
    threshold_metrics = [c for c in threshold_metrics if c in df.columns]

    # -----------------------------
    # Threshold post-processing
    # -----------------------------
    for col in threshold_metrics:
        # mark whether threshold was crossed within horizon
        df[f"{col}_crossed"] = (~df[col].isna()) & (~df[col].isin([math.inf, float("inf")]))
        df[f"{col}_not_crossed"] = ~df[f"{col}_crossed"]

        # right-censored timing for cleaner summary/plots
        df[f"{col}_cens"] = df[col].replace([math.inf, float("inf")], horizon).fillna(horizon)

        # event indicator in survival-analysis style
        df[f"{col}_event"] = df[f"{col}_crossed"].astype(int)

    # -----------------------------
    # Summary helpers
    # -----------------------------
    def _series_summary(s):
        s = s.dropna()
        if len(s) == 0:
            return {
                "mean": math.nan,
                "std": math.nan,
                "median": math.nan,
                "q25": math.nan,
                "q75": math.nan,
                "min": math.nan,
                "max": math.nan,
            }
        return {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "median": float(s.median()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    # -----------------------------
    # Main summary by regime
    # -----------------------------
    summary_rows = []

    for regime, g in df.groupby(regime_col, sort=False):
        row = {regime_col: regime, "n_runs": int(len(g))}

        # final metrics
        for col in final_metrics:
            stats = _series_summary(g[col])
            for k, v in stats.items():
                row[f"{col}_{k}"] = v

        # threshold metrics
        for col in threshold_metrics:
            stats = _series_summary(g[f"{col}_cens"])
            for k, v in stats.items():
                row[f"{col}_cens_{k}"] = v

            row[f"{col}_not_crossed_share"] = float(g[f"{col}_not_crossed"].mean())
            row[f"{col}_crossed_share"] = float(g[f"{col}_crossed"].mean())

        summary_rows.append(row)

    summary_main = pd.DataFrame(summary_rows)

    # -----------------------------
    # Threshold-focused compact table
    # -----------------------------
    thr_rows = []
    for regime, g in df.groupby(regime_col, sort=False):
        row = {regime_col: regime, "n_runs": int(len(g))}
        for col in threshold_metrics:
            row[f"{col}_median_cens"] = float(g[f"{col}_cens"].median())
            row[f"{col}_mean_cens"] = float(g[f"{col}_cens"].mean())
            row[f"{col}_q25_cens"] = float(g[f"{col}_cens"].quantile(0.25))
            row[f"{col}_q75_cens"] = float(g[f"{col}_cens"].quantile(0.75))
            row[f"{col}_not_crossed_share"] = float(g[f"{col}_not_crossed"].mean())
        thr_rows.append(row)
    thresholds_summary = pd.DataFrame(thr_rows)

    # -----------------------------
    # Win shares and average ranks
    # -----------------------------
    # Metrics where "higher is better"
    higher_is_better = [c for c in ["Y_R_100", "P_100", "G_100", "T_100", "food_pc_100", "U_pop_100", "U_elit_100"] if c in df.columns]
    # Metrics where "later threshold crossing is better"
    higher_is_better += [f"{c}_cens" for c in threshold_metrics]

    win_records = []
    rank_records = []

    # ranks within each world/run across regimes
    for metric in higher_is_better:
        tmp = df[[run_col, regime_col, metric]].copy()
        tmp["rank"] = tmp.groupby(run_col)[metric].rank(method="average", ascending=False)

        # win share: tie-aware
        tmp["is_win"] = tmp.groupby(run_col)[metric].transform(lambda x: x == x.max())

        wins = tmp.groupby(regime_col)["is_win"].mean().reset_index()
        wins["metric"] = metric
        wins = wins.rename(columns={"is_win": "win_share"})
        win_records.append(wins)

        ranks = tmp.groupby(regime_col)["rank"].mean().reset_index()
        ranks["metric"] = metric
        ranks = ranks.rename(columns={"rank": "avg_rank"})
        rank_records.append(ranks)

    wins_df = pd.concat(win_records, ignore_index=True) if win_records else pd.DataFrame(columns=[regime_col, "metric", "win_share"])
    ranks_df = pd.concat(rank_records, ignore_index=True) if rank_records else pd.DataFrame(columns=[regime_col, "metric", "avg_rank"])

    # wider versions for convenience
    wins_wide = wins_df.pivot(index=regime_col, columns="metric", values="win_share").reset_index() if not wins_df.empty else wins_df
    ranks_wide = ranks_df.pivot(index=regime_col, columns="metric", values="avg_rank").reset_index() if not ranks_df.empty else ranks_df

    # -----------------------------
    # Optional: compact publication table
    # -----------------------------
    compact_cols = [regime_col, "n_runs"]

    for col in ["G_100", "U_pop_100", "U_elit_100", "Y_R_100", "food_pc_100", "P_100"]:
        if f"{col}_median" in summary_main.columns:
            compact_cols += [f"{col}_median", f"{col}_q25", f"{col}_q75"]

    for col in ["T_G35", "T_F20", "T_P80", "T_Y20"]:
        if f"{col}_median_cens" in thresholds_summary.columns:
            pass

    publication_table = summary_main[compact_cols].copy()

    for col in ["T_G35", "T_F20", "T_P80", "T_Y20"]:
        if col in threshold_metrics:
            publication_table = publication_table.merge(
                thresholds_summary[
                    [
                        regime_col,
                        f"{col}_median_cens",
                        f"{col}_q25_cens",
                        f"{col}_q75_cens",
                        f"{col}_not_crossed_share",
                    ]
                ],
                on=regime_col,
                how="left",
            )

    # -----------------------------
    # Save
    # -----------------------------
    if save_excel:
        with pd.ExcelWriter(save_excel) as writer:
            df.to_excel(writer, sheet_name="mc_enriched", index=False)
            summary_main.to_excel(writer, sheet_name="summary_main", index=False)
            thresholds_summary.to_excel(writer, sheet_name="thresholds_summary", index=False)
            publication_table.to_excel(writer, sheet_name="publication_table", index=False)
            wins_df.to_excel(writer, sheet_name="wins_long", index=False)
            ranks_df.to_excel(writer, sheet_name="ranks_long", index=False)
            if isinstance(wins_wide, pd.DataFrame):
                wins_wide.to_excel(writer, sheet_name="wins_wide", index=False)
            if isinstance(ranks_wide, pd.DataFrame):
                ranks_wide.to_excel(writer, sheet_name="ranks_wide", index=False)

    return {
        "mc_enriched": df,
        "summary_main": summary_main,
        "thresholds_summary": thresholds_summary,
        "publication_table": publication_table,
        "wins": wins_df,
        "ranks": ranks_df,
        "wins_wide": wins_wide,
        "ranks_wide": ranks_wide,
    }


# -----------------------------
# Optimization config
# -----------------------------

POLICY_VARS = [
    "m0",
    "mmax",
    "muU",
    "mu_sec",
    "sE_ag",
    "sE_house",
    "sE_R",
    "sE_sec",
    "taxK_rate",
    "taxW_rate",
    "theta_sec",
    "thetaSVC",
    "thetaU",
    "thta_inv",
]

DEFAULT_POLICY_BOUNDS = {
    # based on existing regime envelope, slightly widened but still disciplined
    "m0": (0.35, 0.90),
    "mmax": (0.60, 1.40),
    "muU": (0.40, 2.00),
    "mu_sec": (0.20, 1.40),

    # raw shares; will be normalized inside decoder
    "sE_ag": (0.05, 0.60),
    "sE_house": (0.05, 0.60),
    "sE_R": (0.05, 0.70),
    "sE_sec": (0.02, 0.30),

    "taxK_rate": (0.05, 0.80),
    "taxW_rate": (0.05, 0.80),

    # raw budget shares; will be normalized inside decoder
    "theta_sec": (0.02, 0.50),
    "thetaSVC": (0.02, 0.60),
    "thetaU": (0.02, 0.40),
    "thta_inv": (0.10, 0.70),
}

DEFAULT_WORLD_RANGES = {
    "gamma": (0.55, 0.75),
    "beta": (0.90, 1.20),
    "kR": (0.08, 0.12),
    "Yld0": (28.0, 42.0),
    "d_soil": (0.02, 0.04),
    "dT": (0.020, 0.035),
    "a4": (0.18, 0.32),
    "dG": (0.022, 0.038),
    "waste": (0.15, 0.25),
}


def _normalize_positive_block(vals, eps: float = 1e-12):
    arr = np.asarray(vals, dtype=float)
    arr = np.maximum(arr, eps)
    s = float(arr.sum())
    if s <= eps:
        arr[:] = 1.0 / len(arr)
    else:
        arr = arr / s
    return arr


def policy_vector_to_params(
    x,
    # base_builder=make_noocracy,
    base_builder=make_noocracy_privinv,
    world_override: Optional[Dict[str, float]] = None,
):
    """
    Decode optimization vector into Params.

    Two blocks are normalized to simplex:
    - energy shares: sE_ag, sE_house, sE_R, sE_sec
    - budget shares: theta_sec, thetaSVC, thetaU, thta_inv
    """
    p = base_builder()

    x = np.asarray(x, dtype=float)
    d = dict(zip(POLICY_VARS, x))

    # scalar policy knobs
    p.m0 = float(d["m0"])
    p.mmax = float(d["mmax"])
    p.muU = float(d["muU"])
    p.mu_sec = float(d["mu_sec"])
    p.taxK_rate = float(d["taxK_rate"])
    p.taxW_rate = float(d["taxW_rate"])

    # normalize energy allocation shares
    e_block = _normalize_positive_block([
        d["sE_ag"],
        d["sE_house"],
        d["sE_R"],
        d["sE_sec"],
    ])
    p.sE_ag = float(e_block[0])
    p.sE_house = float(e_block[1])
    p.sE_R = float(e_block[2])
    p.sE_sec = float(e_block[3])

    # normalize budget allocation shares
    b_block = _normalize_positive_block([
        d["theta_sec"],
        d["thetaSVC"],
        d["thetaU"],
        d["thta_inv"],
    ])
    p.theta_sec = float(b_block[0])
    p.thetaSVC = float(b_block[1])
    p.thetaU = float(b_block[2])
    p.thta_inv = float(b_block[3])

    # apply world overrides if provided
    if world_override:
        for k, v in world_override.items():
            setattr(p, k, float(v))

    return p


def evaluate_policy_single_world(
    x,
    # base_builder=make_noocracy,
    base_builder=make_noocracy_privinv,
    world_override: Optional[Dict[str, float]] = None,
    track_thresholds: bool = False,
):
    p = policy_vector_to_params(
        x=x,
        base_builder=base_builder,
        world_override=world_override,
    )
    model = RegimeModel(p)
    return model.run_fast(track_thresholds=track_thresholds)


def build_world_bank(
    n_worlds: int = 24,
    seed: int = 123,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
):
    """
    Fixed deterministic bank of world parameterizations.
    Same seed => same bank => reproducible robust objective.
    """
    if ranges is None:
        ranges = DEFAULT_WORLD_RANGES

    rng = random.Random(seed)
    worlds = []
    for _ in range(n_worlds):
        p = sample_params(Params(), ranges, rng)
        worlds.append({k: getattr(p, k) for k in ranges.keys()})
    return worlds


def evaluate_policy_robust(
    x,
    # base_builder=make_noocracy,
    base_builder=make_noocracy_privinv,
    world_bank: Optional[List[Dict[str, float]]] = None,
    viability_floors: Optional[Dict[str, float]] = None,
    penalty_weight: float = 10.0,
    return_details: bool = False,
):
    """
    Deterministic robust objective:
    maximize mean U_pop_100 over a fixed world_bank.

    For speed, aggregation is done without pandas.
    """
    if world_bank is None:
        world_bank = [None]

    rows = []
    sum_U_pop = 0.0
    sum_U_elit = 0.0
    sum_G = 0.0
    sum_T = 0.0
    sum_P = 0.0
    sum_food = 0.0

    n = len(world_bank)

    for w in world_bank:
        r = evaluate_policy_single_world(
            x=x,
            base_builder=base_builder,
            world_override=w,
            track_thresholds=False,
        )
        if return_details:
            rows.append(r)

        sum_U_pop += float(r["U_pop_100"])
        sum_U_elit += float(r["U_elit_100"])
        sum_G += float(r["G_100"])
        sum_T += float(r["T_100"])
        sum_P += float(r["P_100"])
        sum_food += float(r["food_pc_100"])

    mean_U_pop = sum_U_pop / n
    mean_U_elit = sum_U_elit / n
    mean_G = sum_G / n
    mean_T = sum_T / n
    mean_P = sum_P / n
    mean_food = sum_food / n

    penalty = 0.0
    if viability_floors:
        vals = {
            "G_100": mean_G,
            "T_100": mean_T,
            "P_100": mean_P,
            "food_pc_100": mean_food,
        }
        for k, floor in viability_floors.items():
            if floor is not None and vals[k] < floor:
                penalty += penalty_weight * (floor - vals[k])

    result = {
        "objective": mean_U_pop - penalty,
        "mean_U_pop_100": mean_U_pop,
        "mean_U_elit_100": mean_U_elit,
        "mean_G_100": mean_G,
        "mean_T_100": mean_T,
        "mean_P_100": mean_P,
        "mean_food_pc_100": mean_food,
        "penalty": penalty,
    }

    if return_details:
        result["details"] = pd.DataFrame(rows) if pd is not None else rows

    return result


class PolicyOptimizationProblem(ElementwiseProblem):
    def __init__(
        self,
        # base_builder=make_noocracy,
        base_builder=make_noocracy_privinv,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        world_bank: Optional[List[Dict[str, float]]] = None,
        viability_floors: Optional[Dict[str, float]] = None,
        penalty_weight: float = 10.0,
        **kwargs
    ):
        if bounds is None:
            bounds = DEFAULT_POLICY_BOUNDS

        self.base_builder = base_builder
        self.policy_bounds = bounds
        self.world_bank = world_bank
        self.viability_floors = viability_floors
        self.penalty_weight = penalty_weight

        xl = np.array([bounds[k][0] for k in POLICY_VARS], dtype=float)
        xu = np.array([bounds[k][1] for k in POLICY_VARS], dtype=float)

        super().__init__(
            n_var=len(POLICY_VARS),
            n_obj=1,
            n_ieq_constr=1,
            xl=xl,
            xu=xu,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        i_m0 = POLICY_VARS.index("m0")
        i_mmax = POLICY_VARS.index("mmax")

        # constraint: m0 <= mmax
        g_m = float(x[i_m0] - x[i_mmax])

        ev = evaluate_policy_robust(
            x=x,
            base_builder=self.base_builder,
            world_bank=self.world_bank,
            viability_floors=self.viability_floors,
            penalty_weight=self.penalty_weight,
            return_details=False,
        )

        # pymoo minimizes
        out["F"] = np.array([-ev["objective"]], dtype=float)
        out["G"] = np.array([g_m], dtype=float)

class PolicyParetoProblem(ElementwiseProblem):
    """
    Multi-objective policy search:
    maximize mean_U_pop_100 and mean_U_elit_100
    over a fixed world_bank.

    pymoo minimizes, therefore:
        F1 = -mean_U_pop_100
        F2 = -mean_U_elit_100
    """

    def __init__(
        self,
        # base_builder=make_noocracy,
        base_builder=make_noocracy_privinv,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        world_bank: Optional[List[Dict[str, float]]] = None,
        viability_floors: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        if bounds is None:
            bounds = DEFAULT_POLICY_BOUNDS

        self.base_builder = base_builder
        self.policy_bounds = bounds
        self.world_bank = world_bank
        self.viability_floors = viability_floors or {}

        xl = np.array([bounds[k][0] for k in POLICY_VARS], dtype=float)
        xu = np.array([bounds[k][1] for k in POLICY_VARS], dtype=float)

        # constraints:
        # 1) m0 <= mmax
        # 2) viability floors: floor - metric <= 0
        n_ieq = 1 + len(self.viability_floors)

        super().__init__(
            n_var=len(POLICY_VARS),
            n_obj=2,
            n_ieq_constr=n_ieq,
            xl=xl,
            xu=xu,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        ev = evaluate_policy_robust(
            x=x,
            base_builder=self.base_builder,
            world_bank=self.world_bank,
            viability_floors=None,   # here floors go as hard constraints, not penalty
            penalty_weight=0.0,
            return_details=False,
        )

        # maximize both objectives => minimize negatives
        f1 = -float(ev["mean_U_pop_100"])
        f2 = -float(ev["mean_U_elit_100"])

        i_m0 = POLICY_VARS.index("m0")
        i_mmax = POLICY_VARS.index("mmax")

        g_list = []

        # hard constraint: m0 <= mmax
        g_list.append(float(x[i_m0] - x[i_mmax]))

        # hard viability constraints: floor - value <= 0
        metric_map = {
            "G_100": float(ev["mean_G_100"]),
            "T_100": float(ev["mean_T_100"]),
            "P_100": float(ev["mean_P_100"]),
            "food_pc_100": float(ev["mean_food_pc_100"]),
        }

        for metric_name, floor in self.viability_floors.items():
            val = metric_map[metric_name]
            g_list.append(float(floor - val))

        out["F"] = np.array([f1, f2], dtype=float)
        out["G"] = np.array(g_list, dtype=float)

def optimize_policy_pareto(
    # base_builder=make_noocracy,
    base_builder=make_noocracy_privinv,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    world_bank: Optional[List[Dict[str, float]]] = None,
    viability_floors: Optional[Dict[str, float]] = None,
    pop_size: int = 160,
    n_gen: int = 120,
    seed: int = 42,
    verbose: bool = True,
    n_jobs: int = 1,
    save_excel: Optional[str] = None,
):
    """
    Compute Pareto frontier for:
        maximize mean_U_pop_100
        maximize mean_U_elit_100

    under optional hard viability constraints.
    """

    if bounds is None:
        bounds = DEFAULT_POLICY_BOUNDS

    if world_bank is None:
        world_bank = build_world_bank(n_worlds=12, seed=123)

    pool = None
    runner = None

    try:
        if n_jobs is not None and n_jobs > 1:
            pool = Pool(processes=n_jobs)
            runner = StarmapParallelization(pool.starmap)

        problem = PolicyParetoProblem(
            base_builder=base_builder,
            bounds=bounds,
            world_bank=world_bank,
            viability_floors=viability_floors,
            elementwise_runner=runner,
        )

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=20),
            mutation=PM(eta=25),
            eliminate_duplicates=True,
        )

        result = minimize(
            problem,
            algorithm,
            termination=get_termination("n_gen", n_gen),
            seed=seed,
            verbose=verbose,
        )

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # decode Pareto set
    X = np.asarray(result.X, dtype=float)
    F = np.asarray(result.F, dtype=float)

    rows = []
    for i in range(len(X)):
        x_i = X[i]
        p_i = policy_vector_to_params(
            x=x_i,
            base_builder=base_builder,
            world_override=None,
        )

        ev_i = evaluate_policy_robust(
            x=x_i,
            base_builder=base_builder,
            world_bank=world_bank,
            viability_floors=None,
            penalty_weight=0.0,
            return_details=False,
        )

        row = {
            "solution_id": i,
            "mean_U_pop_100": float(ev_i["mean_U_pop_100"]),
            "mean_U_elit_100": float(ev_i["mean_U_elit_100"]),
            "mean_G_100": float(ev_i["mean_G_100"]),
            "mean_T_100": float(ev_i["mean_T_100"]),
            "mean_P_100": float(ev_i["mean_P_100"]),
            "mean_food_pc_100": float(ev_i["mean_food_pc_100"]),
        }

        for k in POLICY_VARS:
            row[k] = float(getattr(p_i, k))

        rows.append(row)

    frontier_df = pd.DataFrame(rows) if pd is not None else rows

    if pd is not None and not frontier_df.empty:
        frontier_df = frontier_df.sort_values(
            by=["mean_U_pop_100", "mean_U_elit_100"],
            ascending=[True, True]
        ).reset_index(drop=True)

        # optional helpers for picking representative points
        frontier_df["rank_U_pop"] = frontier_df["mean_U_pop_100"].rank(ascending=False, method="min")
        frontier_df["rank_U_elit"] = frontier_df["mean_U_elit_100"].rank(ascending=False, method="min")

        # knee-like heuristic: normalize both objectives and maximize sum
        up_min = frontier_df["mean_U_pop_100"].min()
        up_max = frontier_df["mean_U_pop_100"].max()
        ue_min = frontier_df["mean_U_elit_100"].min()
        ue_max = frontier_df["mean_U_elit_100"].max()

        if up_max > up_min:
            frontier_df["U_pop_norm"] = (frontier_df["mean_U_pop_100"] - up_min) / (up_max - up_min)
        else:
            frontier_df["U_pop_norm"] = 1.0

        if ue_max > ue_min:
            frontier_df["U_elit_norm"] = (frontier_df["mean_U_elit_100"] - ue_min) / (ue_max - ue_min)
        else:
            frontier_df["U_elit_norm"] = 1.0

        frontier_df["knee_score"] = frontier_df["U_pop_norm"] + frontier_df["U_elit_norm"]

    if save_excel and pd is not None:
        with pd.ExcelWriter(save_excel) as writer:
            frontier_df.to_excel(writer, sheet_name="pareto_frontier", index=False)

    return {
        "result": result,
        "frontier": frontier_df,
    }

def pick_pareto_representatives(frontier_df: "pd.DataFrame") -> Dict[str, "pd.Series"]:
    """
    Pick three representative points from Pareto frontier:
    - welfare-max
    - elite-max
    - compromise (knee)
    """
    if pd is None or frontier_df is None or frontier_df.empty:
        return {}

    welfare_max = frontier_df.loc[frontier_df["mean_U_pop_100"].idxmax()]
    elite_max = frontier_df.loc[frontier_df["mean_U_elit_100"].idxmax()]
    compromise = frontier_df.loc[frontier_df["knee_score"].idxmax()] if "knee_score" in frontier_df.columns else welfare_max

    return {
        "welfare_max": welfare_max,
        "elite_max": elite_max,
        "compromise": compromise,
    }


def optimize_policy_ga(
    # base_builder=make_noocracy,
    base_builder=make_noocracy_privinv,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    world_bank: Optional[List[Dict[str, float]]] = None,
    viability_floors: Optional[Dict[str, float]] = None,
    penalty_weight: float = 10.0,
    pop_size: int = 120,
    n_gen: int = 120,
    seed: int = 42,
    verbose: bool = True,
    n_jobs: int = 1,
):
    """
    Robust GA optimization of policy parameters.

    Objective:
        maximize mean U_pop_100 over a fixed world_bank,
        minus penalties for violating viability floors.

    Notes
    -----
    - Uses pymoo GA
    - Uses StarmapParallelization when n_jobs > 1
    - Requires PolicyOptimizationProblem to pass **kwargs into ElementwiseProblem
    """

    if bounds is None:
        bounds = DEFAULT_POLICY_BOUNDS

    if world_bank is None:
        world_bank = build_world_bank(n_worlds=12, seed=123)

    pool = None
    runner = None

    try:
        if n_jobs is not None and n_jobs > 1:
            pool = Pool(processes=n_jobs)
            runner = StarmapParallelization(pool.starmap)

        problem = PolicyOptimizationProblem(
            base_builder=base_builder,
            bounds=bounds,
            world_bank=world_bank,
            viability_floors=viability_floors,
            penalty_weight=penalty_weight,
            elementwise_runner=runner,
        )

        algorithm = GA(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=20),
            mutation=PM(eta=25),
            eliminate_duplicates=True,
        )

        result = minimize(
            problem,
            algorithm,
            termination=get_termination("n_gen", n_gen),
            seed=seed,
            verbose=verbose,
        )

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    x_best = np.asarray(result.X, dtype=float)

    p_best = policy_vector_to_params(
        x=x_best,
        base_builder=base_builder,
        world_override=None,
    )

    robust_eval = evaluate_policy_robust(
        x=x_best,
        base_builder=base_builder,
        world_bank=world_bank,
        viability_floors=viability_floors,
        penalty_weight=penalty_weight,
        return_details=True,
    )

    best_policy = {k: getattr(p_best, k) for k in POLICY_VARS}

    summary = {
        "best_objective": float(robust_eval["objective"]),
        "mean_U_pop_100": float(robust_eval["mean_U_pop_100"]),
        "mean_G_100": float(robust_eval["mean_G_100"]),
        "mean_T_100": float(robust_eval["mean_T_100"]),
        "mean_P_100": float(robust_eval["mean_P_100"]),
        "mean_food_pc_100": float(robust_eval["mean_food_pc_100"]),
        "penalty": float(robust_eval["penalty"]),
        **best_policy,
    }

    return {
        "result": result,
        "x_best": x_best,
        "params_best": p_best,
        "best_policy": best_policy,
        "summary": summary,
        "details": robust_eval.get("details", None),
    }


# ------------------------------------------------------------
# Example
# ------------------------------------------------------------

if __name__ == "__main__":
    # if pd is not None:
    #     # outputs, summary_df, thresholds_df = compare_regimes(
    #     #     sample_years=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    #     #     save_excel="regime_comparison.xlsx",
    #     # )
    #     # print(summary_df)
    #     # print(thresholds_df)
    #     # print("Saved: regime_comparison.xlsx")

    #     mc = run_world_monte_carlo(n=100, seed=123, save_excel="world_monte_carlo.xlsx")
    #     # print(mc.head())
    #     # print("Saved: world_monte_carlo.xlsx")
        
    #     pub = build_mc_publication_summary(
    #         mc,
    #         horizon=100.0,
    #         save_excel="world_monte_carlo_publication.xlsx",
    #     )
        
    # else:
    #     outputs, summary_rows, threshold_rows = compare_regimes()
    #     print(summary_rows)
    #     print(threshold_rows)
    
    # check fast
    # p = make_noocracy()

    # m1 = RegimeModel(p)
    # df = m1.run(save_every=1)
    # final_df = df.iloc[-1].to_dict()
    # thr_df = summarize_thresholds(df)
    
    # m2 = RegimeModel(p)
    # fast = m2.run_fast(track_thresholds=True)
    
    # print(final_df["U_pop"], fast["U_pop_100"])
    # print(final_df["U_elit"], fast["U_elit_100"])
    # print(final_df["G"], fast["G_100"])
    # print(final_df["T"], fast["T_100"])
    # print(final_df["P"], fast["P_100"])
    # print(final_df["Y"], fast["Y_100"])
    # print(final_df["YR"], fast["Y_R_100"])
    # print(final_df["food_pc"], fast["food_pc_100"])
    
    # for k, v in thr_df.items():
    #     print(k, v, fast[k])
    
    # 1) fixed deterministic world bank
    world_bank = build_world_bank(
        # n_worlds=24,
        n_worlds=12,
        seed=123,
        ranges=DEFAULT_WORLD_RANGES,
    )

    # 2) optional viability floors to avoid pathological "high U_pop but collapse"
    floors = {
        "G_100": 0.3,
        "T_100": 0.3,
        "P_100": 0.50,
        "food_pc_100": 0.10,
    }

    # # 3) optimize
    # opt = optimize_policy_ga(
    #     base_builder=make_noocracy,
    #     world_bank=world_bank,
    #     viability_floors=floors,
    #     penalty_weight=12.0,
    #     pop_size=48,
    #     n_gen=25,
    #     seed=42,
    #     verbose=True,
    #     n_jobs=8,
    # )

    # print("=== BEST POLICY ===")
    # for k, v in opt["best_policy"].items():
    #     print(f"{k:12s} = {v:.6f}")

    # print("\n=== SUMMARY ===")
    # for k, v in opt["summary"].items():
    #     if isinstance(v, float):
    #         print(f"{k:18s} = {v:.6f}")
    #     else:
    #         print(f"{k:18s} = {v}")

    # if pd is not None:
    #     opt["details"].to_excel("policy_opt_details.xlsx", index=False)
    #     pd.DataFrame([opt["summary"]]).to_excel("policy_opt_summary.xlsx", index=False)
    #     print("\nSaved: policy_opt_details.xlsx, policy_opt_summary.xlsx")
    
    pareto = optimize_policy_pareto(
        # base_builder=make_noocracy,
        base_builder=make_noocracy_privinv,
        world_bank=world_bank,
        viability_floors=floors,
        pop_size=96,
        n_gen=80,
        seed=42,
        verbose=True,
        n_jobs=8,
        save_excel="policy_pareto_frontier.xlsx",
    )

    frontier = pareto["frontier"]
    reps = pick_pareto_representatives(frontier)

    print(frontier[[
        "solution_id",
        "mean_U_pop_100",
        "mean_U_elit_100",
        "mean_G_100",
        "mean_T_100",
        "mean_P_100",
        "mean_food_pc_100",
        "knee_score",
    ]].sort_values("mean_U_pop_100", ascending=False).head(10))

    print("\n=== REPRESENTATIVES ===")
    for name, row in reps.items():
        print(f"\n[{name}]")
        print(row[[
            "mean_U_pop_100",
            "mean_U_elit_100",
            "mean_G_100",
            "mean_T_100",
            "mean_P_100",
            "mean_food_pc_100",
        ]])
