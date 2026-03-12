from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable, Tuple
import math
import random

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


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
    L12: float = 0.15
    P12: float = 0.8
    T12: float = 0.35
    G12: float = 0.35
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

    # Energy shares
    sE_ag: float = 0.36
    sE_house: float = 0.2
    sE_R: float = 0.38
    sE_sec: float = 0.06

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
    mu_sec: float = 0.2
    muU: float = 1.0
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
    thetaA: float = 0.4
    thetaH: float = 0.3
    thetaQ: float = 0.3
    theta_sec: float = 0.06
    thetaSVC: float = 0.32
    thetaU: float = 0.18
    thta_inv: float = 0.44
    taxK_rate: float = 0.55
    taxW_rate: float = 0.6

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
    cm0: float = 0.147
    food_pc0: float = 0.53
    yy_elit0: float = 6.06
    c_star: float = 1.0

    # Dividend / enclosure
    gamma_priv: float = 0.0

    # Post-employment / absorptive sector
    m0: float = 0.55
    mmin: float = 0.3
    mmax: float = 0.95

    # State decay of tech capital
    tauA: float = 28.0
    tauH: float = 16.0
    tauQ: float = 12.0
    
    # Private elite reinvestment
    theta_priv_inv: float = 0.0
    theta_priv_A: float = 0.45
    theta_priv_H: float = 0.25
    theta_priv_Q: float = 0.30


@dataclass
class State:
    A: float = 0.6
    Bg: float = 0.1
    C: float = 0.3
    Fs: float = 0.85
    G: float = 0.65
    H: float = 0.03
    L_arb: float = 1.0
    L_sol: float = 0.05
    P: float = 1.0
    Ph: float = 2.0
    Q: float = 0.1
    R: float = 3 * 0.33
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

        # Technology / capital share
        aplha = (
            p.aplha0
            + p.aplhaA * safe_div(s.A, s.A + p.A12, p.eps)
            + p.aplhaH * safe_div(s.H, s.H + p.H12, p.eps)
            + p.aplhaQ * safe_div(s.Q, s.Q + p.Q12, p.eps)
        )

        # Employment structure
        s_task_A = math.exp(-p.kQ * s.Q)
        s_task_R = math.exp(-p.kA * s.A - p.kH * s.H)
        PK = p.sK * s.P
        PA = p.s0A * s.P * s_task_A
        PW = p.s0W * s.P * s_task_R
        PU = max(0.0, s.P - PK - PW - PA)
        U = safe_div(PU, max(s.P, p.eps), p.eps)

        # Resource quality and efficiency
        R_frac = max(0.0, safe_div(s.R, p.R0, p.eps))
        qR = p.q_min + (1.0 - p.q_min) * (R_frac ** p.gamma)
        Eff_gain = (
            p.weA * safe_div(s.A, s.A + p.A12, p.eps)
            + p.weH * safe_div(s.H, s.H + p.H12, p.eps)
            + p.wEQ * safe_div(s.Q, s.Q + p.Q12, p.eps)
        )
        eta = max(1.0 + 1e-6, p.eta_min + (p.eta0 - p.eta_min) * (qR ** p.beta) * (1.0 + p.s_eta * Eff_gain))

        # Energy system
        KE = 1.0 + p.kappaA * s.A + p.kappaH * s.H
        Eg = p.e0 * (KE ** p.phi) * qR
        Enet = Eg * (1.0 - 1.0 / eta)
        EA = p.epsA * s.A
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
        Kp = 1.0 + p.kappa_pA * s.A + p.kappa_pH * s.H
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
        Cost_sec = p.csec * Ceff
        Rent = max(0.0, aplha * Y - Cost_maint - Cost_sec)
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
        yyK = Rent / (PK + p.eps)
        yyW = wagePool_net / (PW + PA + p.eps)
        yy_elit = Rent_net / (PK + p.eps)
        yy_elit_norm = safe_div(yy_elit, p.yy_elit0, p.eps)
        Ineq = yyK / (yyW + p.eps)
        Ineq_eff = Ineq / (Ineq + p.Ineq12)

        # Updated higher-level controls
        Lctrl = p.L0 + p.l1 * Gap + p.l2 * Ineq_eff + p.l3 * U + p.l4 * m
        CollapseRisk = 0.4 * 0.5 / (food_pc + 0.5) + 0.4 * 0.4 / (s.G + 0.4) + 0.2 * s.C / (s.C + 0.35)
        SEC_core = SEC / (SEC + p.SEC12)
        b_cm = p.b0 * (1.0 / (1.0 + (min(max(0.0, cm) / p.cb12, 1000.0) ** p.xi_b)))
        d = p.d0 * (1.0 + p.delta_f * p.fd12 / (food_pc + p.fd12) + p.delta_E * p.Ed12 / (Ehouse + p.Ed12) + p.delta_C * Ceff)
        Birth = b_cm * s.P
        Deaths = d * s.P

        welfare_flow = p.ww_cm * cm_norm + p.ww_f * food_pc_norm + p.ww_t * s.T + p.ww_c * (1.0 - s.C) - p.ww_cr * CollapseRisk
        elite_flow = p.wy * yy_elit_norm + p.ws * SEC_core - p.wc * s.C - p.wsec * SEC - p.wr * CollapseRisk

        return {
            "A": s.A,
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
            "Cost_maint": Cost_maint,
            "Cost_sec": Cost_sec,
            "Rent": Rent,
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
        # dA = a["Itot"] * p.thetaA - s.A / p.tauA
        # dH = a["Itot"] * p.thetaH - s.H / p.tauH
        # dQ = a["Itot"] * p.thetaQ - s.Q / p.tauQ
        dA = a["Itot"] * p.thetaA + a["I_priv_A"] - s.A / p.tauA
        dH = a["Itot"] * p.thetaH + a["I_priv_H"] - s.H / p.tauH
        dQ = a["Itot"] * p.thetaQ + a["I_priv_Q"] - s.Q / p.tauQ
        
        dBg = a["TaxK"] + a["TaxW"] - (a["TU"] + a["SVC"] + a["SEC"] + a["Itot"])

        conflict_up = 0.0
        if s.C < 1.0:
            conflict_up = p.rC * (p.b1 * a["Gap"] + p.b2 * p.fc12 / (a["food_pc"] + p.fc12) + p.b3 * a["Ineq_eff"]) * (1.0 - s.C)
        conflict_down = 0.0
        if s.C > 0.0:
            conflict_down = p.dC * s.C * (1.0 + p.mu_sec * a["SEC"] / (a["SEC"] + p.SEC12))
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
                + p.a4 * (p.fc12 / (a["food_pc"] + p.fc12))
                + p.a5 * (1.0 - a["Geff"])
                + p.a6 * a["CollapseRisk"]
            )
        dT = dT_up - dT_down

        dU_elit = a["elite_flow"]
        dU_pop = a["welfare_flow"]

        # Euler update
        s.A += dt * dA
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

def make_noocracy() -> Params:
    return Params()

def make_noocracy_privinv(theta_priv_inv: float = 0.05) -> Params:
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
    p.mmax = 1.157685 #       1.391355
    p.muU = 1.984067 #        1.894645
    p.mu_sec = 0.973617 #     0.872403

    p.sE_ag = 0.465179 #      0.607360
    p.sE_house = 0.095221 #   0.098629
    p.sE_R = 0.408049 #       0.268761
    p.sE_sec = 0.031551 #     0.025251

    p.taxK_rate = 0.795847 #  0.791472
    p.taxW_rate = 0.056322 #  0.071974

    p.theta_sec = 0.045217 #  0.101401
    p.thetaSVC = 0.528415 #   0.502769 
    p.thetaU = 0.336185 #     0.275529
    p.thta_inv = 0.090182 #   0.120301

    return p

def make_noocracy_opt_privinv(theta_priv_inv: float = 0.05) -> Params:
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
    p.mmax = 1.390736
    p.muU = 1.990414
    p.mu_sec = 0.913350

    p.sE_ag = 0.797605
    p.sE_house = 0.069547
    p.sE_R = 0.105823
    p.sE_sec = 0.027023

    p.taxK_rate = 0.050936
    p.taxW_rate = 0.052420

    p.theta_sec = 0.293033
    p.thetaSVC = 0.363909
    p.thetaU = 0.269393
    p.thta_inv = 0.073663

    return p

def make_noocracy_opt_multi_privinv(theta_priv_inv: float = 0.05) -> Params:
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
    p.mmax = 1.20
    p.muU = 1.60
    p.mu_sec = 0.80

    p.dG = 0.03
    p.rG = 0.05
    p.rT = 0.04

    p.sE_ag = 0.30
    p.sE_house = 0.25
    p.sE_R = 0.35
    p.sE_sec = 0.10

    p.taxK_rate = 0.50
    p.taxW_rate = 0.30

    p.theta_sec = 0.15
    p.thetaSVC = 0.25
    p.thetaU = 0.25
    p.thta_inv = 0.35

    return p

def make_world3_reference_privinv(theta_priv_inv: float = 0.05) -> Params:
    p = make_world3_reference()
    p.theta_priv_inv = theta_priv_inv
    return p

def make_cyberpunk() -> Params:
    p = Params()
    p.b3 = 0.35
    p.m0 = 0.85
    p.mmax = 1.35
    p.muU = 1.8
    p.mu_sec = 0.6
    p.dG = 0.03
    p.rG = 0.05
    p.rT = 0.025
    p.sE_ag = 0.22
    p.sE_house = 0.33
    p.sE_R = 0.33
    p.sE_sec = 0.12
    p.taxK_rate = 0.2
    p.taxW_rate = 0.45
    p.theta_sec = 0.30
    p.thetaSVC = 0.10
    p.thetaU = 0.10
    p.thta_inv = 0.50
    return p

def make_cyberpunk_privinv(theta_priv_inv: float = 0.05) -> Params:
    p = make_cyberpunk()
    p.theta_priv_inv = theta_priv_inv
    return p

def make_fortress_elites() -> Params:
    p = make_cyberpunk()
    p.m0 = 0.45
    p.muU = 0.7
    p.mu_sec = 1.2
    p.sE_sec = 0.22
    p.sE_house = 0.26
    p.sE_R = 0.32
    p.sE_ag = 0.2
    p.taxK_rate = 0.15
    p.taxW_rate = 0.2
    p.theta_sec = 0.37
    p.thetaSVC = 0.08
    p.thetaU = 0.05
    p.thta_inv = 0.5
    return p

def make_fortress_elites_privinv(theta_priv_inv: float = 0.05) -> Params:
    p = make_fortress_elites()
    p.theta_priv_inv = theta_priv_inv
    return p

def make_neo_feudalism() -> Params:
    p = Params()
    p.b3 = 0.35
    p.m0 = 0.45
    p.mmax = 0.85
    p.muU = 0.9
    p.mu_sec = 0.9
    p.dG = 0.03
    p.rG = 0.05
    p.rT = 0.025
    p.sE_ag = 0.28
    p.sE_house = 0.2
    p.sE_R = 0.34
    p.sE_sec = 0.18
    p.taxK_rate = 0.25
    p.taxW_rate = 0.6
    p.theta_sec = 0.35
    p.thetaSVC = 0.12
    p.thetaU = 0.08
    p.thta_inv = 0.45
    return p

def make_neo_feudalism_privinv(theta_priv_inv: float = 0.05) -> Params:
    p = make_neo_feudalism()
    p.theta_priv_inv = theta_priv_inv
    return p

def make_techno_communism() -> Params:
    p = Params()
    p.m0 = 0.45
    p.mmax = 0.7
    p.muU = 0.8
    p.mu_sec = 0.6
    p.dG = 0.03
    p.rG = 0.05
    p.rT = 0.025
    p.sE_ag = 0.34
    p.sE_house = 0.12
    p.sE_R = 0.42
    p.sE_sec = 0.12
    p.taxK_rate = 0.65
    p.taxW_rate = 0.5
    p.theta_sec = 0.15
    p.thetaSVC = 0.3
    p.thetaU = 0.2
    p.thta_inv = 0.35
    return p

def make_techno_communism_privinv(theta_priv_inv: float = 0.05) -> Params:
    p = make_techno_communism()
    p.theta_priv_inv = theta_priv_inv
    return p

def make_techno_socialism() -> Params:
    p = Params()
    p.m0 = 0.8
    p.mmax = 1.25
    p.muU = 1.4
    p.mu_sec = 0.4
    p.dG = 0.03
    p.rG = 0.05
    p.rT = 0.025
    p.sE_ag = 0.3
    p.sE_house = 0.3
    p.sE_R = 0.32
    p.sE_sec = 0.08
    p.taxK_rate = 0.6
    p.taxW_rate = 0.6
    p.theta_sec = 0.08
    p.thetaSVC = 0.38
    p.thetaU = 0.22
    p.thta_inv = 0.32
    return p

def make_techno_socialism_privinv(theta_priv_inv: float = 0.05) -> Params:
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

def run_param_grid_monte_carlo(
    regime,
    base_builder,
    grid_param: str,
    grid_values,
    n_per_level=50,
    seed=42,
    world_ranges=None,
):
    rng = random.Random(seed)
    all_rows = []

    if world_ranges is None:
        world_ranges = {
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
        

    run_id = 0
    for val in grid_values:
        for _ in range(n_per_level):
            p = sample_params(base_builder(), world_ranges, rng)
            setattr(p, grid_param, float(val))

            model = RegimeModel(p)
            df = model.run(save_every=1)
            final = df.iloc[-1].to_dict() if pd is not None else df[-1]
            thr = summarize_thresholds(df)

            row = {
                "Regime": regime,
                "run": run_id,
                grid_param: float(val),
                **{k: getattr(p, k) for k in world_ranges.keys()},
                **thr,
                "Y_100": final["Y"],
                "YR_100": final["YR"],
                "P_100": final["P"],
                "G_100": final["G"],
                "T_100": final["T"],
                "food_100": final["food_pc"],
                "U_pop_100": final["U_pop"],
                "U_elit_100": final["U_elit"],
                "Rent_100": final["Rent"],
                "wagePool_100": final["wagePool_net"],
                "P_U_100": final["PU"],
            }
            all_rows.append(row)
            run_id += 1

    if pd is not None:
        return pd.DataFrame(all_rows)
    return all_rows

def run_param_grid2d_monte_carlo(
    regime,
    base_builder,
    grid_param_x: str,
    grid_values_x,
    grid_param_y: str,
    grid_values_y,
    n_per_cell: int = 50,
    seed: int = 42,
    world_ranges=None,
):
    """
    2D parameter sweep + pseudo Monte Carlo over world_ranges.

    Parameters
    ----------
    regime : str
        Regime label for output table.
    base_builder : callable
        Function returning Params(), e.g. make_cyberpunk_privinv
        or lambda: make_cyberpunk_privinv(theta_priv_inv=0.0) if needed.
    grid_param_x : str
        First swept parameter name, e.g. "theta_priv_inv".
    grid_values_x : iterable
        Values for first parameter.
    grid_param_y : str
        Second swept parameter name, e.g. "yyA".
    grid_values_y : iterable
        Values for second parameter.
    n_per_cell : int
        Number of pseudo-MC worlds per grid cell.
    seed : int
        RNG seed.
    world_ranges : dict[str, tuple[float, float]] | None
        Parameter ranges for random world sampling.
        If None, uses the same defaults as run_param_grid_monte_carlo.

    Returns
    -------
    pd.DataFrame | list[dict]
    """
    rng = random.Random(seed)
    all_rows = []

    if world_ranges is None:
        # world_ranges = {
        #     "gamma": (0.55, 0.75),
        #     "beta": (0.90, 1.20),
        #     "kR": (0.08, 0.12),
        #     "Yld0": (28.0, 42.0),
        #     "d_soil": (0.02, 0.04),
        #     "dT": (0.020, 0.035),
        #     "a4": (0.18, 0.32),
        #     "dG": (0.022, 0.038),
        #     "waste": (0.15, 0.25),
        # }
        
        world_ranges = {
            # "R0": (1.5, 4.5),
            "Yld0": (25, 75),
        }

    if grid_param_x == grid_param_y:
        raise ValueError("grid_param_x and grid_param_y must be different.")

    if grid_param_x in world_ranges:
        raise ValueError(
            f"{grid_param_x=} is also present in world_ranges. "
            "Remove it from world_ranges to avoid overwriting."
        )

    if grid_param_y in world_ranges:
        raise ValueError(
            f"{grid_param_y=} is also present in world_ranges. "
            "Remove it from world_ranges to avoid overwriting."
        )

    run_id = 0

    for x_val in grid_values_x:
        for y_val in grid_values_y:
            for _ in range(n_per_cell):
                # sample uncertain world around the base regime
                p = sample_params(base_builder(), world_ranges, rng)

                # apply 2D grid coordinates AFTER random sampling
                setattr(p, grid_param_x, float(x_val))
                setattr(p, grid_param_y, float(y_val))

                model = RegimeModel(p)
                df = model.run(save_every=1)
                final = df.iloc[-1].to_dict() if pd is not None else df[-1]
                thr = summarize_thresholds(df)

                row = {
                    "Regime": regime,
                    "run": run_id,
                    grid_param_x: float(x_val),
                    grid_param_y: float(y_val),
                    **{k: getattr(p, k) for k in world_ranges.keys()},
                    **thr,
                    "Y_100": final["Y"],
                    "YR_100": final["YR"],
                    "P_100": final["P"],
                    "G_100": final["G"],
                    "T_100": final["T"],
                    "food_100": final["food_pc"],
                    "U_pop_100": final["U_pop"],
                    "U_elit_100": final["U_elit"],
                    "Rent_100": final["Rent"],
                    "wagePool_100": final["wagePool_net"],
                    "P_U_100": final["PU"],
                    # optional diagnostics that are often useful for heatmaps / debugging
                    "cm_100": final["cm"],
                    "Gap_100": final["Gap"],
                    "A_100": final["A"],
                    "H_100": final["H"],
                    "Q_100": final["Q"],
                }

                all_rows.append(row)
                run_id += 1

    if pd is not None:
        return pd.DataFrame(all_rows)
    return all_rows

def build_2d_heatmap_summary(
    mc_runs,
    x_col: str,
    y_col: str,
    value_cols=None,
):
    """
    Aggregate 2D MC results into mean/std tables for heatmaps.

    Returns dict of pivot tables:
        {
            "U_pop_100_mean": ...,
            "U_pop_100_std": ...,
            ...
        }
    """
    if pd is None:
        raise RuntimeError("build_2d_heatmap_summary requires pandas.")

    if value_cols is None:
        value_cols = [
            "U_pop_100",
            "U_elit_100",
            "Y_100",
            "Rent_100",
            "wagePool_100",
            "P_U_100",
        ]

    out = {}

    grouped = mc_runs.groupby([y_col, x_col], sort=False)

    for col in value_cols:
        if col not in mc_runs.columns:
            continue

        mean_df = grouped[col].mean().reset_index()
        std_df = grouped[col].std(ddof=1).reset_index()

        out[f"{col}_mean"] = mean_df.pivot(index=y_col, columns=x_col, values=col)
        out[f"{col}_std"] = std_df.pivot(index=y_col, columns=x_col, values=col)

    return out

def build_mc_publication_summary(
    mc_runs,
    horizon: float = 100.0,
    regime_col: str = "Regime",
    run_col: str = "run",
    group_col=None,
    save_excel: Optional[str] = None,
):
    """
    Universal post-processing for:
    - regime Monte Carlo (group_col="Regime")
    - sensitivity/grid sweeps (group_col="theta_priv_inv")
    - combined grouping (group_col=["Regime", "theta_priv_inv"])

    Parameters
    ----------
    mc_runs : pd.DataFrame
        DataFrame returned by Monte Carlo / grid Monte Carlo.
    horizon : float
        Simulation horizon used for right-censoring threshold metrics.
    regime_col : str
        Default regime column name.
    run_col : str
        Column identifying Monte Carlo world/run id.
    group_col : None | str | list[str]
        Grouping key(s). If None:
        - use regime_col if present
        - otherwise raise error.
    save_excel : Optional[str]
        If provided, save all summary tables to an Excel workbook.
    """
    if pd is None:
        raise RuntimeError("build_mc_publication_summary requires pandas.")

    df = mc_runs.copy()

    # -----------------------------
    # Resolve grouping
    # -----------------------------
    if group_col is None:
        if regime_col in df.columns:
            group_cols = [regime_col]
        else:
            raise KeyError(
                "No group_col provided and default regime_col not found. "
                "Pass group_col='theta_priv_inv' or another column explicitly."
            )
    elif isinstance(group_col, str):
        group_cols = [group_col]
    else:
        group_cols = list(group_col)

    for col in group_cols + [run_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in mc_runs.")

    # pretty label for outputs
    group_label = "__".join(group_cols)

    # -----------------------------
    # Config
    # -----------------------------
    final_metrics = [
        "Y_100",
        "YR_100",          # keep compatibility with your current MC outputs
        "Y_R_100",         # and with other variants
        "P_100",
        "G_100",
        "T_100",
        "food_100",        # compatibility
        "food_pc_100",     # compatibility
        "U_pop_100",
        "U_elit_100",
        "Rent_100",
        "wagePool_100",
        "P_U_100",
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

    final_metrics = [c for c in final_metrics if c in df.columns]
    threshold_metrics = [c for c in threshold_metrics if c in df.columns]

    # normalize aliases for convenience
    rename_map = {}
    if "YR_100" in df.columns and "Y_R_100" not in df.columns:
        rename_map["YR_100"] = "Y_R_100"
    if "food_100" in df.columns and "food_pc_100" not in df.columns:
        rename_map["food_100"] = "food_pc_100"

    if rename_map:
        df = df.rename(columns=rename_map)
        final_metrics = [rename_map.get(c, c) for c in final_metrics]

    # de-duplicate after alias normalization
    final_metrics = list(dict.fromkeys(final_metrics))

    # -----------------------------
    # Threshold post-processing
    # -----------------------------
    for col in threshold_metrics:
        df[f"{col}_crossed"] = (~df[col].isna()) & (~df[col].isin([math.inf, float("inf")]))
        df[f"{col}_not_crossed"] = ~df[f"{col}_crossed"]
        df[f"{col}_cens"] = df[col].replace([math.inf, float("inf")], horizon).fillna(horizon)
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

    # sort=False to preserve natural order (important for theta grid)
    grouped = df.groupby(group_cols, sort=False)

    # -----------------------------
    # Main summary by group
    # -----------------------------
    summary_rows = []

    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        row = {col: val for col, val in zip(group_cols, key)}
        row["n_runs"] = int(len(g))

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
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        row = {col: val for col, val in zip(group_cols, key)}
        row["n_runs"] = int(len(g))

        for col in threshold_metrics:
            row[f"{col}_median_cens"] = float(g[f"{col}_cens"].median())
            row[f"{col}_mean_cens"] = float(g[f"{col}_cens"].mean())
            row[f"{col}_q25_cens"] = float(g[f"{col}_cens"].quantile(0.25))
            row[f"{col}_q75_cens"] = float(g[f"{col}_cens"].quantile(0.75))
            row[f"{col}_not_crossed_share"] = float(g[f"{col}_not_crossed"].mean())

        thr_rows.append(row)

    thresholds_summary = pd.DataFrame(thr_rows)

    # -----------------------------
    # Delta vs baseline for single grid param
    # -----------------------------
    delta_table = None
    baseline_value = None

    if len(group_cols) == 1 and group_cols[0] != regime_col:
        gcol = group_cols[0]
        vals = list(summary_main[gcol])

        # choose baseline:
        # 1) exact 0 if present
        # 2) else min value
        if any(float(v) == 0.0 for v in vals if pd.notna(v)):
            baseline_value = next(v for v in vals if pd.notna(v) and float(v) == 0.0)
        else:
            baseline_value = min(vals)

        baseline_row = summary_main.loc[summary_main[gcol] == baseline_value].iloc[0]
        delta_table = summary_main.copy()

        for metric in [
            "U_pop_100",
            "U_elit_100",
            "Y_100",
            "Y_R_100",
            "P_100",
            "G_100",
            "T_100",
            "food_pc_100",
            "Rent_100",
            "wagePool_100",
            "P_U_100",
        ]:
            mean_col = f"{metric}_mean"
            if mean_col in delta_table.columns and mean_col in baseline_row.index:
                delta_table[f"{metric}_delta_vs_baseline"] = (
                    delta_table[mean_col] - float(baseline_row[mean_col])
                )

    # -----------------------------
    # Win shares and average ranks
    # Only meaningful when comparing regimes within the same run
    # -----------------------------
    wins_df = pd.DataFrame()
    ranks_df = pd.DataFrame()
    wins_wide = pd.DataFrame()
    ranks_wide = pd.DataFrame()

    can_rank = regime_col in df.columns and regime_col not in group_cols

    if can_rank:
        higher_is_better = [
            c for c in [
                "Y_R_100", "P_100", "G_100", "T_100", "food_pc_100",
                "U_pop_100", "U_elit_100", "Rent_100", "wagePool_100"
            ]
            if c in df.columns
        ]
        # lower unemployment is better
        lower_is_better = [c for c in ["P_U_100"] if c in df.columns]

        higher_is_better += [f"{c}_cens" for c in threshold_metrics]

        win_records = []
        rank_records = []

        # if extra grouping columns exist, compare regimes within each run + those grouping columns
        compare_keys = [run_col] + [c for c in group_cols if c != regime_col]

        for metric in higher_is_better:
            tmp = df[compare_keys + [regime_col, metric]].copy()
            tmp["rank"] = tmp.groupby(compare_keys)[metric].rank(method="average", ascending=False)
            tmp["is_win"] = tmp.groupby(compare_keys)[metric].transform(lambda x: x == x.max())

            wins = tmp.groupby(regime_col)["is_win"].mean().reset_index()
            wins["metric"] = metric
            wins = wins.rename(columns={"is_win": "win_share"})
            win_records.append(wins)

            ranks = tmp.groupby(regime_col)["rank"].mean().reset_index()
            ranks["metric"] = metric
            ranks = ranks.rename(columns={"rank": "avg_rank"})
            rank_records.append(ranks)

        for metric in lower_is_better:
            tmp = df[compare_keys + [regime_col, metric]].copy()
            tmp["rank"] = tmp.groupby(compare_keys)[metric].rank(method="average", ascending=True)
            tmp["is_win"] = tmp.groupby(compare_keys)[metric].transform(lambda x: x == x.min())

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

        wins_wide = wins_df.pivot(index=regime_col, columns="metric", values="win_share").reset_index() if not wins_df.empty else wins_df
        ranks_wide = ranks_df.pivot(index=regime_col, columns="metric", values="avg_rank").reset_index() if not ranks_df.empty else ranks_df

    # -----------------------------
    # Compact publication table
    # -----------------------------
    publication_table = summary_main.copy()

    keep_cols = group_cols + ["n_runs"]

    for col in [
        "G_100", "U_pop_100", "U_elit_100", "Y_R_100", "food_pc_100",
        "P_100", "Rent_100", "wagePool_100", "P_U_100"
    ]:
        for suffix in ["mean", "std", "median", "q25", "q75"]:
            c = f"{col}_{suffix}"
            if c in publication_table.columns:
                keep_cols.append(c)

    publication_table = publication_table[[c for c in keep_cols if c in publication_table.columns]].copy()

    for col in ["T_G35", "T_F20", "T_P80", "T_Y20"]:
        if col in threshold_metrics:
            merge_cols = group_cols + [
                f"{col}_median_cens",
                f"{col}_q25_cens",
                f"{col}_q75_cens",
                f"{col}_not_crossed_share",
            ]
            publication_table = publication_table.merge(
                thresholds_summary[merge_cols],
                on=group_cols,
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

            if delta_table is not None:
                delta_table.to_excel(writer, sheet_name="delta_vs_baseline", index=False)

            if isinstance(wins_df, pd.DataFrame) and not wins_df.empty:
                wins_df.to_excel(writer, sheet_name="wins_long", index=False)
            if isinstance(ranks_df, pd.DataFrame) and not ranks_df.empty:
                ranks_df.to_excel(writer, sheet_name="ranks_long", index=False)
            if isinstance(wins_wide, pd.DataFrame) and not wins_wide.empty:
                wins_wide.to_excel(writer, sheet_name="wins_wide", index=False)
            if isinstance(ranks_wide, pd.DataFrame) and not ranks_wide.empty:
                ranks_wide.to_excel(writer, sheet_name="ranks_wide", index=False)

    return {
        "mc_enriched": df,
        "summary_main": summary_main,
        "thresholds_summary": thresholds_summary,
        "publication_table": publication_table,
        "delta_vs_baseline": delta_table,
        "wins": wins_df,
        "ranks": ranks_df,
        "wins_wide": wins_wide,
        "ranks_wide": ranks_wide,
        "group_cols": group_cols,
        "baseline_value": baseline_value,
    }

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_heatmap_from_pivot(
    pivot_mean,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "viridis",
    fmt: str = ".2f",
    pivot_std=None,
    annotate: bool = True,
    annotate_std: bool = False,
    colorbar_label: Optional[str] = None,
    figsize=(8, 5),
    outpath: Optional[str] = None,
    dpi: int = 160,
):
    """
    Plot a heatmap from a pivot table.

    pivot_mean : pd.DataFrame
        Pivot table with index=y values and columns=x values.
    """

    data = pivot_mean.values.astype(float)
    y_labels = list(pivot_mean.index)
    x_labels = list(pivot_mean.columns)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    im = ax.imshow(
        data,
        aspect="auto",
        origin="upper",
        cmap=cmap,
    )

    # ticks
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))

    ax.set_xticklabels([str(x) for x in x_labels])
    ax.set_yticklabels([str(y) for y in y_labels])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # colorbar
    cbar = plt.colorbar(im, ax=ax)

    if colorbar_label:
        cbar.set_label(colorbar_label)

    # grid between cells
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)

    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)

    ax.tick_params(which="minor", bottom=False, left=False)

    # annotations
    if annotate:

        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        threshold = data_min + 0.5 * (data_max - data_min)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):

                val = data[i, j]

                if math.isnan(val):
                    txt = "NA"
                else:
                    txt = format(val, fmt)

                if annotate_std and pivot_std is not None:
                    std_val = float(pivot_std.iloc[i, j])
                    if not math.isnan(std_val):
                        txt = f"{txt}\n±{std_val:{fmt}}"

                color = "white" if (not math.isnan(val) and val < threshold) else "black"

                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

    plt.tight_layout()

    if outpath:
        outpath = str(outpath)
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, bbox_inches="tight", dpi=dpi)

    plt.show()
    plt.close(fig)
    
def plot_heatmap_metric(
    heatmaps: dict,
    metric: str,
    x_label: str,
    y_label: str,
    title: Optional[str] = None,
    cmap: str = "viridis",
    annotate: bool = True,
    annotate_std: bool = False,
    outpath: Optional[str] = None,
):
    """
    Plot one metric from build_2d_heatmap_summary output.

    Example metric:
        "U_pop_100"
        "U_elit_100"
    """

    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"

    if mean_key not in heatmaps:
        raise KeyError(f"{mean_key} not found in heatmaps.")

    pivot_mean = heatmaps[mean_key]
    pivot_std = heatmaps.get(std_key, None)

    if title is None:
        title = f"{metric} heatmap"

    plot_heatmap_from_pivot(
        pivot_mean=pivot_mean,
        pivot_std=pivot_std,
        title=title,
        xlabel=x_label,
        ylabel=y_label,
        cmap=cmap,
        annotate=annotate,
        annotate_std=annotate_std,
        colorbar_label=metric,
        outpath=outpath,
    )    
    
# ------------------------------------------------------------
# Example
# ------------------------------------------------------------

if __name__ == "__main__":
    if pd is not None:
        # outputs, summary_df, thresholds_df = compare_regimes(
        #     sample_years=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        #     save_excel="regime_comparison.xlsx",
        # )
        # print(summary_df)
        # print(thresholds_df)
        # print("Saved: regime_comparison.xlsx")

        # mc = run_world_monte_carlo(n=100, seed=123, save_excel="world_monte_carlo.xlsx")
        # print(mc.head())
        # print("Saved: world_monte_carlo.xlsx")
        
        # pub = build_mc_publication_summary(
        #     mc,
        #     horizon=100.0,
        #     save_excel="world_monte_carlo_publication.xlsx",
        # )
        
        theta_grid = [0.00, 0.03, 0.05, 0.10, 0.20]
        yyA_grid = [0.225, 0.30, 0.375, 0.45, 0.525, 0.60, 0.675]
        yyR_grid = [0.5, 0.667, 0.834, 1.0, 1.167, 1.334, 1.5]
        cA_grid = [0.0015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
        R0_grid = [1.5, 2, 2.5, 3, 3.5, 4, 4.5]
        
        for name, builder in {
            "Cyberpunk": make_cyberpunk_privinv,
            "Techno-communism": make_techno_communism_privinv,
            "Techno-socialism": make_techno_socialism_privinv,
            "Noocracy-opt": make_noocracy_opt_privinv,
        }.items():

            # df = run_param_grid_monte_carlo(
            #     regime=name,
            #     base_builder=builder,
            #     grid_param="theta_priv_inv",
            #     grid_values=(0.00, 0.03, 0.05, 0.10, 0.20),
            #     n_per_level=100,
            #     seed=123,
            # )
            
            df_2d = run_param_grid2d_monte_carlo(
                regime=name,
                base_builder=builder,
                grid_param_x="theta_priv_inv",
                grid_values_x=theta_grid,
                grid_param_y="R0",
                grid_values_y=R0_grid,
                n_per_cell=50,
                seed=42,
            )
        
            # df_2d.to_excel(f"theta_priv_sweep_{name}_2d.xlsx", index=False)
            
            
            # pub = build_mc_publication_summary(
            #     df_2d,
            #     group_col="theta_priv_inv",
            #     horizon=100.0,
            #     save_excel=f"world_monte_carlo_publication_{name}_2d.xlsx",
            # )
            
            heatmaps = build_2d_heatmap_summary(
                df_2d,
                x_col="theta_priv_inv",
                y_col="R0",
                value_cols=["U_pop_100", "U_elit_100"],
            )
            
            plot_heatmap_metric(
                heatmaps,
                metric="U_pop_100",
                x_label="theta_priv_inv",
                y_label="R0",
                title=f"{name}: U_pop at year 100",
                annotate=True,
                outpath=f"outputs/heatmap_upop_{name}.png",
            )
            
            plot_heatmap_metric(
                heatmaps,
                metric="U_elit_100",
                x_label="theta_priv_inv",
                y_label="R0",
                title=f"{name}: U_elit at year 100",
                annotate=True,
                outpath=f"outputs/heatmap_uelit_{name}.png",
            )
        
    else:
        outputs, summary_rows, threshold_rows = compare_regimes()
        print(summary_rows)
        print(threshold_rows)
