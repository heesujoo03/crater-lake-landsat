# -*- coding: utf-8 -*-
"""Assemble the robustness summary table from the analysis outputs in
results/11_analysis/.

    python -m pipeline.analysis.robustness_table
"""
import numpy as np
import pandas as pd
from pipeline.config import settings as S

R = S.ROOT / "results" / "11_analysis"
ROWS = [("taal", "2020-01-12", 365), ("taal", "2021-07-01", 365), ("taal", "2022-10-05", 365),
        ("taal", "2024-04-12", 365), ("santa_ana", "2005-06-16", 180),
        ("santa_ana", "2005-06-16", 365), ("ruapehu", "2007-09-25", 180)]
NAME = {"taal": "Taal", "santa_ana": "Santa Ana", "ruapehu": "Ruapehu"}


def fmt(d, exc, tested=True):
    if not tested or d is None or not np.isfinite(d):
        return "n.t."
    return f"{d:+.1f}{'*' if exc else ''}"


def main():
    bs = pd.read_csv(R / "water_fraction/boundary_sensitivity.csv")
    bs = bs[bs.f_min.eq(0.7)]
    w = pd.read_csv(R / "unmixing_psf/event_contrasts_by_scenario_wide.csv")
    m = pd.read_csv(R / "unmixing_psf/event_maxrank_by_scenario.csv")
    cc = pd.read_csv(R / "reference_model/corrected_contrasts.csv")
    bc = pd.read_csv(R / "reference_model/baseline_comparison.csv")
    rs = pd.read_csv(R / "placebo_design/reanalysis_summary.csv")
    rs = rs[rs.analysis.eq("post")]
    out = []
    for s, o, W in ROWS:
        k = lambda df: df[(df.site == s) & (df.onset == o) & (df.window_d == W)]
        base = k(bs[bs.scenario.eq("baseline")]).iloc[0]
        r = {"Event": f"{NAME[s]} {o[:7]}, {W} d",
             "Reported": f"{base.delta:+.1f} ({base.p_placebo_year:.3f}; {int(base.placebo_n_years)})"}
        ww = k(w[w.estimator.eq("dT_syn_weather")]).iloc[0]
        mm = k(m[m.estimator.eq("dT_syn_weather")]).iloc[0]
        r["No unmixing"] = fmt(ww.no_unmix, mm.no_unmix)
        sc = ["native_box_scaled", "native_box_scaled_x0.7", "native_box_scaled_x1.3",
              "gauss_fwhm_native", "common_tm_worst"]
        vals = [ww[c] for c in sc]
        allexc = all(bool(mm[c]) for c in sc)
        lo, hi = f"{min(vals):+.1f}", f"{max(vals):+.1f}"
        rng = lo if lo == hi else f"{lo} to {hi}"
        # annual Santa Ana becomes untestable under x1.3 (fewer than five years)
        r["Instrument-specific PSF"] = rng + ("*" if allexc else " (*/n.t.)")
        sf = k(bs[bs.scenario.eq("same_scene_landsat_fraction")]).iloc[0]
        tested = sf.placebo_n_years >= 5 and np.isfinite(sf.delta)
        exc = tested and abs(sf.p_placebo_year - 1 / (sf.placebo_n_years + 1)) < 2e-4
        r["Same-scene water fraction"] = fmt(sf.delta, exc, tested)
        c2 = k(cc[cc.design.str.contains("C2|leave", case=False, regex=True)])
        c2 = c2.iloc[0]
        r["Leave-window-out refit"] = fmt(c2.delta, c2.max_exceed)
        sb = k(bc[(bc.fit == "published_fit") & (bc.estimator == "seasonal_baseline")]).iloc[0]
        r["Seasonal baseline"] = fmt(sb.delta, sb.max_exceed)
        q = k(rs).iloc[0]
        r["Same-date collapsed"] = fmt(q.collapse_delta, q.collapse_max_exceedance == True)
        t2 = q.tol2_both_placebo_n_years >= 5
        r["Joint count matching ±2"] = ("n.t." if not t2 else
                                       f"{q.delta:+.1f}{'*' if q.tol2_both_max_exceedance == True else ''}")
        r["Leave-one-date-out range"] = (f"{q.lodo_delta_min:+.1f} to {q.lodo_delta_max:+.1f} "
                                         f"({int(q.lodo_n_keep_max_exceedance)}/{int(q.lodo_n_variants)})")
        out.append(r)
    d = pd.DataFrame(out)
    for c in d.columns[1:]:
        d[c] = d[c].str.replace("-", "−", regex=False)
    p = R / "synthesis" / "table8_robustness.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(p, index=False, encoding="utf-8-sig")
    print(d.to_string())
    print("designs in corrected_contrasts:", cc.design.unique())


if __name__ == "__main__":
    main()
