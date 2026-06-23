"""Ta-versus-Tb input experiment.

The literature review found the operational Basist code embeds Grody and Basist
(1996) antenna-temperature thresholds, so the algorithm likely expects antenna
temperature (Ta/TDR). We have been feeding the CSU brightness-temperature CDR
(Tb/SDR). As a first-order test, convert our Tb to an Ta-equivalent by subtracting
Grody's per-channel antenna-to-brightness corrections, re-run the engine, and
compare the product and the SWAMPS inundation skill against the Tb baseline.

    python -m scripts.experiment_ta_vs_tb MONTH_DIR SWAMPS_FW.nc

Both runs omit the 85-to-91 calibration so the comparison isolates the Ta-vs-Tb
convention shift.
"""

import sys

import numpy as np

from swi import monthly, core_numpy, validate as val
from swi.channels import N_CHANNELS

# Grody and Basist (1996, p.239) antenna-to-brightness corrections (K), added to
# Ta to get Tb; here subtracted from Tb to approximate Ta. Channel order
# 19V 19H 22V 37V 37H 85V(91) 85H(91).
GRODY_OFFSET = np.array([7.0, 7.0, 6.0, 4.0, 4.0, 3.0, 3.0])


class _Engine:
    def __init__(self, ta_equiv):
        self.ta_equiv = ta_equiv

    def evaluate_kelvin(self, tb):
        x = tb - GRODY_OFFSET if self.ta_equiv else tb
        return core_numpy.evaluate_kelvin(x)


def composite_dsc(files, ta_equiv):
    comp = monthly.composite(files, engine=_Engine(ta_equiv), pass_list=("dsc",))
    return comp["lat"], comp["lon"], comp["by_pass"]["dsc"]


def swamps_skill(lat, lon, fields, swamps):
    from scripts.validate_swamps import load_swamps_fw
    wet = fields["wetness_index_mean"]
    snowf = fields["snow_frequency"]
    slat, slon, fw = load_swamps_fw(swamps)
    fw_on = val.regrid_nearest(slat, slon, fw, lat, lon)
    m = np.isfinite(fw_on) & (wet >= 0) & ~(snowf > 0.5)
    W, F = wet[m], fw_on[m]
    dc = val.detection_contrast(W, F, thr=0.0)
    s = val.skill_scores(W, F)
    hi = F > 0.10
    inund = 100 * (W[hi] > 0).mean() if hi.sum() > 100 else np.nan
    return {"n": s["n"], "spearman": s["spearman_r"], "ratio": dc["ratio"],
            "n_wet": dc["n_hi"], "inund_detect": inund,
            "mean_wet": float(np.nanmean(W[W > 0])) if (W > 0).any() else 0.0,
            "wet_frac": float((W > 0).mean())}


def main():
    import glob
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    files = sorted(glob.glob(f"{sys.argv[1]}/*.nc"))
    swamps = sys.argv[2]
    print(f"compositing {len(files)} days two ways (Tb baseline, Ta-equivalent)...")

    with np.errstate(divide="ignore", invalid="ignore"):
        lat, lon, tb_fields = composite_dsc(files, ta_equiv=False)
        _, _, ta_fields = composite_dsc(files, ta_equiv=True)
        tb = swamps_skill(lat, lon, tb_fields, swamps)
        ta = swamps_skill(lat, lon, ta_fields, swamps)

    print(f"\n{'metric':<22}{'Tb (SDR)':>14}{'Ta-equiv (TDR)':>16}")
    rows = [("SWAMPS Spearman r", "spearman", "{:+.3f}"),
            ("surface-water ratio", "ratio", "{:.2f}x"),
            ("inundation detect %", "inund_detect", "{:.0f}%"),
            ("WET>0 fraction", "wet_frac", "{:.3f}"),
            ("mean WET (>0)", "mean_wet", "{:.1f}"),
            ("n co-located", "n", "{:,}")]
    for label, key, fmt in rows:
        print(f"{label:<22}{fmt.format(tb[key]):>14}{fmt.format(ta[key]):>16}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
