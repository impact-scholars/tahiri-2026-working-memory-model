from Working_Memory_Model import run_wm_combined, DEFAULT_CFG
from brian2 import Hz

cfg = dict(DEFAULT_CFG)
condition = {"control":   dict(nmda_scale=1.1, gaba_scale=0.2),}
cfg.update(
    condition="control",   # change this
    #delay=1500,
    Jp=1.84,
    cue_rate=200*Hz,
    plot=True
)

conditions = ["control", "-NMDA", "+NMDA"]
for condition in conditions:
    cfg["condition"] = condition
    run_wm_combined(cfg, plot=True)

