# Changelog

## 0.1.0 — GitHub publication cleanup

Repository packaging performed around the two supplied research scripts.

Code-level cleanup was intentionally conservative:

- renamed the supplied files to stable repository filenames;
- clarified that TTA averages logits before softmax;
- documented `--resize-size` as a legacy compatibility argument because the current transform resizes directly to `--image-size`;
- corrected human-readable M1/M2/M4/M5 morphology labels so they match their implemented identity-plus-operator definitions;
- made the ablation script's `verify-model` check tolerant of architecture variants that intentionally omit a gate map;
- did **not** alter the primary M3 training objective, morphology math, optimizer, EMA, data split logic, metrics, or external-evaluation logic.
