# External validation — field photos (inference only)

**Date:** 2026-10-17  
**Models:** router_v1_mobilenet_v3_small, leaf_v1_efficientnet_b0, fruit_v1_yolov8n  
**Data:** Google Drive validation folder (`data/external_validation/`, not in git). Validation only — nothing was trained or tuned on it. No disease labels, so accuracy cannot be computed; this report shows what the pipeline did.

**Coverage:** 62 of the 109 Drive photos (33 of 80 `fruit pics`, all 29 `Article related_other pics`). The other 47 could not be downloaded because Google Drive throttled anonymous downloads. To finish: download the folder as a zip in a browser, unzip into `data/external_validation/` (keep the two folder names) and rerun `python scripts/eval_external_validation.py`.

## `Article related_other pics` — expected: out_of_scope (n=29)

- Routed: {'fruit': 28, 'leaf': 1} (median router confidence 0.9963)
- Predicted (primary) class: {'No fruit detected': 17, 'Healthy_Tomato': 7, 'Mold': 4, 'Septoria leaf spot': 1}
- Median confidence 0.0; 24 below 0.5
- These photos are not tomato close-ups; the pipeline has no 'not a tomato' output, so every one of them still receives a leaf or fruit diagnosis.

## `fruit pics` — expected: fruit (n=33)

- Routed: {'fruit': 33} (median router confidence 1.0)
- Routed to fruit: **100.0%**
- Fruit images with no YOLO detection: 1
- Detected box classes: {'Healthy_Tomato': 80, 'Catfaced': 2, 'Blossom_End_Rot': 2, 'Mold': 1, 'Spotted_Wilt_Virus': 1}
- Predicted (primary) class: {'Healthy_Tomato': 28, 'Catfaced': 2, 'No fruit detected': 1, 'Mold': 1, 'Spotted_Wilt_Virus': 1}
- Median confidence 0.8468; 7 below 0.5

## Per image

| Folder | Image | Routed (conf) | Class | Conf | Severity | Boxes |
|---|---|---|---|---:|---:|---|
| Article | IMG-20261011-WA0031.jpg | fruit (1.00) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG-20261011-WA0032.jpg | fruit (1.00) | Healthy_Tomato | 0.70 | 0.0 | Healthy_Tomato, Healthy_Tomato |
| Article | IMG-20261011-WA0033.jpg | fruit (0.89) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG-20261011-WA0034.jpg | fruit (1.00) | Mold | 0.27 | 11.6 | Mold |
| Article | IMG-20261011-WA0035.jpg | fruit (1.00) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG-20261011-WA0036.jpg | fruit (1.00) | Mold | 0.37 | 20.6 | Mold |
| Article | IMG-20261011-WA0037.jpg | fruit (1.00) | Mold | 0.45 | 100.0 | Mold, Mold |
| Article | IMG-20261011-WA0042.jpg | fruit (1.00) | Healthy_Tomato | 0.46 | 0.0 | Healthy_Tomato, Healthy_Tomato |
| Article | IMG-20261011-WA0043.jpg | fruit (1.00) | Healthy_Tomato | 0.51 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| Article | IMG-20261011-WA0044.jpg | fruit (1.00) | Healthy_Tomato | 0.59 | 0.0 | Healthy_Tomato, Healthy_Tomato |
| Article | IMG-20261011-WA0045.jpg | fruit (1.00) | Healthy_Tomato | 0.47 | 0.0 | Healthy_Tomato |
| Article | IMG-20261011-WA0046.jpg | fruit (1.00) | Healthy_Tomato | 0.35 | 0.0 | Healthy_Tomato, Healthy_Tomato |
| Article | IMG-20261011-WA0047.jpg | fruit (0.99) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG-20261011-WA0049.jpg | fruit (0.87) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG-20261011-WA0051.jpg | fruit (0.98) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG-20261011-WA0052.jpg | fruit (0.99) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG-20261011-WA0053.jpg | fruit (0.99) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG-20261011-WA0057.jpg | fruit (0.95) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG-20261011-WA0059.jpg | fruit (0.76) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG-20261011-WA0062.jpg | fruit (0.81) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG-20261011-WA0069.jpg | fruit (0.64) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG-20261011-WA0102.jpg | fruit (1.00) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG_20260927_152750.jpg | fruit (0.99) | Mold | 0.54 | 80.5 | Mold |
| Article | IMG_20260927_152752.jpg | fruit (1.00) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG_20260927_152756.jpg | fruit (1.00) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG_20260927_152856.jpg | fruit (1.00) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG_20260927_152858.jpg | fruit (0.97) | No fruit detected | 0.00 | 0.0 | — |
| Article | IMG_20260927_155901.jpg | leaf (0.57) | Septoria leaf spot | 0.53 | 32.1 | — |
| Article | IMG_20260927_155956.jpg | fruit (1.00) | Healthy_Tomato | 0.44 | 0.0 | Healthy_Tomato |
| fruit | 20260927_153024.jpg | fruit (1.00) | Healthy_Tomato | 0.91 | 0.0 | Healthy_Tomato, Healthy_Tomato |
| fruit | 20260927_153037.jpg | fruit (1.00) | Healthy_Tomato | 0.42 | 0.0 | Healthy_Tomato |
| fruit | 20260927_153043.jpg | fruit (1.00) | No fruit detected | 0.00 | 0.0 | — |
| fruit | 20260927_153059.jpg | fruit (1.00) | Healthy_Tomato | 0.82 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | 20260927_153102.jpg | fruit (1.00) | Mold | 0.28 | 2.9 | Mold |
| fruit | 20260927_153210.jpg | fruit (1.00) | Healthy_Tomato | 0.88 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | 20260927_153414.jpg | fruit (1.00) | Healthy_Tomato | 0.72 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | 20260927_153504.jpg | fruit (1.00) | Healthy_Tomato | 0.90 | 0.0 | Healthy_Tomato, Healthy_Tomato |
| fruit | 20260927_153507.jpg | fruit (1.00) | Healthy_Tomato | 0.84 | 0.0 | Healthy_Tomato, Healthy_Tomato |
| fruit | IMG-20261011-WA0014.jpg | fruit (1.00) | Healthy_Tomato | 0.41 | 0.0 | Healthy_Tomato |
| fruit | IMG-20261011-WA0016.jpg | fruit (1.00) | Healthy_Tomato | 0.94 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | IMG-20261011-WA0024.jpg | fruit (1.00) | Healthy_Tomato | 0.65 | 0.0 | Healthy_Tomato, Healthy_Tomato |
| fruit | IMG-20261011-WA0025.jpg | fruit (1.00) | Healthy_Tomato | 0.58 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | IMG-20261011-WA0026.jpg | fruit (1.00) | Healthy_Tomato | 0.94 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | IMG-20261011-WA0028.jpg | fruit (1.00) | Healthy_Tomato | 0.81 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | IMG-20261011-WA0029.jpg | fruit (1.00) | Healthy_Tomato | 0.87 | 0.0 | Healthy_Tomato |
| fruit | IMG-20261011-WA0038.jpg | fruit (1.00) | Healthy_Tomato | 0.90 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | IMG-20261011-WA0039.jpg | fruit (1.00) | Catfaced | 0.62 | 7.8 | Catfaced |
| fruit | IMG-20261011-WA0040.jpg | fruit (1.00) | Spotted_Wilt_Virus | 0.41 | 7.5 | Spotted_Wilt_Virus |
| fruit | IMG-20261011-WA0041.jpg | fruit (1.00) | Catfaced | 0.37 | 24.8 | Catfaced |
| fruit | IMG-20261011-WA0054.jpg | fruit (1.00) | Healthy_Tomato | 0.90 | 0.0 | Healthy_Tomato |
| fruit | IMG-20261011-WA0055.jpg | fruit (1.00) | Healthy_Tomato | 0.86 | 0.0 | Healthy_Tomato |
| fruit | IMG-20261011-WA0056.jpg | fruit (1.00) | Healthy_Tomato | 0.65 | 0.0 | Healthy_Tomato |
| fruit | IMG-20261011-WA0060.jpg | fruit (1.00) | Healthy_Tomato | 0.45 | 52.6 | Healthy_Tomato, Blossom_End_Rot, Blossom_End_Rot |
| fruit | IMG-20261011-WA0063.jpg | fruit (1.00) | Healthy_Tomato | 0.88 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | IMG-20261011-WA0064.jpg | fruit (1.00) | Healthy_Tomato | 0.93 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | IMG-20261011-WA0065.jpg | fruit (1.00) | Healthy_Tomato | 0.91 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | IMG-20261011-WA0066.jpg | fruit (1.00) | Healthy_Tomato | 0.59 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | IMG-20261011-WA0067.jpg | fruit (1.00) | Healthy_Tomato | 0.92 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | IMG-20261011-WA0070.jpg | fruit (1.00) | Healthy_Tomato | 0.91 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | IMG-20261011-WA0072.jpg | fruit (1.00) | Healthy_Tomato | 0.85 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
| fruit | IMG-20261011-WA0073.jpg | fruit (1.00) | Healthy_Tomato | 0.95 | 0.0 | Healthy_Tomato |
| fruit | IMG-20261011-WA0074.jpg | fruit (1.00) | Healthy_Tomato | 0.90 | 0.0 | Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato, Healthy_Tomato |
