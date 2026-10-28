Results over 5 seeds: mean ± standard deviation across seeds, each metric computed within a seed.

### Negative controls (base model, nothing planted)

| Detector | AUROC GSM8K test vs train [90% CI] | within 0.5 ± 0.05 |
|---|---|---|
| Loss | 0.538 [0.517, 0.558] | no |
| Zlib | 0.525 [0.505, 0.544] | yes |
| Min-K% | 0.537 [0.517, 0.557] | no |
| Min-K%++ | 0.532 [0.512, 0.552] | no |
| Ref (70m) | 0.511 [0.490, 0.531] | yes |

AUROC of each group-to-be-planted vs its unseen comparison, before any training [90% CI; * = interval not inside 0.5 ± 0.05]:

| Detector | k=1 | k=2 | k=4 | k=8 | k=16 | template originals |
|---|---|---|---|---|---|---|
| Loss | 0.503 [0.48, 0.52] | 0.509 [0.49, 0.53] | 0.493 [0.47, 0.51] | 0.483 [0.46, 0.51] | 0.511 [0.49, 0.53] | 0.468 [0.42, 0.51] * |
| Zlib | 0.488 [0.47, 0.51] | 0.484 [0.46, 0.51] | 0.487 [0.47, 0.51] | 0.494 [0.47, 0.51] | 0.503 [0.48, 0.52] | 0.476 [0.43, 0.52] * |
| Min-K% | 0.501 [0.48, 0.52] | 0.493 [0.47, 0.51] | 0.488 [0.47, 0.51] | 0.493 [0.47, 0.52] | 0.499 [0.48, 0.52] | 0.456 [0.41, 0.50] * |
| Min-K%++ | 0.494 [0.47, 0.52] | 0.483 [0.46, 0.50] | 0.479 [0.46, 0.50] | 0.498 [0.48, 0.52] | 0.503 [0.48, 0.52] | 0.465 [0.42, 0.51] * |
| Ref (70m) | 0.500 [0.48, 0.52] | 0.504 [0.48, 0.53] | 0.486 [0.47, 0.51] | 0.490 [0.47, 0.51] | 0.518 [0.50, 0.54] | 0.448 [0.41, 0.49] * |

### A. Test bed: AUROC of planted items vs the 619 unseen test items, by number of exposures

| Detector | k=1 | k=2 | k=4 | k=8 | k=16 | template originals, k=16 |
|---|---|---|---|---|---|---|
| Loss | 0.684 ± 0.008 | 0.780 ± 0.038 | 0.900 ± 0.013 | 0.986 ± 0.005 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Zlib | 0.600 ± 0.028 | 0.663 ± 0.016 | 0.778 ± 0.016 | 0.919 ± 0.011 | 0.989 ± 0.004 | 0.993 ± 0.003 |
| Min-K% | 0.691 ± 0.017 | 0.789 ± 0.034 | 0.917 ± 0.012 | 0.990 ± 0.003 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Min-K%++ | 0.687 ± 0.026 | 0.797 ± 0.041 | 0.936 ± 0.012 | 0.993 ± 0.004 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Ref (70m) | 0.678 ± 0.028 | 0.778 ± 0.022 | 0.900 ± 0.010 | 0.972 ± 0.005 | 0.995 ± 0.001 | 0.996 ± 0.001 |
| Oracle | 0.666 ± 0.032 | 0.764 ± 0.023 | 0.887 ± 0.012 | 0.965 ± 0.004 | 0.992 ± 0.001 | 0.991 ± 0.002 |

### B. Where the threshold comes from (fine-tuned model, 5% target false-positive rate)

| Threshold from | Detector | Unseen test items flagged | detected, k=1 | detected, k=2 | detected, k=4 | detected, k=8 | detected, k=16 |
|---|---|---|---|---|---|---|---|
| unseen GSM8K train items | Loss | 5.9% ± 1.6 | 17.5% ± 1.6 | 26.5% ± 4.5 | 59.7% ± 4.0 | 94.8% ± 0.7 | 100.0% ± 0.0 |
| unseen GSM8K train items | Zlib | 4.8% ± 0.9 | 11.0% ± 2.8 | 18.3% ± 2.8 | 34.3% ± 3.3 | 65.8% ± 3.7 | 94.2% ± 2.4 |
| unseen GSM8K train items | Min-K% | 5.3% ± 1.1 | 14.8% ± 1.5 | 25.3% ± 4.9 | 61.0% ± 3.7 | 95.7% ± 1.2 | 100.0% ± 0.0 |
| unseen GSM8K train items | Min-K%++ | 5.9% ± 0.3 | 17.8% ± 4.3 | 31.5% ± 8.6 | 73.8% ± 6.1 | 98.7% ± 1.0 | 100.0% ± 0.0 |
| unseen GSM8K train items | Ref (70m) | 5.1% ± 0.6 | 13.2% ± 4.3 | 19.7% ± 2.7 | 42.8% ± 3.4 | 85.2% ± 3.9 | 99.2% ± 0.6 |
| unseen GSM8K train items | Oracle | 4.6% ± 0.2 | 11.8% ± 4.1 | 16.2% ± 4.0 | 33.8% ± 2.9 | 76.2% ± 3.4 | 98.2% ± 1.1 |
| GSM-Symbolic rewrites | Loss | 0.0% ± 0.1 | 0.3% ± 0.5 | 1.3% ± 1.0 | 5.8% ± 2.6 | 39.8% ± 8.1 | 98.2% ± 1.2 |
| GSM-Symbolic rewrites | Zlib | 0.2% ± 0.1 | 0.5% ± 0.7 | 2.2% ± 1.5 | 4.7% ± 1.3 | 25.5% ± 4.7 | 71.2% ± 7.0 |
| GSM-Symbolic rewrites | Min-K% | 0.3% ± 0.2 | 1.0% ± 1.4 | 2.8% ± 1.3 | 10.5% ± 4.0 | 61.3% ± 5.9 | 99.8% ± 0.4 |
| GSM-Symbolic rewrites | Min-K%++ | 0.3% ± 0.2 | 0.5% ± 0.7 | 0.7% ± 0.7 | 6.0% ± 2.7 | 58.3% ± 5.0 | 99.7% ± 0.5 |
| GSM-Symbolic rewrites | Ref (70m) | 0.5% ± 0.2 | 2.2% ± 0.5 | 1.8% ± 0.7 | 7.0% ± 2.5 | 28.0% ± 6.9 | 74.3% ± 1.8 |
| GSM-Symbolic rewrites | Oracle | 1.1% ± 0.2 | 2.8% ± 0.5 | 2.2% ± 0.7 | 7.8% ± 2.3 | 30.2% ± 6.0 | 70.0% ± 6.6 |
| normalised rewrites | Loss | 0.0% ± 0.1 | 0.3% ± 0.5 | 0.8% ± 0.6 | 4.7% ± 2.7 | 36.5% ± 8.6 | 97.0% ± 1.9 |
| normalised rewrites | Zlib | 0.2% ± 0.1 | 0.5% ± 0.7 | 2.0% ± 1.7 | 4.5% ± 1.4 | 23.5% ± 3.9 | 70.2% ± 7.1 |
| normalised rewrites | Min-K% | 0.2% ± 0.2 | 0.8% ± 1.0 | 1.8% ± 1.1 | 7.5% ± 2.2 | 52.2% ± 8.4 | 99.3% ± 0.7 |
| normalised rewrites | Min-K%++ | 0.2% ± 0.1 | 0.3% ± 0.5 | 0.5% ± 0.5 | 4.8% ± 2.6 | 56.2% ± 5.9 | 99.7% ± 0.5 |
| normalised rewrites | Ref (70m) | 0.5% ± 0.2 | 2.2% ± 0.5 | 1.8% ± 0.7 | 6.5% ± 1.9 | 27.0% ± 6.3 | 72.8% ± 1.3 |
| normalised rewrites | Oracle | 1.0% ± 0.3 | 2.5% ± 0.0 | 2.2% ± 0.7 | 7.2% ± 2.3 | 28.0% ± 5.2 | 67.5% ± 5.8 |
| rewrites of unplanted templates only (diagnostic) | Loss | 10.0% ± 4.6 | 27.3% ± 8.1 | 39.3% ± 4.8 | 70.8% ± 7.8 | 97.3% ± 1.9 | 100.0% ± 0.0 |
| rewrites of unplanted templates only (diagnostic) | Zlib | 9.7% ± 2.6 | 19.0% ± 6.0 | 25.0% ± 4.9 | 44.7% ± 8.3 | 74.3% ± 6.0 | 95.8% ± 2.3 |
| rewrites of unplanted templates only (diagnostic) | Min-K% | 14.5% ± 6.4 | 35.7% ± 11.4 | 52.2% ± 9.7 | 82.0% ± 8.5 | 98.5% ± 1.5 | 100.0% ± 0.0 |
| rewrites of unplanted templates only (diagnostic) | Min-K%++ | 7.8% ± 3.0 | 21.3% ± 3.5 | 36.8% ± 6.8 | 77.8% ± 4.6 | 98.8% ± 0.7 | 100.0% ± 0.0 |
| rewrites of unplanted templates only (diagnostic) | Ref (70m) | 10.5% ± 2.5 | 25.0% ± 4.6 | 38.8% ± 4.0 | 64.7% ± 3.6 | 94.7% ± 2.1 | 99.7% ± 0.5 |
| rewrites of unplanted templates only (diagnostic) | Oracle | 12.8% ± 3.4 | 27.3% ± 6.8 | 37.7% ± 7.3 | 64.8% ± 6.9 | 94.8% ± 3.0 | 99.8% ± 0.4 |
| normalised rewrites of unplanted templates only (diagnostic) | Loss | 8.3% ± 3.9 | 23.0% ± 7.1 | 33.3% ± 5.2 | 66.0% ± 8.0 | 96.8% ± 2.3 | 100.0% ± 0.0 |
| normalised rewrites of unplanted templates only (diagnostic) | Zlib | 9.8% ± 2.7 | 19.0% ± 6.0 | 25.0% ± 5.2 | 45.0% ± 8.5 | 74.3% ± 6.0 | 95.8% ± 2.3 |
| normalised rewrites of unplanted templates only (diagnostic) | Min-K% | 12.2% ± 5.2 | 31.5% ± 9.3 | 45.7% ± 5.7 | 78.2% ± 8.9 | 98.3% ± 1.8 | 100.0% ± 0.0 |
| normalised rewrites of unplanted templates only (diagnostic) | Min-K%++ | 6.8% ± 2.8 | 19.5% ± 2.9 | 33.3% ± 5.3 | 74.7% ± 5.6 | 98.8% ± 0.7 | 100.0% ± 0.0 |
| normalised rewrites of unplanted templates only (diagnostic) | Ref (70m) | 10.2% ± 2.2 | 23.7% ± 5.3 | 37.2% ± 4.6 | 64.0% ± 3.1 | 94.2% ± 2.4 | 99.7% ± 0.5 |
| normalised rewrites of unplanted templates only (diagnostic) | Oracle | 11.8% ± 3.2 | 25.2% ± 6.8 | 34.3% ± 5.4 | 61.8% ± 6.6 | 93.8% ± 2.7 | 99.7% ± 0.5 |

Share of the rewrites above the threshold that belong to the 50 planted templates:

| Detector | raw rewrites | normalised rewrites |
|---|---|---|
| Loss | 99.5% ± 0.7 | 99.7% ± 0.5 |
| Zlib | 100.0% ± 0.0 | 100.0% ± 0.0 |
| Min-K% | 98.6% ± 1.9 | 99.2% ± 1.1 |
| Min-K%++ | 99.8% ± 0.2 | 99.9% ± 0.2 |
| Ref (70m) | 99.8% ± 0.2 | 99.8% ± 0.2 |

### C. Rewrite audit on the clean base model (all 100 originals unseen)

| Detector | Rewrites | Mean rank of original among its rewrites [95% CI] | Flagged vs own rewrites [95% CI] | Flagged by pooled rewrite threshold [95% CI] | All 1,319 test items flagged | Wilcoxon p |
|---|---|---|---|---|---|---|
| Loss | raw | 0.734 [0.677, 0.790] | 29.0% [21.0, 38.0] | 6.0% [3.0, 13.0] | 6.1% | 9.4e-09 |
| Loss | normalised | 0.758 [0.698, 0.814] | 34.0% [25.0, 44.0] | 7.0% [4.0, 15.0] | 6.4% | 6.8e-10 |
| Loss | normalised + adjusted | 0.809 [0.753, 0.860] | 47.0% [38.0, 57.0] | 12.0% [7.0, 16.0] | - | 2.1e-13 |
| Loss | question only, raw | 0.774 [0.716, 0.829] | 31.0% [22.0, 41.0] | 7.0% [5.0, 11.0] | - | 1.3e-10 |
| Loss | question only, normalised | 0.761 [0.701, 0.818] | 29.0% [20.0, 38.0] | 7.0% [5.0, 11.0] | - | 1e-09 |
| Zlib | raw | 0.599 [0.538, 0.661] | 19.0% [12.0, 27.0] | 5.0% [3.0, 8.0] | 6.2% | 0.01 |
| Zlib | normalised | 0.659 [0.599, 0.717] | 21.0% [14.0, 29.0] | 6.0% [4.0, 9.0] | 7.6% | 2.7e-05 |
| Zlib | normalised + adjusted | 0.749 [0.694, 0.799] | 34.0% [25.0, 43.0] | 6.0% [4.0, 11.0] | - | 3e-10 |
| Zlib | question only, raw | 0.625 [0.559, 0.691] | 24.0% [16.0, 32.0] | 6.0% [4.0, 10.0] | - | 0.0082 |
| Zlib | question only, normalised | 0.644 [0.578, 0.711] | 26.0% [18.0, 35.0] | 6.0% [4.0, 10.0] | - | 0.0011 |
| Min-K% | raw | 0.759 [0.705, 0.809] | 32.0% [23.0, 42.0] | 11.0% [6.0, 16.0] | 9.6% | 4.5e-12 |
| Min-K% | normalised | 0.770 [0.717, 0.821] | 34.0% [25.0, 44.0] | 11.0% [6.0, 16.0] | 9.3% | 1.6e-12 |
| Min-K% | normalised + adjusted | 0.794 [0.741, 0.842] | 36.0% [27.0, 45.0] | 11.0% [7.0, 17.0] | - | 1.6e-13 |
| Min-K% | question only, raw | 0.730 [0.675, 0.781] | 26.0% [18.0, 35.0] | 9.0% [6.0, 13.0] | - | 7.3e-09 |
| Min-K% | question only, normalised | 0.710 [0.653, 0.763] | 24.0% [16.0, 33.0] | 9.0% [5.0, 13.0] | - | 4.8e-08 |
| Min-K%++ | raw | 0.602 [0.541, 0.664] | 14.0% [8.0, 21.0] | 10.0% [4.0, 14.0] | 7.7% | 0.0015 |
| Min-K%++ | normalised | 0.631 [0.570, 0.694] | 15.0% [8.0, 22.0] | 10.0% [5.0, 15.0] | 8.4% | 0.00019 |
| Min-K%++ | normalised + adjusted | 0.682 [0.623, 0.737] | 23.0% [15.0, 31.0] | 11.0% [7.0, 17.0] | - | 9.9e-07 |
| Min-K%++ | question only, raw | 0.629 [0.567, 0.688] | 15.0% [8.0, 22.0] | 6.0% [2.0, 9.0] | - | 0.00092 |
| Min-K%++ | question only, normalised | 0.616 [0.552, 0.675] | 15.0% [8.0, 22.0] | 6.0% [2.0, 9.0] | - | 0.0024 |
| Ref (70m) | raw | 0.448 [0.385, 0.511] | 7.0% [2.0, 12.0] | 1.0% [0.0, 4.0] | 2.4% | 0.072 |
| Ref (70m) | normalised | 0.462 [0.399, 0.527] | 7.0% [2.0, 12.0] | 1.0% [0.0, 5.0] | 2.6% | 0.18 |
| Ref (70m) | normalised + adjusted | 0.479 [0.416, 0.544] | 10.0% [4.0, 16.0] | 2.0% [0.0, 5.0] | - | 0.39 |
| Ref (70m) | question only, raw | 0.483 [0.426, 0.543] | 3.0% [0.0, 7.0] | 3.0% [1.0, 7.0] | - | 0.58 |
| Ref (70m) | question only, normalised | 0.505 [0.446, 0.566] | 5.0% [1.0, 10.0] | 3.0% [1.0, 7.0] | - | 0.97 |

Under exchangeability the mean rank is 0.5, 3.9% of originals are flagged against their own rewrites, and at most 5% by the pooled threshold.

### D. Rewrite audit on the contaminated model (50 originals planted 16 times, 50 unseen)

| Detector | Rewrites | Pooled threshold: planted caught | Pooled threshold: unseen flagged | Own rewrites: planted caught | Own rewrites: unseen flagged | Mean rank, planted | Mean rank, unseen |
|---|---|---|---|---|---|---|---|
| Loss | raw | 97.2% ± 3.0 | 0.0% ± 0.0 | 100.0% ± 0.0 | 40.4% ± 7.5 | 1.000 ± 0.000 | 0.805 ± 0.029 |
| Loss | normalised | 95.6% ± 4.1 | 0.0% ± 0.0 | 100.0% ± 0.0 | 34.0% ± 7.1 | 1.000 ± 0.000 | 0.776 ± 0.029 |
| Zlib | raw | 66.4% ± 5.2 | 0.0% ± 0.0 | 100.0% ± 0.0 | 29.2% ± 6.1 | 1.000 ± 0.000 | 0.761 ± 0.037 |
| Zlib | normalised | 64.8% ± 5.9 | 0.0% ± 0.0 | 100.0% ± 0.0 | 26.4% ± 5.9 | 1.000 ± 0.000 | 0.743 ± 0.037 |
| Min-K% | raw | 99.2% ± 1.1 | 0.0% ± 0.0 | 100.0% ± 0.0 | 41.2% ± 6.4 | 1.000 ± 0.000 | 0.822 ± 0.016 |
| Min-K% | normalised | 98.0% ± 2.0 | 0.0% ± 0.0 | 100.0% ± 0.0 | 37.2% ± 7.3 | 1.000 ± 0.000 | 0.788 ± 0.018 |
| Min-K%++ | raw | 100.0% ± 0.0 | 0.0% ± 0.0 | 100.0% ± 0.0 | 12.0% ± 5.3 | 1.000 ± 0.000 | 0.531 ± 0.045 |
| Min-K%++ | normalised | 99.6% ± 0.9 | 0.0% ± 0.0 | 100.0% ± 0.0 | 12.0% ± 5.1 | 1.000 ± 0.000 | 0.519 ± 0.053 |
| Ref (70m) | raw | 73.2% ± 5.4 | 0.0% ± 0.0 | 99.2% ± 1.1 | 24.0% ± 5.3 | 0.999 ± 0.001 | 0.647 ± 0.034 |
| Ref (70m) | normalised | 70.4% ± 7.1 | 0.0% ± 0.0 | 99.2% ± 1.1 | 15.2% ± 6.4 | 0.999 ± 0.001 | 0.603 ± 0.040 |

Diagnostic: pooled threshold from the rewrites of the 50 unplanted templates only (an auditor cannot do this):

| Detector | Rewrites | planted caught | unseen flagged |
|---|---|---|---|
| Loss | raw | 100.0% ± 0.0 | 14.4% ± 4.1 |
| Loss | normalised | 100.0% ± 0.0 | 12.4% ± 3.8 |
| Zlib | raw | 98.0% ± 2.0 | 8.0% ± 2.8 |
| Zlib | normalised | 98.0% ± 2.0 | 8.0% ± 2.8 |
| Min-K% | raw | 100.0% ± 0.0 | 19.2% ± 3.6 |
| Min-K% | normalised | 100.0% ± 0.0 | 14.8% ± 2.7 |
| Min-K%++ | raw | 100.0% ± 0.0 | 9.2% ± 4.1 |
| Min-K%++ | normalised | 100.0% ± 0.0 | 8.0% ± 4.0 |
| Ref (70m) | raw | 100.0% ± 0.0 | 8.0% ± 1.4 |
| Ref (70m) | normalised | 100.0% ± 0.0 | 7.2% ± 1.1 |

NLL drop from fine-tuning, unplanted templates: originals 0.924 ± 0.034, their rewrites 0.808 ± 0.017 nats per token.

### E. Spill-over: score gain of rewrites, planted templates minus unplanted templates

| Detector | Difference in differences | AUROC of per-template gain, planted vs unplanted | permutation p |
|---|---|---|---|
| Loss | 0.904 ± 0.043 | 0.944 ± 0.010 | 0.0001 |
| Zlib | 0.004 ± 0.000 | 0.835 ± 0.002 | 0.0001 |
| Min-K% | 2.056 ± 0.093 | 0.904 ± 0.011 | 0.0001 |
| Min-K%++ | 4.178 ± 3.143 | 0.902 ± 0.010 | 0.0001 |
| Ref (70m) | 0.904 ± 0.043 | 0.944 ± 0.010 | 0.0001 |

Loss and Ref give the same gain by construction (the reference model's term cancels), and both equal the Oracle score of the rewrites.

### Pre-registered hypotheses (Holm-corrected over the five detectors)

| Hypothesis | Detector | Statistic | p | Holm p |
|---|---|---|---|---|
| H1 clean originals flagged by raw-rewrite threshold, above exchangeable rate | Loss | 0.06 | 0.35 | 1 |
| H1 clean originals flagged by raw-rewrite threshold, above exchangeable rate | Zlib | 0.05 | 0.69 | 1 |
| H1 clean originals flagged by raw-rewrite threshold, above exchangeable rate | Min-K% | 0.11 | 0.0001 | 0.0005 |
| H1 clean originals flagged by raw-rewrite threshold, above exchangeable rate | Min-K%++ | 0.1 | 0.0021 | 0.0084 |
| H1 clean originals flagged by raw-rewrite threshold, above exchangeable rate | Ref (70m) | 0.01 | 1 | 1 |
| H2 mean rank ≠ 0.5 (normalised) | Loss | 0.758 | 0.0001 | 0.0005 |
| H2 mean rank ≠ 0.5 (normalised) | Zlib | 0.659 | 0.0001 | 0.0005 |
| H2 mean rank ≠ 0.5 (normalised) | Min-K% | 0.77 | 0.0001 | 0.0005 |
| H2 mean rank ≠ 0.5 (normalised) | Min-K%++ | 0.631 | 0.0001 | 0.0005 |
| H2 mean rank ≠ 0.5 (normalised) | Ref (70m) | 0.462 | 0.2 | 0.2 |
| H3 spill-over > 0 | Loss | 0.904 | 0.0001 | 0.0005 |
| H3 spill-over > 0 | Zlib | 0.00361 | 0.0001 | 0.0005 |
| H3 spill-over > 0 | Min-K% | 2.06 | 0.0001 | 0.0005 |
| H3 spill-over > 0 | Min-K%++ | 4.18 | 0.0001 | 0.0005 |
| H3 spill-over > 0 | Ref (70m) | 0.904 | 0.0001 | 0.0005 |

### Blind baseline: surface features only, originals vs rewrites (base model)

| Rewrites | Classifier: pooled AUROC [95% CI] | Classifier: mean rank among own rewrites [95% CI] | Loss: pooled AUROC / mean rank | Zlib: pooled AUROC / mean rank | Min-K%: pooled AUROC / mean rank | Min-K%++: pooled AUROC / mean rank | Ref (70m): pooled AUROC / mean rank |
|---|---|---|---|---|---|---|---|
| raw | 0.702 [0.657, 0.750] | 0.659 [0.586, 0.729] | 0.558 / 0.734 | 0.507 / 0.599 | 0.585 / 0.759 | 0.534 / 0.602 | 0.480 / 0.448 |
| normalised | 0.650 [0.612, 0.693] | 0.791 [0.735, 0.842] | 0.567 / 0.758 | 0.517 / 0.659 | 0.589 / 0.770 | 0.545 / 0.631 | 0.491 / 0.462 |

### Exploratory: single exposure early vs late in training (train-reference threshold)

| Detector | detected, first half | detected, second half |
|---|---|---|
| Loss | 17.5% ± 2.5 | 17.4% ± 2.7 |
| Zlib | 8.2% ± 2.6 | 13.9% ± 4.5 |
| Min-K% | 13.7% ± 4.9 | 15.6% ± 4.3 |
| Min-K%++ | 16.6% ± 6.3 | 19.1% ± 3.1 |
| Ref (70m) | 13.7% ± 4.6 | 12.7% ± 4.4 |
| Oracle | 12.8% ± 6.0 | 11.1% ± 2.9 |
