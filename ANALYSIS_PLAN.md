# Analysis plan

Written and committed before the main runs. The learning rate was then chosen by the pilot rule
in section 5, using GSM8K *train* items only. Sections 7 and 8 record the fixes made after this
commit but before the main runs, and the analyses added after the first seed had finished.

## 1. Question

A common way to audit a model for benchmark contamination is to compare its scores on the
original benchmark items with scores on freshly rewritten versions of the same items, which the
model cannot have seen. When ground truth is known, how often does this audit

* flag items that were never seen (false alarms), and
* miss items that were seen (misses),

compared with calibrating on genuinely unseen items from the original distribution?

The planted dose-response (more exposures, easier detection) is a known result. Here it is a
correctness check of the test bed, not a finding.

## 2. Models and data

* Target: `EleutherAI/pythia-160m-deduped`. Reference: `EleutherAI/pythia-70m-deduped`, same
  tokenizer and pretraining data. Revisions are pinned in `scripts/run_experiments.py`.
* GSM8K (released October 2021) and GSM-Symbolic (2024) are newer than the Pile (December 2020),
  so the base model has not seen either. A negative control (section 4) checks this.
* GSM-Symbolic `main`: 100 GSM8K-test templates x 50 rewrites. Its `original_id` indexes GSM8K
  test; the original questions match exactly (checked in code).
* Item text: `Question: ...\nAnswer: ...` with GSM8K calculator annotations removed. Scoring
  prepends `<|endoftext|>`, matching the document boundary every planted copy sits behind.

## 3. Design

Per seed (5 seeds, 0-4), with fresh random groups each time:

* 1,219 non-template GSM8K test items: 120 each planted 1, 2, 4, 8, 16 times; the other 619
  are unseen.
* 100 template originals: 50 planted 16 times, 50 unseen.
* Continued pretraining on 4M tokens of WikiText-103 plus the planted copies. Copies stay whole
  inside one 512-token block; copies of one item go to different blocks. One epoch, AdamW,
  cosine schedule, batch 16 x 512, bfloat16 autocast with float32 weights.

Scored sets: all 1,319 GSM8K test items; a fixed sample of 1,000 GSM8K train items (unseen,
in-distribution reference); the 5,000 rewrites, raw and surface-normalised; and on the base
model also question-only versions.

Detectors (higher = more likely seen): Loss, Zlib (mean NLL / zlib bytes), Min-K% (k = 20%),
Min-K%++ (k = 20%), Ref (NLL of pythia-70m minus NLL of target). Oracle = NLL before minus after
fine-tuning; it is an upper bound an auditor never has, and is excluded from every hypothesis
family.

## 4. Analyses

All metrics are computed within a seed, then reported as the mean and standard deviation over
seeds, with per-seed values in the JSON output. Intervals that involve rewrites resample whole
templates.

Thresholds use the conformal rule: with n reference scores, flag when the score is strictly
above the ceil((n + 1)(1 - 0.05))-th smallest reference score.

**Negative controls (base model).** AUROC of GSM8K test vs the 1,000 train items, and of every
planted group vs the unseen items before fine-tuning. Both should be about 0.5.

**A. Test bed check.** AUROC of each planted group vs the 619 unseen non-template items.

**B. Where the threshold comes from (main table).** Thresholds from three reference sets:
(i) 1,000 unseen GSM8K train items, (ii) the 5,000 raw rewrites, (iii) the 5,000 normalised
rewrites. For each: the false-positive rate on the 619 unseen test items, and the detection rate
of each planted group.

**C. Audit on a clean model (base model).** For each template, the rank of the original among
its 50 rewrites, as a fraction (rank / 51). Under exchangeability this is uniform with mean 0.5.
Reported: the mean rank fraction, the share of templates where the original falls above the
95th percentile of its own rewrites, a Wilcoxon signed-rank test of original minus median
rewrite, and the share of the 100 originals (and of all 1,319 test items) flagged by the
threshold from (ii) or (iii). Three layers: raw rewrites; normalised rewrites (currency signs,
`.0` decimals, `÷`, blank line before `####`, NFC); and normalised with a covariate adjustment
for token count and share of digit characters, fit within templates. Question-only scoring is a
sensitivity check.

**D. Audit on a contaminated model.** Same audit after fine-tuning: recall on the 50 planted
template originals, false alarms on the 50 unseen ones.

**E. Spill-over.** Difference in differences: change in a rewrite's score from base to
fine-tuned, averaged per template, planted templates minus unplanted templates.

**Blind baseline.** Logistic regression on surface features only (characters, tokens, digit
count, longest number, non-ASCII share, non-dollar currency signs, `.0` count), with 5-fold
cross-validation grouped by template, separating originals from rewrites.

## 5. Pilot rule for the learning rate

Pilot: plant GSM8K train items (outside the 1,000-item reference sample) with the same
exposures, score them against 600 unseen train items. Start at 1e-4. If Loss AUROC at k = 4 is
at least 0.98, use 3e-5; if Loss AUROC at k = 16 is below 0.75, use 3e-4. Otherwise keep 1e-4.

## 6. Hypotheses and multiplicity

Primary families, each Holm-corrected over the five detectors:

* H1 (clean-model false alarms): with the threshold from raw rewrites, the share of the 100
  unseen originals flagged on the base model exceeds 5%.
* H2 (rewrite effect beyond surface): within-template mean rank fraction on normalised rewrites
  differs from 0.5.
* H3 (spill-over): the difference in differences in E is above zero.

Everything else is descriptive or exploratory and is reported with unadjusted intervals only.
Claims of "no effect" use equivalence bounds (AUROC within 0.5 +/- 0.05) on 90% intervals.

## 7. Amendments before the main runs

Made after the commit above, before any main run and before any score on GSM8K test items was
computed. A code review found them; each is a fix, not a choice informed by results.

1. **Training precision.** Training now runs in float32 throughout instead of bfloat16 autocast.
   Pythia's logits share a large offset (mean |logit| about 740 on pythia-160m), and under
   autocast the rounding of the logits alone raised the cross-entropy of a 16 x 512 WikiText
   batch from 3.81 to 4.36 nats; keeping only the head in float32 still left +0.07. The learning
   rate pilot was rerun in float32 with the same rule.
2. **Every planted copy is trained on.** The block count is now a multiple of the 16 blocks per
   optimizer step; before, up to 15 blocks at the end of the shuffled corpus, with the planted
   copies in them, were never trained on.
3. **Threshold rule.** The code now implements the conformal rule of section 4 (it used a plain
   empirical quantile before).
4. **Bookkeeping.** Results are written atomically, cached base and reference scores are tied to
   a hash of the scored texts, a seed is skipped on resume only if it was run with the same
   settings, and the default results directory includes the learning rate.

## 8. Added after the first seed's results

A second review of the analysis code, run once seed 0 had finished, led to these additions. None
changes a pre-registered number; the additions are labelled as diagnostic or post hoc in the
README.

1. **Spill-over diagnostic for the rewrite threshold.** After fine-tuning, every rewrite above
   the pooled rewrite threshold came from the 50 planted templates, so the B and D results with
   rewrite thresholds depend on how many templates are planted and how often. Added: the same
   thresholds built only from the rewrites of the 50 unplanted templates (an auditor cannot do
   this), and the share of above-threshold rewrites that come from planted templates.
2. **Oracle is not an upper bound.** Section 3 calls it one; the data show it below Loss,
   Min-K% and Min-K%++, because the NLL drop of unseen items depends on how hard they were to
   begin with. It is now described as "NLL before minus after fine-tuning", still outside every
   hypothesis family.
3. **Coverage fixes.** D is reported for normalised rewrites as well as raw ones; the template
   originals in A are compared with the 619 unseen items as section 4 says; the before-training
   negative controls now include the template group, with 90% intervals and the equivalence
   rule; the blind baseline also reports the within-template rank used in C and H2.
