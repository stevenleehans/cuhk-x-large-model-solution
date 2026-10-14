# Fixed recipe: reproducible full-clip action v1

Prespecified before launch, September 8, 2026. One private Kaggle GPU run, six existing subject folds, 16 epochs per fold, seed 20260908 + fold. No validation-driven stopping or parameter sweep. Legacy v5 scores remain the HAU control; equal-weight legacy ensemble remains HARn control.

Purpose: obtain usable checkpoints for fresh-data inference while testing whole-clip temporal coverage and the official Kinetics input transform as a combined recipe. This does not separately estimate each change's effect. Previous contiguous half-clip crops retained whole-clip labels, potentially omitting labeled actions. New training samples one frame from each of 16 chronological bins spanning all 32 stored frames; inference averages the two full-span phase offsets and horizontal flips. This preserves broad coverage but cannot guarantee every short action is sampled.

Official torchvision R2Plus1D_18 KINETICS400_V1 transform: resize to 128x171, center crop 112x112, rescale to 0-1, normalize with its stored means/stds. Source: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html . The input is grayscale depth replicated to three channels; RGB pretraining remains a domain mismatch.

BCE action-presence loss plus 0.5 HARn-only cross entropy, AdamW lr 0.0003, weight decay 0.0001, OneCycleLR, batch 16, no take head. No test answers, manual labels, or test-score selection. Nonfinite loss, malformed data, missing GPU, incomplete OOF coverage, or checkpoint roundtrip mismatch stops execution.

Outputs: six final-epoch state_dict checkpoints, per-fold held subjects and indices, exact training/index/meta/source hashes, runtime/device/dependency provenance, OOF/test action probabilities and indices, checkpoint roundtrip evidence, inference timings. Fresh-array inference loads saved weights without downloading pretrained weights.

Promotion gate: validate all outputs, then compare frozen split-source QA pipelines at HAU weight 0.75 on identical held questions. Require positive paired weighted accuracy with a subject-cluster 95% lower bound above zero versus the corrected legacy control, no unexplained category collapse, and passing checkpoint-to-score reproduction. Reused folds are development evidence, not independent generalization proof. Do not promote on action AP alone or automatically submit. If weak, retain checkpoints as a reproducible fallback without claiming score superiority.

Compute ceiling: one run, estimated 2 hours on a P100 based on previous logs; Kaggle GPU only. Inspect authoritative status; never restart on an observation timeout. Do not overwrite previous notebooks, submissions, or datasets.
