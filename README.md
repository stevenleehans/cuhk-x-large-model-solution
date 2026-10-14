# CUHK-X: Human Activity Understanding

Hans Lee’s public source portfolio for the **CUHK-X Large Model Track**: multiple-choice question answering over anonymised human-activity videos.

**[Kaggle competition](https://www.kaggle.com/competitions/cuhk-x-competition-large-model-track)** · **[Portfolio](https://stevenleehans.github.io/#projects)** · **[Organizer’s challenge page](https://openaiotlab.github.io/CUHK-X-Challenge/)**

Hans received Top 15 Finalist recognition and an invitation to present a poster for Top 6 selection. He did not attend that subsequent stage; this is not a claim of Top 6 participation.

## What I built

A hybrid pipeline combining structured question evidence with video-based action scores:

1. Parse question categories and enforce the appropriate answer grammar.
2. Group questions that refer to the same clip, using option recurrence and a latent action-set model.
3. Combine action posteriors with canonical video scores through log-odds fusion.
4. Extract depth-video features for pace/take estimation and category-specific decoding.
5. Evaluate with subject-disjoint folds, then write validated predictions.

The implementation includes a 3D video action recognizer, CPU preprocessing, optional Gemini video prompting, and regression tests for video-score switching, serialization, path safety and label removal.

## Source map

| Path | Purpose |
| --- | --- |
| `src/common.py` | Question loading, clip identities, answer grammar and subject-disjoint folds |
| `src/text_prior.py`, `src/clip_graph.py`, `src/action_set.py` | Structured question baselines and action-set reasoning |
| `src/actions.py`, `src/fusion.py`, `src/pipeline.py` | Canonical action vocabulary, fusion and category decoding |
| `src/depth_features.py`, `src/take_model.py` | Video-derived pace features and emotion/take models |
| `src/action_recognizer.py`, `src/preprocess_clips.py` | Video action training and clip preparation |
| `src/inference_io.py`, `src/infer.py`, `src/build_bundle.py` | Input contracts, bundle tooling and prediction entry point |
| `src/gemini_video.py` | Optional externally configured video prompting |
| `kernels/` | Kaggle GPU training recipes and reproducibility protocol |
| `tests/` | CPU fixture and regression checks |

## Run the included checks

Use Python 3.11 or later in an isolated environment:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Tests use small generated fixtures and do not need competition data, API credentials or a GPU. The included 11 tests passed on 14 October 2026. Passing them verifies specific software contracts; it does not reproduce the final leaderboard result.

## Use with competition data

Obtain the competition inputs and permissions from the organizers. Dataset files, media, labels and model weights are not redistributed here.

For the CSV text baseline, place `training_qa.csv` in `data/raw/`, following the organizer’s schema, then run:

```sh
python src/evaluate.py
```

For GPU training, install `requirements-training.txt` in an appropriate GPU environment and inspect the recipes in `kernels/`. Video-enabled inference also requires the matching feature tables, action-score arrays and compatible model bundle described by the source contracts. Those artifacts are not included; this repository is not a one-command reproduction of the ranked submission.

## Evaluation and scope

Questions share clips and recording subjects, so the code uses subject-disjoint validation rather than a random row split. Regression checks preserve the distinction between training video scores and the current prediction split. Input-contract checks remove any answer/prediction columns from inference inputs and reject unsafe paths or misaligned scores.

This public repository contains selected automated pipeline components and training recipes from the local research project. It excludes the final staged verification package, frozen test-output tables and later hand-edited answer stages. It does not claim that these selected components reproduce the final entry, independently certify rule compliance, or establish unseen-data performance. Competition recognition and software validation are separate evidence.

## Publication notes

The original research folder and its private archive remain intact. Credentials, payment receipts, gated datasets, media, model weights, private logs and submission archives are excluded. The optional Gemini helper reads credentials from the environment, sends the key in a request header, and defaults to one explicitly configured model rather than the original experimental model list. No API calls occur during the included checks.
