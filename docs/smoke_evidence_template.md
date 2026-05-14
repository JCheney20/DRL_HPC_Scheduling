# Smoke Evidence Template (Submission 2)

Use this template to record selector-only smoke evidence in a consistent, audit-ready way.

## 1. Run Context

| field | value |
|---|---|
| date (UTC/local) | |
| operator | |
| git commit hash | |
| environment | Nix |
| machine | |
| split id | |
| dev split path | |
| holdout split path | |
| split metadata json | |

## 2. Split Generation Evidence

| check | status | evidence path / note |
|---|---|---|
| dev split file exists (`*_dev70.tsv`) | | |
| holdout split file exists (`*_holdout30.tsv`) | | |
| metadata json exists (`data/splits/logs/<split_id>.json`) | | |
| row counts add correctly (`dev_rows + holdout_rows = total_rows`) | | |
| stdout metadata printed source/rows/ratio/paths/split_id | | |

## 3. Guard Test Evidence

| test | command | expected result | actual result | status |
|---|---|---|---|---|
<<<<<<< HEAD
| negative holdout guard | `python train_agents.py --algo maskable_dqn --trace splits/<name>_holdout30.tsv --save_interval 1000 --total_saving 1 --seed <seed>` | fail-fast before env setup with holdout error | | |
| positive dev start | `python train_agents.py --algo maskable_dqn --trace splits/<name>_dev70.tsv --save_interval 1000 --total_saving 1 --seed <seed> --buffer-size <small>` | starts env setup/training normally | | |
=======
| negative holdout guard | `python src/train_agents.py --algo maskable_dqn --trace splits/<name>_holdout30.tsv --save_interval 1000 --total_saving 1 --seed <seed>` | fail-fast before env setup with holdout error | | |
| positive dev start | `python src/train_agents.py --algo maskable_dqn --trace splits/<name>_dev70.tsv --save_interval 1000 --total_saving 1 --seed <seed> --buffer-size <small>` | starts env setup/training normally | | |
>>>>>>> e7ed95b (update:ver1.1)

## 4. Smoke Matrix Evidence

Use fixed seed and smoke controls (`--save_interval 1000 --total_saving 1`).

| run_id | algorithm | masking | seed | command | expected checkpoint | observed checkpoint path | exit code | status |
|---|---|---|---|---|---|---|---|---|
| smoke_a2c_mask_on | maskable_a2c | true | | | `trained_model/smoke_a2c_mask_on/selector/1000.zip` | | | |
| smoke_dqn_mask_on | maskable_dqn | true | | | `trained_model/smoke_dqn_mask_on/selector/1000.zip` | | | |
| smoke_dqn_mask_off | maskable_dqn | false | | | `trained_model/smoke_dqn_mask_off/selector/1000.zip` | | | |

## 5. Metrics/Quality Sanity

| check | pass/fail | note |
|---|---|---|
| run completed without traceback | | |
| rewards finite (no NaN/Inf) | | |
| action masks valid in mask-on runs | | |
| no holdout used for training | | |
| outputs written to expected directories | | |

## 6. Artefact Index

| artefact type | path |
|---|---|
| split metadata json | |
| train logs directory | |
| model checkpoint(s) | |
| console transcript (optional) | |

## 7. Sign-off

| role | name | date | signature/initials |
|---|---|---|---|
| run operator | | | |
| reviewer | | | |
