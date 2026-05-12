# EMNLP 2026 Submission — Plans Index

This index points to one implementation plan per sub-project from the design spec at `docs/superpowers/specs/2026-05-12-emnlp-submission-design.md`.

**Deadline:** May 25 2026 AoE
**Compute:** Local A100 80GB (this machine) + 1 rented A100 80GB (SSH worker)

## Order of execution

Day 1 starts with S1 (environments). S2 / S3 begin on day 2 once envs work. S4 / S5 / S6 / S7 / S8 / S9 run in parallel from day 3 onward. S10 (paper) runs continuously from day 1.

## Plans

| # | Plan | Wall-clock | Compute |
|---|---|---|---|
| S1 | [2026-05-12-s1-environments.md](2026-05-12-s1-environments.md) | Day 1 | Local + Remote (parallel) |
| S2 | [2026-05-12-s2-ahcptq-reproduction.md](2026-05-12-s2-ahcptq-reproduction.md) | Days 2–4 | Remote |
| S3 | [2026-05-12-s3-torchao-integration.md](2026-05-12-s3-torchao-integration.md) | Days 2, 8–9 | Local + Remote |
| S4 | [2026-05-12-s4-dbaf-weak-baselines.md](2026-05-12-s4-dbaf-weak-baselines.md) | Days 3–5 | Local + Remote |
| S5 | [2026-05-12-s5-kv-pcsa-ruler.md](2026-05-12-s5-kv-pcsa-ruler.md) | Days 5–9 | Local |
| S6 | [2026-05-12-s6-real-int4-deployment.md](2026-05-12-s6-real-int4-deployment.md) | Days 7–9 | Local + Remote |
| S7 | [2026-05-12-s7-ablations.md](2026-05-12-s7-ablations.md) | Day 10 | Local |
| S8 | [2026-05-12-s8-compsrt.md](2026-05-12-s8-compsrt.md) | Days 6, 9–10 | Remote |
| S9 | [2026-05-12-s9-mmlu-gsm8k.md](2026-05-12-s9-mmlu-gsm8k.md) | Days 7–8 | Remote |
| S10 | [2026-05-12-s10-paper-revision.md](2026-05-12-s10-paper-revision.md) | Days 1–13 | (writing) |

## Cross-plan checkpoints + sync

All plans write artifacts under `/home/ubuntu/unifying-ptq/results/<sub-project>/<model>/<method>/<seed>/`. The rsync wrapper at `/home/ubuntu/unifying-ptq/scripts/sync_results.sh` (created in S1) keeps local and remote in sync.

## How to execute

Each plan stands alone. Pick a plan and use the `superpowers:executing-plans` skill to work through its tasks, or `superpowers:subagent-driven-development` to dispatch agents per task. Plans link to the spec for context.

Start with **S1**. S10 can also be started from day 1 in parallel (no compute dependency).
