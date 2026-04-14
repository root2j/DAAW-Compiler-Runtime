# DAAW eval — 2026-04-14 12:10

Provider: `groq` · Model: `llama-3.1-8b-instant`

Ran **21** prompts: 15 success · 6 failed · 0 skipped
Total wall time: 982s · API calls: 119 · Tokens: 79311

## Per-prompt summary

| ID | Complexity | Status | Tasks | Pass/Fail | Compile s | Exec s | Total s |
|----|-----------|--------|-------|-----------|-----------|--------|---------|
| P01 | Simple | success | 2 | 1/2 | 40.9 | 86.8 | 134.7 |
| P02 | Simple | success | 2 | 1/2 | 3.1 | 32.0 | 41.4 |
| P03 | Medium | failure | 0 | 0/0 | 13.9 | 0.0 | 0.0 |
| P04 | Medium | success | 2 | 1/2 | 15.2 | 30.3 | 53.1 |
| P05 | Simple | failure | 0 | 0/0 | 17.8 | 0.0 | 0.0 |
| P06 | Simple | success | 3 | 1/3 | 21.9 | 40.1 | 71.0 |
| P07 | Medium | success | 2 | 1/2 | 12.3 | 41.5 | 61.4 |
| P08 | Medium | success | 3 | 2/3 | 18.8 | 36.2 | 64.1 |
| P09 | Simple | success | 2 | 0/2 | 2.5 | 54.7 | 64.4 |
| P10 | Medium | failure | 0 | 0/0 | 18.7 | 0.0 | 0.0 |
| P11 | Medium | failure | 0 | 0/0 | 20.8 | 0.0 | 0.0 |
| P12 | Medium | success | 2 | 2/2 | 22.2 | 51.8 | 78.5 |
| P13 | Simple | success | 3 | 1/3 | 10.9 | 154.5 | 175.2 |
| P14 | Medium | success | 1 | 0/1 | 17.2 | 7.7 | 24.9 |
| P15 | Medium | success | 1 | 0/1 | 16.6 | 8.8 | 25.5 |
| P16 | Medium | success | 2 | 1/2 | 6.2 | 45.9 | 57.5 |
| P17 | Complex | success | 1 | 0/1 | 10.4 | 49.4 | 59.8 |
| P18 | Complex | failure | 0 | 0/0 | 2.5 | 0.0 | 0.0 |
| P19 | Simple | success | 2 | 1/2 | 9.1 | 29.3 | 38.7 |
| P20 | Medium | success | 2 | 1/2 | 2.4 | 24.2 | 32.3 |
| P21 | Complex | failure | 0 | 0/0 | 16.7 | 0.0 | 0.0 |

## Failures

- **P03** — `compile: RuntimeError: Compiler failed after 3 attempts: No tasks generated. The workflow must have at least one task.`
- **P05** — `compile: RuntimeError: Compiler failed after 3 attempts: No tasks generated. The workflow must have at least one task.`
- **P10** — `compile: RuntimeError: Compiler failed after 3 attempts: No tasks generated. The workflow must have at least one task.`
- **P11** — `compile: RuntimeError: Compiler failed after 3 attempts: No tasks generated. The workflow must have at least one task.`
- **P18** — `compile: RuntimeError: Compiler failed after 3 attempts: No tasks generated. The workflow must have at least one task.`
- **P21** — `compile: RuntimeError: Compiler failed after 3 attempts: No tasks generated. The workflow must have at least one task.`