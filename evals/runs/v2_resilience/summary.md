# DAAW eval — 2026-04-14 15:21

Provider: `groq` · Model: `llama-3.1-8b-instant`

Ran **21** prompts: 3 success · 18 failed · 0 skipped
Total wall time: 384s · API calls: 43 · Tokens: 29560

## Per-prompt summary

| ID | Complexity | Status | Tasks | Pass/Fail | Compile s | Exec s | Total s |
|----|-----------|--------|-------|-----------|-----------|--------|---------|
| P01 | Simple | success | 4 | 4/4 | 1.5 | 171.0 | 184.2 |
| P02 | Simple | success | 4 | 4/4 | 1.4 | 130.0 | 144.8 |
| P03 | Medium | success | 3 | 3/3 | 1.1 | 45.5 | 54.9 |
| P04 | Medium | failure | 0 | 0/0 | 0.5 | 0.0 | 0.0 |
| P05 | Simple | failure | 0 | 0/0 | 0.3 | 0.0 | 0.0 |
| P06 | Simple | failure | 0 | 0/0 | 0.3 | 0.0 | 0.0 |
| P07 | Medium | failure | 0 | 0/0 | 0.2 | 0.0 | 0.0 |
| P08 | Medium | failure | 0 | 0/0 | 0.2 | 0.0 | 0.0 |
| P09 | Simple | failure | 0 | 0/0 | 0.3 | 0.0 | 0.0 |
| P10 | Medium | failure | 0 | 0/0 | 0.7 | 0.0 | 0.0 |
| P11 | Medium | failure | 0 | 0/0 | 0.3 | 0.0 | 0.0 |
| P12 | Medium | failure | 0 | 0/0 | 0.3 | 0.0 | 0.0 |
| P13 | Simple | failure | 0 | 0/0 | 0.3 | 0.0 | 0.0 |
| P14 | Medium | failure | 0 | 0/0 | 0.3 | 0.0 | 0.0 |
| P15 | Medium | failure | 0 | 0/0 | 0.4 | 0.0 | 0.0 |
| P16 | Medium | failure | 0 | 0/0 | 0.4 | 0.0 | 0.0 |
| P17 | Complex | failure | 0 | 0/0 | 0.6 | 0.0 | 0.0 |
| P18 | Complex | failure | 0 | 0/0 | 1.3 | 0.0 | 0.0 |
| P19 | Simple | failure | 0 | 0/0 | 0.2 | 0.0 | 0.0 |
| P20 | Medium | failure | 0 | 0/0 | 0.2 | 0.0 | 0.0 |
| P21 | Complex | failure | 0 | 0/0 | 0.2 | 0.0 | 0.0 |

## Failures

- **P04** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99932, Requested 841. Please try again in 11m7.872s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P05** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99926, Requested 742. Please try again in 9m37.152s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P06** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99919, Requested 837. Please try again in 10m53.184s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P07** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99913, Requested 845. Please try again in 10m54.912s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P08** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99907, Requested 748. Please try again in 9m25.92s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P09** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99901, Requested 4486. Please try again in 1h3m10.368s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P10** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99894, Requested 4495. Please try again in 1h3m12.096s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P11** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99888, Requested 864. Please try again in 10m49.728s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P12** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99882, Requested 847. Please try again in 10m29.856s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P13** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99875, Requested 4482. Please try again in 1h2m44.448s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P14** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99869, Requested 776. Please try again in 9m17.28s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P15** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99863, Requested 856. Please try again in 10m21.216s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P16** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99856, Requested 4497. Please try again in 1h2m40.991999999s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P17** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99850, Requested 854. Please try again in 10m8.256s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P18** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99842, Requested 862. Please try again in 10m8.256s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P19** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99836, Requested 753. Please try again in 8m28.896s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P20** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99830, Requested 761. Please try again in 8m30.624s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
- **P21** — `compile: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kja7k862fcattbev6bsrhprf` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99824, Requested 4513. Please try again in 1h2m27.168s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`