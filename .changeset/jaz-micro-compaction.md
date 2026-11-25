---
"@moonshot-ai/kimi-code": minor
---

Clear old tool result outputs from the context of long sessions to reduce token cost: the full output is saved to a file and replaced with a short pointer the model can read back. Configure thresholds under [micro_compaction] in config.toml, or set enabled = false there to turn it off.
