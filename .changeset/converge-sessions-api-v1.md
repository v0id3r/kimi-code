---
"@moonshot-ai/kimi-code": patch
---

Move the `/api/v2/sessions` list and batch `:archive`/`:restore` endpoints onto `/api/v1/sessions`, replacing `include_archive`, `archived_only`, and `exclude_empty` with `archived` and `has_prompt`.
