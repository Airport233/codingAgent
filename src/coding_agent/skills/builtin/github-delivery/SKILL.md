---
name: github-delivery
description: Standardized GitHub engineering delivery through pull requests. Use when contributing code, creating PRs, reviewing diffs, or merging changes.
---

# GitHub Engineering Delivery

This skill guides formal code delivery through pull requests. Read the
scenario that matches your current phase with `read_skill_resource`.

## Scenarios

1. **start-work** — Branch from main, name it. → read `references/start-work.md`
2. **commit** — Atomic, conventional commits. → read `references/commit.md`
3. **pre-pr** — Run gates, self-review diff. → read `references/pre-pr.md`
4. **create-pr** — gh pr create, title, body. → read `references/create-pr.md`
5. **address-review** — Respond to feedback, re-test. → read `references/address-review.md`
6. **merge** — Squash, cleanup, sync. → read `references/merge.md`

## Non-negotiable

- Never force-push to main/master.
- Never commit secrets, `.env`, or credentials.
- Never skip the gate (tests, lint, type checks) before requesting review.
- Preserve unrelated user changes; do not delete or rewrite Git history unless explicitly asked.
