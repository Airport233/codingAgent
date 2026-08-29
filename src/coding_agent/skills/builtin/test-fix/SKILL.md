---
name: test-fix
description: Reproduce, diagnose, fix, and verify failing tests. Use when tests or CI fail, a regression needs a focused test, or the user asks to repair a broken test suite.
---

# Test Fix Workflow

1. Inspect `git_status` when available, then read the failure, the relevant test, and the smallest related production surface.
2. Run the narrowest command that reliably reproduces the failure. Record the actual error.
3. Identify the root cause. Do not weaken assertions, skip tests, or hide errors merely to turn CI green.
4. Add or refine a focused regression test when the existing failure does not fully express the bug.
5. Make the smallest coherent production change that fixes the cause.
6. Re-run the focused test, then the relevant module suite, static checks, and broader tests in proportion to risk.
7. Report changed files, verification evidence, and any remaining uncertainty.

Preserve unrelated user changes. Do not commit, push, delete, or rewrite Git history unless the user explicitly asks.
