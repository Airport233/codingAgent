+++
name = "code-review"
description = "Review code changes for concrete, actionable defects"
+++

# Code Review Workflow

1. Determine the exact review scope and inspect repository guidance before judging the change.
2. Read the diff and enough surrounding code to understand contracts, callers, and platform behavior.
3. Run targeted read-only checks or tests when they can confirm or reject a suspected defect.
4. Report only actionable findings: incorrect behavior, regressions, security issues, data loss, or missing tests for risky logic.
5. Rank findings by severity and include a precise file and location plus a reproducible failure scenario.
6. Separate confirmed defects from questions or optional improvements.
7. If there are no findings, say so and state the residual testing gaps.

Review only. Do not modify files unless the user separately asks for fixes.
