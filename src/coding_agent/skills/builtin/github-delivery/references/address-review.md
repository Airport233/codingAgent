# Address review: respond, re-test

1. Respond to every review comment: either address it, push back with reasoning, or mark it resolved with an explanation.
2. Do not dismiss a comment without a written reason.
3. Do not squash or rebase while review is in progress — it destroys the reviewer's diff context. Add fixup commits instead.
4. After each change: re-run the focused tests for the changed area, then the full gate before re-requesting review.
5. If a reviewer requests a fundamental design change, discuss it in comments before rewriting.

## Checklist
- [ ] Every comment has a response.
- [ ] No squash/rebase since review started.
- [ ] Gate is green after the latest changes.
- [ ] Re-requested review from all reviewers.
