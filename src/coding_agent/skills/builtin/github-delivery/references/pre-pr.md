# Pre-PR: run gates, self-review diff

1. Run the full quality gate before requesting review:
   - Tests: `uv run pytest` (or the project's test command).
   - Lint: `uv run ruff check src tests`.
   - Types: `uv run pyright src`.
   - Coverage: `uv run pytest --cov` (if the project has coverage thresholds).
2. Fix everything red. Do not open a PR with failing gates.
3. Self-review your diff: `git diff main...HEAD` (or `git log main...HEAD --oneline` for an overview).
4. Remove debug output, commented-out code, and `# TODO` markers that aren't tracked.
5. Verify the working tree is clean: `git status` should show nothing uncommitted.

## Checklist
- [ ] All gate commands pass.
- [ ] You have read every line of your own diff.
- [ ] No debug prints, commented code, or stray markers.
- [ ] Working tree is clean.
