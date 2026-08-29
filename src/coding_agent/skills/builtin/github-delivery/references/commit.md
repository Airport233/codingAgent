# Commit: atomic, conventional

1. Make atomic commits: one logical change per commit. If a commit does two unrelated things, split it.
2. Use Conventional Commits format: `type(scope): subject` where type is `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`, `style`, or `perf`.
3. Subject line: imperative mood, lowercase, no trailing period, max 72 chars.
4. Body (optional): explain *why*, not *what*. Wrap at 72 chars.
5. Do not commit:
   - Secrets, API keys, `.env` files, or credentials.
   - Generated files (build output, `.coverage`, `dist/`, `__pycache__/`).
   - IDE-specific config that should be local (`.idea/`, `.vscode/settings.json` unless shared).
6. Do not amend or rebase pushed commits. If you need to fix a pushed commit, add a new commit.

## Checklist
- [ ] Each commit is a single logical change.
- [ ] Subject uses Conventional Commits format.
- [ ] No secrets or generated files staged.
