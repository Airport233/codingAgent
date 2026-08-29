# Start work: branch from main

1. Ensure you are on the latest main: `git checkout main && git pull`.
2. Create a branch named after the change: `feature/<short-kebab-name>` for features, `fix/<short-kebab-name>` for bug fixes, `docs/<short-kebab-name>` for documentation.
3. Branch names: lowercase, hyphenated, 3-40 chars, describe the concrete change (e.g. `feature/shell-classifier`, `fix/thinking-panel-stuck`).
4. Do not branch from another feature branch unless explicitly coordinating a stacked PR.

## Checklist
- [ ] `git status` is clean before branching.
- [ ] Branch name matches the convention.
- [ ] You are not on main/master when starting work.
