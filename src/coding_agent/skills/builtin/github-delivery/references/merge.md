# Merge: squash, cleanup, sync

1. Prefer **squash-merge** for feature branches into main. This keeps history linear and each main commit is a complete, reviewable unit.
2. Use the PR title (Conventional Commits format) as the squash commit message.
3. After merge:
   - Delete the remote branch: `gh pr merge --squash --delete-branch` or the GitHub UI.
   - Switch to main: `git checkout main`.
   - Pull latest: `git pull origin main`.
   - Delete the local branch: `git branch -d <branch-name>`.
4. Do not merge with failing CI. If CI is flaky, re-run the failing job and explain in a comment.
5. Do not force-push to main to "fix" history. If main needs repair, coordinate with the team first.

## Checklist
- [ ] CI is green on the PR.
- [ ] Squash-merge used (or the team's agreed strategy).
- [ ] Remote branch deleted.
- [ ] Local main is up to date.
- [ ] Local feature branch deleted.
