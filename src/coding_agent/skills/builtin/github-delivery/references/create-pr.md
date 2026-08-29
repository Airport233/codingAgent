# Create PR: gh pr create, title, body

1. Push the branch: `git push -u origin <branch-name>`.
2. Create the PR:
   ```
   gh pr create --title "type(scope): subject" --body "$(cat <<'EOF'
   ## What
   One-line summary of the change.

   ## Why
   The problem or motivation. Link the issue/ticket if applicable.

   ## How
   Key implementation decisions and the files/modules touched.

   ## Verification
   - Gate commands run and green (list them).
   - Manual checks performed (describe them).
   EOF
   )"
   ```
3. Title uses Conventional Commits format, matching the branch intent.
4. Add reviewers, labels, and link the issue if the project uses them.
5. Do not force-push after requesting review — it invalidates the reviewer's context.

## Checklist
- [ ] Branch is pushed.
- [ ] PR title uses Conventional Commits.
- [ ] Body has What / Why / How / Verification.
- [ ] Reviewers assigned if applicable.
