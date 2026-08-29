---
name: project-init
description: Draft safe project memory instructions without writing files. Use when the user runs /init or asks to create, refresh, or preview repository guidance for future coding-agent sessions.
---

# Project Memory Draft Workflow

Create a concise, evidence-based candidate for this repository's `CODING_AGENT.md`.

1. Read `references/memory-template.md` with `read_skill_resource` when you are ready to structure the draft.
2. Read an existing `CODING_AGENT.md` when present, then inspect the top-level tree, package metadata, executable entry points, tests, CI, and relevant configuration.
3. Identify verified run, test, lint, format, and build commands. Do not invent commands that are not supported by repository files.
4. Summarize the architecture, module boundaries, platform constraints, coding conventions, and safety rules that a cold-start coding agent needs.
5. Distinguish repository facts from inferences, and call out important uncertainties rather than guessing.

This workflow is strictly preview-only. Use only read-oriented inspection such as file reading, code search, and read-only Git status or diff. Do not invoke write, edit, directory creation, or mutating shell operations. Do not create, modify, or overwrite any file, even when no existing memory file is present.
