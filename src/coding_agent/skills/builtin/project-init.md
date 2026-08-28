+++
name = "project-init"
description = "Draft safe project memory instructions without writing files"
+++

# Project Memory Draft Workflow

Create a concise, evidence-based candidate for this repository's `CODING_AGENT.md`.

1. Read an existing `CODING_AGENT.md` when present, then inspect the top-level tree, package metadata, executable entry points, tests, CI, and relevant configuration.
2. Identify verified run, test, lint, format, and build commands. Do not invent commands that are not supported by repository files.
3. Summarize the architecture, module boundaries, platform constraints, coding conventions, and safety rules that a cold-start coding agent needs.
4. Distinguish repository facts from inferences, and call out important uncertainties rather than guessing.
5. Return a compact fenced Markdown candidate headed `# CODING_AGENT.md draft`, followed by a short note telling the user how to request saving it.

This workflow is strictly preview-only. Use only read-oriented inspection such as file reading, code search, and read-only Git status or diff. Do not invoke write, edit, directory creation, or mutating shell operations. Do not create, modify, or overwrite any file, even when no existing memory file is present.
