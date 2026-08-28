+++
name = "project-map"
description = "Map a repository's entry points, architecture, tests, and workflows"
+++

# Project Map Workflow

1. Read repository guidance and list the top-level tree without traversing generated or dependency directories.
2. Identify language, package manager, executable entry points, configuration, and test commands from source files.
3. Trace the main runtime path through a small number of important modules and interfaces.
4. Locate tests, fixtures, CI, persistence, external integrations, and platform-specific code.
5. Distinguish facts read from the repository from reasonable inferences.
6. Produce a compact map containing: how to run, architecture flow, module ownership, test strategy, and risky extension points.

This is a read-oriented workflow. Do not edit the project unless the user explicitly includes an implementation task.
