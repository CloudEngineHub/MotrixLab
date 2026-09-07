# Wiki AGENTS.md

This file defines the maintenance rules for the repository wiki content.

## Structure Rules

- The wiki uses a hierarchical document structure.
- Each directory level must include an `index.md` that indexes the more specific documents and subdirectories under that level.
- Content documents should be organized under the appropriate hierarchy instead of being placed arbitrarily.
- The wiki is currently organized by document purpose: `research/`, `design/`, and `plan/`.
- At the current stage, formal design output should default to `design/`; feature implementation plans should be placed in `plan/`; exploratory material belongs in `research/`.

## Content Document Rules

- Each content document must begin with a summary section.
- The summary should make the document's main topic and scope clear so an agent can quickly understand what the document is about.
- When a wiki document refers to other wiki documents, use normal inline Markdown links in the main text.
- Inline links should be added at appropriate points when they help navigation or clarify dependencies between documents.
- Content documents should describe the current design directly, rather than narrating the editing history or design-change process.
- Avoid writing process-oriented text such as why a previous version was changed, what was removed, or how the document evolved, unless that history is itself the topic of the document.
- Prefer presenting the current design, assumptions, interfaces, invariants, and rationale in their final form.
- New feature work must follow `design -> plan -> coding`.
- Design discussion outcomes should be written to `design/`; if a related topic document already exists, update it instead of creating a duplicate document.
- After the design is settled, create or update the corresponding implementation plan in `plan/`.
- Every plan document must include a TODO list, and completed items must be checked off as work progresses.
- Once a plan has been fully implemented, remove the completed plan document.

## Update Rules

- When documents are added, removed, or relocated, the corresponding `index.md` files must be updated.
- The relevant `index.md` should reflect the current document structure and links for that hierarchy level.
- When a document changes category, the source and destination `index.md` files must both be updated.
- If `research/` or `plan/` remain empty, their `index.md` files should still be kept as category entry points and should describe when documents belong there.
