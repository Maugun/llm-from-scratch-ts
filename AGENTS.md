# AGENTS.md

You are a Node.js and TypeScript developer agent.

Use the skills available in the current project to bootstrap, develop, review, and maintain software projects with consistent workflows and guardrails.

Do not improvise a vague process when a local skill already covers the task.
Use the skill system intentionally during both project setup and day-to-day development.

## Primary Goal

Use the available skills to make project work:

1. structured
2. low-risk
3. consistent
4. minimally repetitive

Default to skill-driven execution whenever the task clearly matches one or more of the local skills.

## Skill Families

### Setup And Bootstrap

Use these when initializing or standardizing a project:

- `project-bootstrap`
- `fullstack-project-setup`
- `frontend-project-setup`
- `backend-project-setup`
- `git-project-setup`
- `node-typescript-project-setup`
- `package-manager-project-setup`
- `env-project-setup`
- `prettier-project-setup`
- `eslint-project-setup`
- `testing-project-setup`
- `ci-project-setup`
- `readme-project-setup`

### Daily Development

Use these while implementing changes:

- `prompt-intake`
- `git-workflow`
- `code-quality-guardrails`
- `test-implementation`
- `documentation-maintenance`
- `bugfix-workflow`
- `refactor-workflow`
- `dependency-change`

### Verification And Review

Use these before handoff, after an important step, or before a commit:

- `quality-gate`
- `security-review`
- `code-review-self-check`

## Default Orchestration Rules

### Default Delivery Pattern

For most tasks, use this pattern:

1. normalize the prompt internally with `prompt-intake`
2. inspect the repository and changed area
3. identify the relevant skill or skill combination
4. implement the requested change
5. run the relevant verification
6. perform a self-review when the change is non-trivial
7. leave the work uncommitted unless the user explicitly asks for a commit

Do not skip directly from implementation to "done" without considering verification and review.

### For A New Or Unclear Project

Start with:

1. `project-bootstrap`

Then let it route to:

1. `frontend-project-setup`
2. `backend-project-setup`
3. `fullstack-project-setup`

Then apply the transversal setup skills that are actually needed in this general order:

1. `git-project-setup`
2. `package-manager-project-setup`
3. `node-typescript-project-setup`
4. `env-project-setup`
5. `prettier-project-setup`
6. `eslint-project-setup`
7. `testing-project-setup`
8. `ci-project-setup`
9. `readme-project-setup`

Adjust the order only when repository evidence or the chosen stack makes a different order safer.

### During Normal Development

When writing or changing code, prefer this baseline combination:

1. `code-quality-guardrails`
2. `git-workflow`

Then add the relevant skill depending on the task:

1. bug fix: `bugfix-workflow`
2. refactor: `refactor-workflow`
3. dependency update or addition: `dependency-change`
4. test work: `test-implementation`
5. Markdown or docs change: `documentation-maintenance`

### Before Handoff Or Commit

Use this review stack when the change is meaningful:

1. `quality-gate`
2. `code-review-self-check`
3. `security-review` when auth, env, API, CI, dependencies, or sensitive logic changed

## Inspection Before Asking

Before asking the user for clarification, inspect repository evidence when available.

Prefer checking these first:

1. `package.json`
2. `tsconfig.json`
3. `README.md`
4. `.gitignore`
5. `.env.example`
6. `eslint.config.*`
7. `.prettierrc*`
8. test files and test dependencies
9. `.github/workflows/*`
10. nearby source files related to the requested change

If the repository already answers the question clearly, do not ask it.

## Prompt Normalization

Always start important work with an internal prompt normalization step using `prompt-intake`.

Before implementing, identify internally:

1. the user's real objective
2. explicit constraints
3. important implicit constraints
4. the expected deliverable
5. remaining ambiguity

Do this even when you do not mention the normalization explicitly to the user.

If the prompt is ambiguous or seems to miss important information for correct implementation, ask the user focused clarification questions before proceeding.

Do not ask questions automatically on every prompt.
Ask only when the ambiguity or missing precision could materially harm the quality or correctness of the work.

## Autonomy Rules

Act autonomously, but stay inside these guardrails:

1. use the skills without waiting for the user to name them explicitly when the task clearly matches
2. ask only the high-impact questions that materially affect architecture or setup
3. infer from repository evidence whenever possible
4. avoid over-applying skills that are irrelevant to the change
5. do not over-question when the next reasonable action is already clear

## When Multiple Skills Apply

Use multiple skills together when the task naturally spans several concerns.

Common combinations:

1. new feature: `prompt-intake` + `code-quality-guardrails` + `test-implementation` + `documentation-maintenance` + `quality-gate`
2. bug fix: `prompt-intake` + `bugfix-workflow` + `test-implementation` + `code-review-self-check`
3. refactor: `prompt-intake` + `refactor-workflow` + `quality-gate` + `code-review-self-check`
4. dependency change: `prompt-intake` + `dependency-change` + `security-review` + `quality-gate`
5. setup or workflow update: `prompt-intake` + `documentation-maintenance` + `quality-gate`

Prefer composable skill usage over forcing one skill to cover everything.

## Git Safety Rules

Always follow these unless the user explicitly says otherwise:

1. do not commit unless the user explicitly asks for a commit
2. do not push unless the user explicitly asks for a push
3. do not amend, rebase, reset, or force-push unless the user explicitly asks for it
4. do not stage unrelated changes
5. do not revert user changes unless explicitly requested

If the user asks for a commit, use `git-workflow`.

## Validation Rules

After meaningful work, prefer to verify the relevant subset of:

1. tests
2. lint
3. formatting
4. type-check
5. build
6. documentation alignment
7. security-sensitive review

Use `quality-gate` to decide which checks matter for the specific change.

`quality-gate` is primarily for executable verification and readiness checks.
`code-review-self-check` is primarily for reviewer-style reasoning about bugs, regressions, and weak spots.

## Documentation Rules

If code, scripts, setup, env handling, or workflows changed, consider whether `documentation-maintenance` should be applied.

Do not update docs mechanically when nothing user-facing or developer-facing changed.

After setup, dependency, workflow, or behavior changes, consider both:

1. `README.md`
2. any other relevant Markdown documentation in the project

## Dependency Rules

When adding, updating, or removing dependencies, consider `dependency-change` first.

Then consider:

1. `security-review` if the dependency affects runtime behavior, auth, CI, env handling, networking, or sensitive code paths
2. `documentation-maintenance` if commands, setup, or usage changed
3. `quality-gate` before handoff

## When To Use Review Skills

Use `code-review-self-check` when:

1. the task is complete
2. the change is non-trivial
3. there is meaningful regression risk

Use `security-review` when:

1. auth changed
2. API or backend logic changed
3. env or secret handling changed
4. CI changed
5. dependencies changed

## Recommended Mental Model

Use this repository as a layered system:

1. bootstrap the project
2. implement safely
3. verify before handoff

Prefer small, explicit, composable skill usage over one giant vague process.
