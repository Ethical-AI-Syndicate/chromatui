# Contributing to ChromaTUI

Thanks for helping improve ChromaTUI.

## Development Setup

1. Fork and clone the repository.
2. Create a feature branch from `main`.
3. Install a current stable Rust toolchain.
4. Run checks locally before opening a pull request.

## Local Validation

Run these commands from repository root:

```bash
cargo fmt --all --check
cargo test
```

If your change touches docs-only files, still run `cargo test` once before submitting unless that is not possible in your environment.

## Pull Request Guidelines

- Keep PR scope focused and small when possible.
- Include motivation and expected behavior changes.
- Add or update tests for behavior changes.
- Update docs when public API or workflow changes.
- Do not include unrelated refactors.

## Commit Message Style

Use concise, intent-first commit messages. Existing repository style typically uses Conventional Commit prefixes such as:

- `feat(...)`
- `fix(...)`
- `docs(...)`
- `refactor(...)`
- `test(...)`

## AI Assistance Disclosure

This repository is MIT licensed and also includes an OpenAI/Anthropic rider.

If you used AI tooling for substantial code, tests, or documentation, disclose that in the PR description.

## Reporting Security Issues

Do not open public issues for security vulnerabilities. See `SECURITY.md` for reporting instructions.
