# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement (CLA). You (or your employer) retain the copyright to your
contribution; this simply gives us permission to use and redistribute your
contributions as part of the project. Head over to
<https://cla.developers.google.com/> to see your current agreements on file or
to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted
one (even if it was for a different project), you probably don't need to do it
again.

## Code Reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose.

Here is how the process works for external contributors:

1. **Create a PR**: Submit your changes as a Pull Request on GitHub.
   * Please consider updating the `CHANGELOG.md` file if your changes are user-facing.
2. **Automation**: Our automation will automatically add the `"pull ready"`
   label to your PR (for new PRs) and import it into Google's internal code
   review system.
3. **Review**: A member of the Orbax team will review your change.
4. **Merge**: Once the review is approved and submitted internally, your PR
   will be automatically merged on GitHub.

## Code Style & Linting

Orbax follows the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
(80-character lines, 2-space indents, Google-style docstrings). Three tools
enforce it, wired together through the
[`pre-commit`](https://pre-commit.com/) framework:

| Tool | Role | Config |
|---|---|---|
| [`pylint`](https://pylint.readthedocs.io/) | Linter | `.pylintrc` |
| [`isort`](https://pycqa.github.io/isort/) | Import order | `[tool.isort]` in `pyproject.toml` |
| [`pytype`](https://github.com/google/pytype) | Static Type Inferece & Verification | `.pre-commit-config.yaml` |

Install the hooks once, then they run on every `git commit`:

```bash
pip install pre-commit
pre-commit install
```

To check your modified files before submitting a PR:

```bash
pre-commit run --files $(git diff --name-only origin/main)
```

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).
