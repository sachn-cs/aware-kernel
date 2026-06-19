# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within aware-kernel, please send an
email to **[INSERT SECURITY EMAIL]**. All security vulnerabilities will be
promptly addressed.

**Please do not report security vulnerabilities through public GitHub issues.**

### What to include

When reporting a vulnerability, please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)

### Response timeline

- **Acknowledgment**: Within 48 hours of your report.
- **Initial assessment**: Within 1 week.
- **Resolution**: Depends on severity, but we aim for 2-4 weeks for critical issues.

## Disclosure Policy

We follow a coordinated disclosure process:

1. **Report received**: We acknowledge receipt and begin investigation.
2. **Verification**: We reproduce and verify the vulnerability.
3. **Fix developed**: We develop and test a fix.
4. **Release**: We release a patched version.
5. **Disclosure**: We publish a security advisory after the fix is available.

We ask that you give us reasonable time to address the issue before public
disclosure.

## Security Best Practices

When using aware-kernel in production:

- Keep dependencies updated (`pip install --upgrade numpy scipy scikit-learn`).
- Use virtual environments to isolate installations.
- Review `pyproject.toml` for dependency pinning.
- Run `safety check` or `pip-audit` regularly to scan for known vulnerabilities.

## Scope

This security policy applies to the `aware-kernel` Python package distributed
via PyPI. It does not apply to:

- Third-party forks or modified versions
- Code in example scripts (which are for demonstration only)
- Issues in upstream dependencies (report those to their respective maintainers)
