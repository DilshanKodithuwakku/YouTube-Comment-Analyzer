# Security Policy

## Supported Versions

Security fixes are provided for the latest version available on the `main` branch.

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues.

If you discover a vulnerability, use GitHub’s private vulnerability reporting feature for this repository. Include:

- A clear description of the issue
- Steps to reproduce it
- The potential impact
- Any suggested fix, if available

We will review the report and aim to respond within 7 days.

## Security Practices

This project follows these practices:

- API keys and secrets are stored in `.env` files and excluded from Git.
- `.env.example` contains only placeholder values.
- Public YouTube comments are processed locally.
- Dependencies are defined in `requirements.txt`.
- Generated datasets and trained model files are excluded from the repository.

## Sensitive Information

Never commit any of the following:

- YouTube Data API keys
- Personal access tokens
- Passwords
- Private datasets
- User-identifying information

If a secret is accidentally committed, revoke or rotate it immediately.
