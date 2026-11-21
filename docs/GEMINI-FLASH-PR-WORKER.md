# Gemini Flash PR worker: activation and controls

This workflow is staged as a pull request. Scheduled execution starts only after review and merge into the repository's default branch.

## One-time configuration

1. Add repository Actions secret named `GEMINI_API_KEY` containing a Gemini API key. Never commit or paste the key into an issue or prompt.
2. In repository Settings → Actions → General → Workflow permissions, enable “Allow GitHub Actions to create and approve pull requests”. The worker never approves or merges its own PRs.
3. Optional: add Actions variable `GEMINI_MODEL` with the allowed model ID. Default: `gemini-3.5-flash-lite`.
4. Merge the workflow PR. GitHub schedules it hourly at minute 07 UTC; it can also be started manually via workflow_dispatch.

## Separation of duties

- The model step gets the model API key only, and no GitHub write credentials.
- Gemini CLI v0.68.0 is pinned and restricts tools to read and edit files. No shell or web-fetch tools.
- Agent patches are bounded by path allowlist, size, symlink checks and `git diff --check`.
- Tests run in a separate step without the model key. A separate job with write-scoped `GITHUB_TOKEN` replays the verified artifact into a new branch and opens a **draft PR**.
- Issue/PR text and repository content are untrusted. No private Noosphere data or credentials are supplied.

## Known limitations

- The Gemini web GitHub connector is unrelated to headless Gemini CLI credentials.
- Flash-Lite and Gemini API free tiers have rate limits; do not assume unlimited API requests.
- The workflow requires the API secret and the repository Actions permission for PR creation.
- No API key or permissions means no autonomous PRs.
- Failed tests or no useful changes mean no new PR.