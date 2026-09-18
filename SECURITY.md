# Security

## Reporting a vulnerability

Report privately through GitHub's
[Security Advisories](https://github.com/maher-nakesh/Vervemint/security/advisories/new)
page, or by email to <barakodabcc@gmail.com>. Please do not open a public
issue for a vulnerability. Expect a first reply within a week.

Include what you did, what happened, and the version (`GET /health`
returns it).

## What to know before deploying

Vervemint is a single-instance, single-tenant application. A few
properties are by design, not bugs:

- **Secrets are stored as readable JSON.** API keys, the Telegram bot
  token and the web UI password live in `data/credentials.json` (owner
  only, `0600`) so the running app can use them. Protect the data volume,
  or inject them from a secrets manager as environment variables, which
  take precedence and lock the field in Settings.
- **One shared API token** guards every endpoint except `GET /health`.
  It is generated on the first start and compared with
  `hmac.compare_digest`. There are no user accounts and no per-user
  scopes.
- **The web UI password is off by default.** Publishing the UI on the
  internet? Set `VERVEMINT_UI_REQUIRE_PASSWORD=true` and put the UI
  behind the `proxy` profile so it is only reachable over HTTPS.
- **Guardrails are pattern-based.** They catch obvious prompt injection,
  not paraphrased attempts. Do not treat them as a security boundary.
- **Logs contain user questions and document text** (`logs/`), and
  `logs/llm.log` holds full prompts and replies unless
  `log_llm_messages: false`. API keys and bot tokens are never logged.
  Keep `logs/` private.

Compose runs the containers unprivileged, read-only, with `cap_drop:
ALL` and `no-new-privileges`, and publishes ports on `127.0.0.1` only.
