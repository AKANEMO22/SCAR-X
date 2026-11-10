# Acceptance attempt failed

The authenticated Hermes route mounted, but SCAR-X rejected mutation without its session token. This attempt is retained. The plugin was corrected to fetch and retain the existing mutation token only in its Python backend; the successful later run is recorded separately. No security guard was removed.
