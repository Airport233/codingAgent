from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def normalize_sdk_base_url(configured_url: str) -> str:
    """Convert a Provider base URL or /v1/messages endpoint into an SDK base URL."""
    parsed = urlsplit(configured_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Provider Base URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("Provider Base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Provider Base URL must not contain a query or fragment")

    path = parsed.path.rstrip("/")
    for suffix in ("/v1/messages", "/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    normalized_path = f"{path.rstrip('/')}/" if path else "/"
    return urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))
