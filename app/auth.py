"""API-key auth, same scheme as the multi-tenant RAG service: keys are shown once, only a peppered hash is stored."""
import hashlib
import hmac
import secrets

from fastapi import Header, HTTPException

from app import registry
from app.config import ADMIN_KEY, API_KEY_PEPPER


def hash_secret(plaintext: str) -> str:
    return hashlib.sha256((plaintext + API_KEY_PEPPER).encode()).hexdigest()


def new_secret(prefix: str) -> tuple[str, str]:
    plaintext = f"{prefix}{secrets.token_urlsafe(32)}"
    return plaintext, hash_secret(plaintext)


def get_tenant(x_api_key: str = Header(..., alias="X-API-Key")) -> dict:
    tenant = registry.get_tenant_by_key_hash(hash_secret(x_api_key))
    if tenant is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant


def require_admin(x_admin_key: str = Header(default="")) -> None:
    if ADMIN_KEY and not hmac.compare_digest(x_admin_key, ADMIN_KEY):
        raise HTTPException(status_code=401, detail="Admin key required")


def verify_secret(plaintext: str, stored_hash: str | None) -> bool:
    return bool(stored_hash) and hmac.compare_digest(hash_secret(plaintext), stored_hash)
