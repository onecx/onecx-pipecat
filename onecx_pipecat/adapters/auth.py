"""Authentication adapter: exchange a credential for a typed Principal.

The wire contract: the client's first message carries the user's
``apm-principal-token`` (and the browser's ``Authorization`` value). The
adapter verifies that token — signature, issuer and expiry for a real
deployment — and reports expiry so a session can end with a typed error when
the tokens lapse without a refresh. ``ClientCredentialsAuth`` obtains Pipecat's
own OIDC client-credentials token for outbound calls to OneCX services (the
apm-principal-token is propagated on top of it, BFF-style).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Protocol

import httpx

from onecx_pipecat.config import AuthSettings
from onecx_pipecat.models import Principal

# Refresh the service token this long before it expires.
_TOKEN_EXPIRY_SKEW_SECONDS = 30.0
# Unknown ``kid`` values refetch the JWKS at most this often (rotation, not a DoS lever).
_JWKS_MIN_REFETCH_SECONDS = 60.0


def _keycloak_endpoint(settings: AuthSettings, path: str) -> str:
    return f"{settings.issuer.rstrip('/')}/protocol/openid-connect/{path}"


class AuthenticationError(Exception):
    """Raised when a credential cannot be validated."""


class ServiceTokenError(httpx.HTTPError):
    """The token endpoint answered without a usable access token.

    An ``httpx.HTTPError`` so HTTP adapters fail closed exactly as for any
    other transport failure.
    """


class ClientCredentialsAuth(httpx.Auth):
    """Attach Pipecat's OIDC client-credentials token as ``Authorization: Bearer``.

    The token is requested from ``auth.token_endpoint`` (default: the Keycloak
    endpoint of ``auth.issuer``) with ``client_id``/``client_secret`` and cached
    until shortly before it expires.
    """

    def __init__(self, settings: AuthSettings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client
        self._token = ""
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    async def token(self) -> str:
        async with self._lock:
            if self._token and time.time() < self._expires_at - _TOKEN_EXPIRY_SKEW_SECONDS:
                return self._token
            client = self._client or httpx.AsyncClient()
            try:
                response = await client.post(
                    self._settings.token_endpoint or _keycloak_endpoint(self._settings, "token"),
                    data={"grant_type": "client_credentials"},
                    auth=httpx.BasicAuth(self._settings.client_id, self._settings.client_secret),
                )
                response.raise_for_status()
                body = response.json()
                self._token = str(body["access_token"])
                self._expires_at = time.time() + float(body.get("expires_in", 60))
            except (KeyError, TypeError, ValueError) as exc:
                raise ServiceTokenError("token endpoint returned no access_token") from exc
            finally:
                if self._client is None:
                    await client.aclose()
            return self._token

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        request.headers["Authorization"] = f"Bearer {await self.token()}"
        yield request


class AuthAdapter(Protocol):
    """Replaceable authentication boundary."""

    async def authenticate(self, token: str) -> Principal: ...

    def is_expired(self, token: str) -> bool:
        """True once the session token has lapsed and cannot be used."""
        ...

    async def refresh(self, token: str) -> Principal:
        """Validate and return the principal for a refreshed token."""
        ...

    async def service_token(self) -> str:
        """A bearer token Pipecat uses for its own outbound service calls."""
        ...


class JwtAuthAdapter:
    """Verify ``apm-principal-token`` against Keycloak's JWKS.

    Validates the signature (RS256), issuer and ``exp`` the way
    ``onecx-permission-svc`` does, using the configured issuer and key set.
    ``settings.jwks`` is either an embedded JWKS JSON document or a URL; when
    empty, the Keycloak ``certs`` endpoint of ``auth.issuer`` is fetched. The
    token's ``kid`` header selects which JWKS key verifies it, and an unknown
    ``kid`` refetches the set (key rotation), rate-limited.
    """

    def __init__(
        self,
        settings: AuthSettings | None = None,
        service_auth: ClientCredentialsAuth | None = None,
    ) -> None:
        self._settings = settings or AuthSettings()
        self._service_auth = service_auth or ClientCredentialsAuth(self._settings)
        self._client: httpx.AsyncClient | None = None
        self._keys: list[dict] | None = None
        self._keys_by_kid: dict[str, object] | None = None
        self._jwks_url: str | None = None
        self._fetched_at = float("-inf")
        jwks = self._settings.jwks.strip()
        if jwks.startswith(("{", "[")):
            self._keys = self._normalize(jwks)
        else:
            self._jwks_url = jwks or _keycloak_endpoint(self._settings, "certs")

    def set_client(self, client: httpx.AsyncClient) -> None:
        """Inject the client used to fetch the JWKS (tests avoid a live network call)."""
        self._client = client

    def set_jwks(self, jwks: object) -> None:
        """Pin an explicit key set; it is never refetched."""
        self._keys = self._normalize(jwks)
        self._keys_by_kid = None
        self._jwks_url = None

    async def _ensure_keys(self, kid: str | None) -> None:
        """Fetch the JWKS when none is loaded or the token's ``kid`` is unknown."""
        if self._jwks_url is None:
            return
        if self._keys is not None and (kid is None or kid in {k.get("kid") for k in self._keys}):
            return
        if time.monotonic() - self._fetched_at < _JWKS_MIN_REFETCH_SECONDS:
            return
        self._fetched_at = time.monotonic()
        client = self._client or httpx.AsyncClient()
        try:
            response = await client.get(self._jwks_url)
            response.raise_for_status()
            self._keys = self._normalize(response.json())
            self._keys_by_kid = None
        except (httpx.HTTPError, ValueError) as exc:
            raise AuthenticationError(f"JWKS unavailable: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()

    @staticmethod
    def _normalize(jwks: object) -> list[dict]:
        """Turn a JWKS document (dict or JSON string) into its ``keys`` list.

        A bare JWK (dict without a ``keys`` member) or a list of JWKs is also
        accepted, so callers may pass either the full document or the keys.
        """
        if isinstance(jwks, str):
            jwks = json.loads(jwks)
        if isinstance(jwks, dict):
            return list(jwks.get("keys", []))
        if isinstance(jwks, list):
            return [k for k in jwks if isinstance(k, dict)]
        return []

    def _principal(self, claims: dict) -> Principal:
        subject = claims.get(self._settings.token_claim_subject)
        if not isinstance(subject, str) or not subject:
            raise AuthenticationError("token missing subject claim")
        tenant = claims.get("tenant_id", claims.get("azp", "local"))
        roles = claims.get("roles") or []
        if not isinstance(roles, list):
            roles = []
        return Principal(subject=subject, tenant_id=str(tenant), roles=[str(r) for r in roles])

    def _key_for_kid(self, kid: str | None) -> object:
        """The public key matching the token's ``kid`` header, or None.

        A JWKS routinely carries several keys (rotation); the token selects the
        one that signed it via the ``kid`` in its header. A token with no ``kid``
        falls back to the first key, and an unknown ``kid`` resolves to ``None``
        so the token is rejected instead of trying the wrong key.
        """
        if not self._keys:
            return None
        if kid is None:
            return self._first_key()
        if self._keys_by_kid is None:
            self._keys_by_kid = {
                k.get("kid"): k for k in self._keys if isinstance(k.get("kid"), str)
            }
        candidate = self._keys_by_kid.get(kid)
        if candidate is None:
            return None
        return self._build_key(candidate)

    def _first_key(self) -> object:
        """The first usable key (used only for tokens that carry no ``kid``)."""
        for key in self._keys:
            built = self._build_key(key)
            if built is not None:
                return built
        return None

    @staticmethod
    def _build_key(key: dict) -> object:
        """Build the public key from a single JWK dict, or None if unusable."""
        try:
            from jwt import PyJWK

            return PyJWK.from_dict(key).key
        except Exception:  # noqa: BLE001 - no usable key configured
            return None

    async def authenticate(self, token: str) -> Principal:
        try:
            import jwt
        except ImportError as exc:  # pragma: no cover - guarded in tests
            raise AuthenticationError("jwt verification unavailable") from exc
        try:
            kid = jwt.get_unverified_header(token).get("kid")
        except Exception:  # noqa: BLE001 - malformed header; the token is rejected below
            kid = None
        await self._ensure_keys(kid)
        key = self._key_for_kid(kid)
        if key is None:
            raise AuthenticationError("no JWKS key matches the token")
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                issuer=self._settings.issuer,
                # Even without an expected audience, PyJWT rejects any token that
                # has an `aud` claim, so the audience check must be switched off.
                options={"verify_aud": False},
            )
        except Exception as exc:  # noqa: BLE001 - all validation failures are auth failures
            raise AuthenticationError(f"token verification failed: {exc}") from exc
        return self._principal(claims)

    def is_expired(self, token: str) -> bool:
        """Verify the signature and read ``exp`` without enforcing it.

        An unparseable, unknown, or untrusted token is treated as lapsed.
        """
        try:
            import jwt
        except ImportError:  # pragma: no cover
            return True
        try:
            kid = jwt.get_unverified_header(token).get("kid")
        except Exception:  # noqa: BLE001 - malformed header; an untrusted token is lapsed
            kid = None
        key = self._key_for_kid(kid)
        if key is None:
            return True
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                issuer=self._settings.issuer,
                options={"verify_exp": False, "verify_aud": False},
            )
            return int(claims.get("exp", 0)) <= time.time()
        except Exception:  # noqa: BLE001 - untrusted token is treated as lapsed
            return True

    async def refresh(self, token: str) -> Principal:
        return await self.authenticate(token)

    async def service_token(self) -> str:
        return await self._service_auth.token()
