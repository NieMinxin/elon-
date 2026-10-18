from __future__ import annotations

import base64
import hashlib
import hmac
import time
from urllib.parse import quote_plus

import httpx


class DingTalkNotifier:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        webhook: str | None,
        secret: str | None,
    ) -> None:
        self.client = client
        self.webhook = webhook
        self.secret = secret

    @property
    def enabled(self) -> bool:
        return bool(self.webhook)

    def _signed_url(self) -> str:
        assert self.webhook is not None
        if not self.secret:
            return self.webhook
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}\n{self.secret}".encode()
        signature = hmac.new(self.secret.encode(), message, hashlib.sha256).digest()
        separator = "&" if "?" in self.webhook else "?"
        return (
            f"{self.webhook}{separator}timestamp={timestamp}"
            f"&sign={quote_plus(base64.b64encode(signature).decode())}"
        )

    async def send(self, *, title: str, text: str) -> None:
        if not self.enabled:
            return
        response = await self.client.post(
            self._signed_url(),
            json={
                "msgtype": "markdown",
                "markdown": {"title": title, "text": f"### {title}\n\n{text}"},
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errcode") != 0:
            raise RuntimeError(f"DingTalk rejected alert: {payload}")
