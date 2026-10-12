"""Minimal PII masking (emails and phone numbers) applied before anything is stored."""
from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_CANDIDATE_RE = re.compile(r"(?<![\w.])\+?\(?\d[\d\s().\-]{7,}\d(?!\w)")
ALLOWED_EMAIL_DOMAINS = ("aurorasync.example",)  # the product's own support address is not personal data


def _mask_email(match: re.Match) -> str:
    domain = match.group(0).rsplit("@", 1)[-1].lower()
    return match.group(0) if domain in ALLOWED_EMAIL_DOMAINS else "[EMAIL]"


def _mask_phone(match: re.Match) -> str:
    digits = sum(ch.isdigit() for ch in match.group(0))
    return "[PHONE]" if 9 <= digits <= 15 else match.group(0)


def mask_pii(text: str) -> str:
    if not text:
        return text
    return PHONE_CANDIDATE_RE.sub(_mask_phone, EMAIL_RE.sub(_mask_email, text))
