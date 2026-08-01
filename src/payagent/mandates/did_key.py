"""`did:key` encoding for Ed25519 public keys — multicodec + multibase, no registry.

A `did:key` is self-certifying: the identifier is derived from the public key itself, so
"whose key is this" and "does this signature verify" are the same question answered by the
same bytes. Spec: https://w3c-ccg.github.io/did-method-key/. Ed25519 public keys carry the
multicodec prefix `0xed01` (varint-encoded), then the 32 raw key bytes, base58btc-encoded with
a leading `z` (the multibase prefix for base58btc) and a leading `did:key:`.

No `base58` dependency: base58btc is ~15 lines and the only alternative library adds a whole
package for one function this package will call maybe twice a day.
"""

from __future__ import annotations

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

_ED25519_MULTICODEC_PREFIX = bytes((0xED, 0x01))
_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _base58btc_encode(data: bytes) -> str:
    """Base58 (Bitcoin alphabet) encode, preserving leading zero bytes as leading '1's."""
    n = int.from_bytes(data, "big")
    encoded = ""
    while n > 0:
        n, remainder = divmod(n, 58)
        encoded = _BASE58_ALPHABET[remainder] + encoded
    leading_zeros = len(data) - len(data.lstrip(b"\x00"))
    return "1" * leading_zeros + encoded


def did_key_from_ed25519_public_key(public_key: Ed25519PublicKey) -> str:
    """Render `public_key` as a `did:key:z...` identifier."""
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return "did:key:z" + _base58btc_encode(_ED25519_MULTICODEC_PREFIX + raw)
