"""Validate basic PRSC Knowledge Gram invariants using only the standard library.

This is intentionally a small convenience validator, not a replacement for the
JSON Schema validator or semantic/cache-compatibility checks described in paper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_TOP_LEVEL = {
    "schema_version", "gram_id", "model_compatibility", "pattern_ref",
    "residual", "relations", "provenance", "activation", "sensitivity",
    "validity",
}
VALID_ENCODINGS = {"text", "structured_json", "latent_ref", "binary_ref", "patch"}
VALID_MODES = {"exact_source", "reconstruct_then_validate", "precomputed_kv", "latent_activation"}
VALID_SENSITIVITY = {"public", "internal", "confidential", "restricted"}


class ValidationError(ValueError):
    """Raised when a K-gram violates the minimal local contract."""


def validate_gram(data: Any) -> None:
    """Raise ValidationError if the record violates the basic local contract."""
    if not isinstance(data, dict):
        raise ValidationError("top-level JSON value must be an object")
    missing = REQUIRED_TOP_LEVEL - data.keys()
    if missing:
        raise ValidationError(f"missing top-level fields: {', '.join(sorted(missing))}")

    for key in ("schema_version", "gram_id"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValidationError(f"{key} must be a non-empty string")

    model = data["model_compatibility"]
    if not isinstance(model, dict):
        raise ValidationError("model_compatibility must be an object")
    for key in ("base_model", "base_model_fingerprint", "tokenizer_fingerprint", "encoder_revision"):
        if not isinstance(model.get(key), str) or not model[key].strip():
            raise ValidationError(f"model_compatibility.{key} must be a non-empty string")

    pattern = data["pattern_ref"]
    if pattern is not None:
        if not isinstance(pattern, dict):
            raise ValidationError("pattern_ref must be an object or null")
        for key in ("pattern_id", "pattern_type", "catalogue_revision"):
            if not isinstance(pattern.get(key), str) or not pattern[key].strip():
                raise ValidationError(f"pattern_ref.{key} must be a non-empty string")

    residual = data["residual"]
    if not isinstance(residual, dict):
        raise ValidationError("residual must be an object")
    if residual.get("encoding") not in VALID_ENCODINGS:
        raise ValidationError(f"residual.encoding must be one of {sorted(VALID_ENCODINGS)}")
    confidence = residual.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValidationError("residual.confidence must be a number in [0, 1]")
    if "payload" not in residual:
        raise ValidationError("residual.payload is required")

    relations = data["relations"]
    if not isinstance(relations, list):
        raise ValidationError("relations must be an array")
    for i, relation in enumerate(relations):
        if not isinstance(relation, dict):
            raise ValidationError(f"relations[{i}] must be an object")
        for key in ("target_gram_id", "relation_type"):
            if not isinstance(relation.get(key), str) or not relation[key].strip():
                raise ValidationError(f"relations[{i}].{key} must be a non-empty string")
        if not isinstance(relation.get("required_for_activation"), bool):
            raise ValidationError(f"relations[{i}].required_for_activation must be boolean")

    provenance = data["provenance"]
    if not isinstance(provenance, list) or not provenance:
        raise ValidationError("provenance must be a non-empty array")
    for i, source in enumerate(provenance):
        if not isinstance(source, dict):
            raise ValidationError(f"provenance[{i}] must be an object")
        for key in ("source_id", "uri", "content_sha256", "captured_at"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                raise ValidationError(f"provenance[{i}].{key} must be a non-empty string")

    activation = data["activation"]
    if not isinstance(activation, dict):
        raise ValidationError("activation must be an object")
    if activation.get("preferred_mode") not in VALID_MODES:
        raise ValidationError(f"activation.preferred_mode must be one of {sorted(VALID_MODES)}")
    if not isinstance(activation.get("exact_fallback_uri"), str) or not activation["exact_fallback_uri"].strip():
        raise ValidationError("activation.exact_fallback_uri must be a non-empty string")

    if data["sensitivity"] not in VALID_SENSITIVITY:
        raise ValidationError(f"sensitivity must be one of {sorted(VALID_SENSITIVITY)}")

    validity = data["validity"]
    if not isinstance(validity, dict):
        raise ValidationError("validity must be an object")
    for key in ("archive_revision", "valid_from"):
        if not isinstance(validity.get(key), str) or not validity[key].strip():
            raise ValidationError(f"validity.{key} must be a non-empty string")
    if validity.get("valid_until") is not None and not isinstance(validity["valid_until"], str):
        raise ValidationError("validity.valid_until must be a string or null")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("Usage: python -m prsc.validate PATH.json", file=sys.stderr)
        return 2
    path = Path(args[0])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        validate_gram(data)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"VALID: {data['gram_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
