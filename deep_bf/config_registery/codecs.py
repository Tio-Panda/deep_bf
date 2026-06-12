import ast
import json
from typing import Any


def encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def decode_json_dict(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = ast.literal_eval(raw)

    if not isinstance(value, dict):
        raise TypeError(f"Expected dict payload, got {type(value).__name__}")

    return value


def decode_json_list_str(raw: str) -> list[str]:
    value = json.loads(raw)
    if not isinstance(value, list):
        raise TypeError(f"Expected list payload, got {type(value).__name__}")

    return [str(item) for item in value]


def decode_kernel(raw: str) -> tuple[int, int]:
    value = json.loads(raw)
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError("Kernel payload must be a JSON list with two elements")

    return int(value[0]), int(value[1])


def decode_bool(raw: int) -> bool:
    return bool(int(raw))
