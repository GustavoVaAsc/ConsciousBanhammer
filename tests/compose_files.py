import re
from functools import cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROXY_CONF = REPO_ROOT / "docker" / "judge-proxy.conf"

LOCATION = re.compile(r"location\s+((?:[=~^]+\s+)?\S+)\s*\{")
HOST_ENTRY = re.compile(r'^\s*"([^"]+)"\s+1;', re.MULTILINE)
PROXY_PASS = re.compile(r"proxy_pass\s+(\S+);")
REQUEST_URI_RULE = re.compile(r'if\s*\(\$request_uri\s+(!?~)\s+"([^"]+)"\)\s*\{\s*return\s+(\d+);')


@cache
def load_yaml(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((REPO_ROOT / name).read_text(encoding="utf-8"))
    return loaded


def compose() -> dict[str, Any]:
    return load_yaml("compose.yaml")


def services() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = compose()["services"]
    return loaded


def judge_params() -> dict[str, Any]:
    loaded: dict[str, Any] = load_yaml("params.yaml")["judge"]
    return loaded


def judge_command() -> tuple[str, dict[str, str]]:
    model, *options = services()["judge"]["command"]
    flags = {
        key.removeprefix("--"): value
        for key, _, value in (option.partition("=") for option in options)
    }
    return model, flags


def environment_of(name: str) -> dict[str, str]:
    pairs = (entry.partition("=") for entry in services()[name]["environment"])
    return {key: value for key, _, value in pairs}


@cache
def proxy_conf() -> str:
    return PROXY_CONF.read_text(encoding="utf-8")


def allowed_hosts() -> set[str]:
    return set(HOST_ENTRY.findall(proxy_conf()))


def proxy_pass_targets() -> list[str]:
    return PROXY_PASS.findall(proxy_conf())


def location_patterns() -> list[str]:
    return LOCATION.findall(proxy_conf())


def rejects_raw_uri(raw_uri: str) -> bool:
    rules = REQUEST_URI_RULE.findall(proxy_conf())
    return any(
        (re.search(pattern, raw_uri) is None) == (operator == "!~")
        for operator, pattern, _status in rules
    )
