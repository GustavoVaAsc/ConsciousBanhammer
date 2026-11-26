import re

import pytest

from compose_files import (
    allowed_hosts,
    location_patterns,
    proxy_conf,
    proxy_pass_targets,
    rejects_raw_uri,
    services,
)

VLLM_ORIGIN = "http://judge:8000"


def published_ports() -> tuple[str, str]:
    (mapping,) = services()["judge-proxy"]["ports"]
    _address, host_port, container_port = mapping.split(":")
    return host_port, container_port


def test_host_allow_list_matches_the_published_port_and_service_name() -> None:
    host_port, container_port = published_ports()
    assert allowed_hosts() == {
        f"localhost:{host_port}",
        f"127.0.0.1:{host_port}",
        f"judge-proxy:{container_port}",
    }


def test_proxy_listens_on_the_published_container_port() -> None:
    container_port = published_ports()[1]
    assert re.search(rf"listen\s+{container_port}\s+default_server;", proxy_conf())


def test_unexpected_host_is_rejected_before_any_location() -> None:
    assert re.search(r"\$judge_host_allowed\s*=\s*0\)\s*\{\s*return\s+403;", proxy_conf())


def test_every_proxy_pass_carries_a_uri_part_so_nginx_forwards_the_normalized_uri() -> None:
    targets = proxy_pass_targets()
    assert len(targets) == 2
    assert all(re.fullmatch(rf"{re.escape(VLLM_ORIGIN)}/\S*", target) for target in targets)


def test_only_health_and_v1_are_forwarded() -> None:
    assert location_patterns() == ["= /health", "^~ /v1/", "/"]
    assert proxy_pass_targets() == [f"{VLLM_ORIGIN}/health", f"{VLLM_ORIGIN}/v1/"]


def test_default_location_returns_404() -> None:
    assert re.search(r"location\s+/\s*\{\s*return\s+404;\s*\}", proxy_conf())


@pytest.mark.parametrize(
    "raw_uri",
    [
        "/metrics/../v1/",
        "/metrics/../health",
        "/metrics%2F..%2Fv1%2F",
        "/metrics%2f..%2fv1%2f",
        "/v1/%2e%2e/metrics",
        "/v1/%2E%2E/metrics",
        "/invocations/../v1/models",
        "/v1//models",
        "//metrics",
        "/v1/models%5cadmin",
        "/v1\\models",
        "v1/models",
    ],
)
def test_ambiguous_raw_uris_are_rejected(raw_uri: str) -> None:
    assert rejects_raw_uri(raw_uri)


@pytest.mark.parametrize(
    "raw_uri",
    [
        "/health",
        "/health?verbose=1",
        "/v1/models",
        "/v1/models?limit=2&after=a%2Fb",
        "/v1/chat/completions",
        "/v1/models/Qwen",
        "/metrics",
        "/docs",
    ],
)
def test_ordinary_raw_uris_are_not_rejected_here(raw_uri: str) -> None:
    assert not rejects_raw_uri(raw_uri)
