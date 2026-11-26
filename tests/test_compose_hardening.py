import pytest

from compose_files import compose, environment_of, judge_command, services

SERVICES = ("judge", "judge-proxy", "work")


@pytest.mark.parametrize("name", SERVICES)
def test_service_drops_capabilities_and_blocks_privilege_gain(name: str) -> None:
    service = services()[name]
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]


@pytest.mark.parametrize("name", SERVICES)
def test_service_never_restarts_on_its_own(name: str) -> None:
    assert services()[name]["restart"] == "no"


def test_vllm_publishes_no_port() -> None:
    assert "ports" not in services()["judge"]


def test_vllm_sits_only_on_the_isolated_internal_network() -> None:
    network = compose()["networks"]["judge_net"]
    assert services()["judge"]["networks"] == ["judge_net"]
    assert network["internal"] is True
    assert network["driver_opts"]["com.docker.network.bridge.gateway_mode_ipv4"] == "isolated"


def test_work_is_not_on_the_judge_network() -> None:
    assert "networks" not in services()["work"]


def test_proxy_bridges_the_internal_and_default_networks() -> None:
    assert services()["judge-proxy"]["networks"] == ["judge_net", "default"]


def test_proxy_is_the_only_published_port_and_binds_loopback() -> None:
    published = {name: s["ports"] for name, s in services().items() if "ports" in s}
    assert published == {"judge-proxy": ["127.0.0.1:8000:8080"]}


def test_proxy_waits_for_a_healthy_judge() -> None:
    assert services()["judge-proxy"]["depends_on"] == {"judge": {"condition": "service_healthy"}}


def test_proxy_root_filesystem_is_read_only_with_a_tmpfs() -> None:
    proxy = services()["judge-proxy"]
    assert proxy["read_only"] is True
    # D3: nginx keeps its pid file and temp paths under /tmp on a read-only root
    assert proxy["tmpfs"] == ["/tmp"]  # noqa: S108


def test_api_key_stays_out_of_the_vllm_command_line() -> None:
    assert not any("api-key" in argument for argument in services()["judge"]["command"])


def test_judge_runs_offline_without_telemetry() -> None:
    environment = environment_of("judge")
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert environment["VLLM_NO_USAGE_STATS"] == "1"
    assert environment["DO_NOT_TRACK"] == "1"


def test_work_disables_hub_telemetry() -> None:
    assert environment_of("work")["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_judge_mounts_the_model_cache_read_only() -> None:
    (volume,) = services()["judge"]["volumes"]
    assert volume.endswith(":/hf:ro")


def test_judge_serves_without_api_docs_and_with_prefix_caching() -> None:
    flags = services()["judge"]["command"]
    assert "--disable-fastapi-docs" in flags
    assert "--enable-prefix-caching" in flags
    assert judge_command()[1]["generation-config"] == "vllm"


def test_judge_entrypoint_refuses_a_short_key() -> None:
    script = services()["judge"]["entrypoint"][2]
    assert 'test "$${#VLLM_API_KEY}" -ge 32' in script
    assert "exit 1" in script


def test_work_hides_dotenv_and_mounts_git_read_only() -> None:
    volumes = services()["work"]["volumes"]
    assert "./.git:/workspace/.git:ro" in volumes
    assert "/dev/null:/workspace/.env:ro" in volumes
