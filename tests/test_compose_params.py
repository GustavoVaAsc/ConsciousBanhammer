import re

from compose_files import judge_command, judge_params

COMMIT_HASH = re.compile(r"[0-9a-f]{40}")


def test_compose_model_equals_params() -> None:
    assert judge_command()[0] == judge_params()["model"]


def test_compose_revisions_equal_params() -> None:
    flags = judge_command()[1]
    assert flags["revision"] == judge_params()["revision"]
    assert flags["tokenizer-revision"] == judge_params()["tokenizer_revision"]


def test_compose_context_length_equals_params() -> None:
    assert int(judge_command()[1]["max-model-len"]) == judge_params()["max_model_len"]


def test_compose_gpu_memory_utilization_equals_params() -> None:
    expected = judge_params()["gpu_memory_utilization"]
    assert float(judge_command()[1]["gpu-memory-utilization"]) == expected


def test_revisions_are_full_commit_hashes() -> None:
    params = judge_params()
    assert COMMIT_HASH.fullmatch(str(params["revision"]))
    assert COMMIT_HASH.fullmatch(str(params["tokenizer_revision"]))
