import importlib
import subprocess
import sys
import tomllib

import pytest

from compose_files import REPO_ROOT

PYPROJECT = REPO_ROOT / "pyproject.toml"
BANNED = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["ruff"]["lint"][
    "flake8-tidy-imports"
]["banned-api"]
CALLS = {
    "torch.export.load": "import torch\ntorch.export.load('a.pt2')\n",
    "torch.export.pt2_archive": "from torch.export.pt2_archive import _package\n",
    "torch._inductor.aoti_load_package": (
        "import torch\ntorch._inductor.aoti_load_package('a.pt2')\n"
    ),
    "torch._inductor.package": "from torch._inductor.package import load_package\n",
    "torch._export.aot_load": "import torch._export\ntorch._export.aot_load('a.so', 'cpu')\n",
    "torch._C._aoti.AOTIModelPackageLoader": (
        "import torch\ntorch._C._aoti.AOTIModelPackageLoader('a.pt2', 'm')\n"
    ),
    "torch.hub.load": "import torch\ntorch.hub.load('repo', 'model')\n",
    "torch.hub.load_state_dict_from_url": (
        "from torch.hub import load_state_dict_from_url\nload_state_dict_from_url('u')\n"
    ),
    "torch.distributed.checkpoint": "import torch.distributed.checkpoint as dcp\n",
    "torch.utils.model_zoo": "from torch.utils.model_zoo import load_url\n",
    "torch.compiler.load_cache_artifacts": (
        "import torch\ntorch.compiler.load_cache_artifacts(b'')\n"
    ),
    "torch.package": "from torch.package import PackageImporter\n",
    "torch.compiler.load_compiled_function": (
        "from torch.compiler import load_compiled_function\n"
    ),
}


CITATIONS = {
    "torch.utils.model_zoo": "CVE-2026-136431",
    "torch.compiler.load_cache_artifacts": "AGENTS.md H16",
    "torch.package": "AGENTS.md H16",
}
DEFAULT_CITATION = "CVE-2026-136426..136433"
EXTRA_SNIPPETS = {
    "torch.distributed.checkpoint": [
        "from torch.distributed.checkpoint import FileSystemReader\n",
        "from torch.distributed.checkpoint.format_utils import torch_save_to_dcp\n",
    ],
    "torch.utils.model_zoo": [
        "import torch\ntorch.utils.model_zoo.load_url('u')\n",
        "import torch.utils.model_zoo as zoo\nzoo.load_url('u')\n",
    ],
}


def ruff_banned_api_report(source: str) -> str:
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--config",
            str(PYPROJECT),
            "--select",
            "TID251",
            "--output-format",
            "concise",
            "--stdin-filename",
            "snippet.py",
            "-",
        ],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout


def resolves_in_installed_torch(dotted_name: str) -> bool:
    parts = dotted_name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            target = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for attribute in parts[split:]:
            target = getattr(target, attribute, None)
        return target is not None
    return False


def test_every_banned_loader_has_a_violating_snippet() -> None:
    assert set(CALLS) == set(BANNED)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_ruff_rejects_the_banned_loader(name: str) -> None:
    report = ruff_banned_api_report(CALLS[name])
    assert f"`{name}` is banned" in report
    assert CITATIONS.get(name, DEFAULT_CITATION) in report


@pytest.mark.parametrize(
    ("name", "source"),
    [(name, source) for name, sources in sorted(EXTRA_SNIPPETS.items()) for source in sources],
)
def test_ruff_rejects_every_way_into_the_banned_module(name: str, source: str) -> None:
    assert f"`{name}` is banned" in ruff_banned_api_report(source)


def test_ruff_still_allows_torch_load_with_weights_only() -> None:
    source = "import torch\ntorch.load('a.pt', weights_only=True)\n"
    assert "TID251" not in ruff_banned_api_report(source)


@pytest.mark.parametrize("name", sorted(BANNED))
def test_banned_name_exists_in_the_installed_torch(name: str) -> None:
    assert resolves_in_installed_torch(name)
