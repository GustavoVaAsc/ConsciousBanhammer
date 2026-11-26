import textwrap
from pathlib import Path

import pytest

from hygiene_checker import HygieneChecker, RepositoryScanner, SourceFile

KNOWN_MODULES = frozenset(
    {
        "banhammer",
        "banhammer.core",
        "banhammer.core.constants",
        "banhammer.messages",
        "banhammer.messages.constants",
        "banhammer.messages.dedup",
        "banhammer.rules",
        "banhammer.rules.constants",
        "banhammer.rules.atomize",
    }
)


def rules_found(
    snippet: str,
    *,
    module: str | None = None,
    is_package_init: bool = False,
    executable: bool = False,
    test_package: str | None = None,
) -> list[str]:
    file = SourceFile(
        path="snippet.py",
        text=textwrap.dedent(snippet),
        executable=executable,
        module=module,
        is_package_init=is_package_init,
        test_package=test_package,
    )
    return [violation.rule for violation in HygieneChecker(KNOWN_MODULES).check(file)]


@pytest.mark.parametrize(
    "snippet",
    [
        "# D7: V0 admits severity 0 only\nx = 1\n",
        "# Proposal 6.1: the classifier sees the parent\nx = 1\n",
        "# Proposal 4: top-level section\nx = 1\n",
        "# arXiv:2407.21783: Llama Guard recipe\nx = 1\n",
        "# https://github.com/huggingface/peft/issues/1: adapter bug\nx = 1\n",
        "x = 1  # noqa: PLC0415\n",
        "x = 1  # noqa: E501, F401\n",
        "x = f()  # type: ignore[arg-type]\n",
        "x = f()  # type: ignore[arg-type, misc]\n",
        "x = f()  # type: ignore[arg-type]  # noqa: E501\n",
        "x = 1  # pragma: no cover\n",
        "key = 'abc'  # pragma: allowlist secret\n",
        "x = '# looks like a comment but is a string'\n",
    ],
)
def test_allowed_comments_pass(snippet: str) -> None:
    assert rules_found(snippet) == []


@pytest.mark.parametrize(
    "snippet",
    [
        "# plain explanation\nx = 1\n",
        "# D7\nx = 1\n",
        "# D7:\nx = 1\n",
        "# D7:no space\nx = 1\n",
        "# D: no number\nx = 1\n",
        "# Proposal: no section\nx = 1\n",
        "# TODO: later\nx = 1\n",
        "# x = commented_out_code()\n",
        "x = 1  # noqa\n",
        "x = 1  # noqa: PLC0415 gpu extra is absent\n",
        "x = 1  # noqa: plc0415\n",
        "x = f()  # type: ignore\n",
        "x = f()  # type: ignore[arg-type] because torch\n",
        "x = 1  # pragma: no branch\n",
        "# fmt: off\nx = 1\n",
        "x = 1  # fmt: skip\n",
        "x = f()  # type: ignore[arg-type]  # explanation\n",
        "# D7: because # noqa: F401\nx = 1\n",
        "x = 1  # D7: reason # noqa: F401\n",
        "# https://example.org/x: reason # type: ignore[misc]\nx = 1\n",
        "x = 1  # explanation\n",
    ],
)
def test_other_comments_are_rejected(snippet: str) -> None:
    assert rules_found(snippet) == ["H1"]


def test_shebang_is_allowed_on_the_first_line_of_an_executable_script() -> None:
    assert rules_found("#!/usr/bin/env python\nx = 1\n", executable=True) == []


def test_shebang_is_rejected_in_a_file_that_is_not_executable() -> None:
    assert rules_found("#!/usr/bin/env python\nx = 1\n") == ["H1"]


def test_shebang_is_rejected_after_the_first_line() -> None:
    assert rules_found("x = 1\n#!/usr/bin/env python\n", executable=True) == ["H1"]


@pytest.mark.parametrize(
    "snippet",
    [
        '"""Module docstring."""\n',
        'def f() -> None:\n    """Function docstring."""\n',
        'async def f() -> None:\n    """Coroutine docstring."""\n',
        'class A:\n    """Class docstring."""\n',
        'class A:\n    """Doc."""\n\n    x: int = 1\n    """Attribute docstring."""\n',
        'class A:\n    """Doc."""\n\n    x = 1\n    """Attribute docstring."""\n',
        "x = 1\nprint(x)\n",
        "def f() -> None:\n    ...\n",
    ],
)
def test_docstrings_and_attribute_docstrings_pass(snippet: str) -> None:
    assert rules_found(snippet) == []


@pytest.mark.parametrize(
    "snippet",
    [
        'x = 1\n"""Module-level attribute docstring."""\n',
        'def f() -> None:\n    """Doc."""\n    x = 1\n    """A comment in disguise."""\n',
        'def f() -> None:\n    """Doc."""\n    """Second docstring."""\n',
        'def f() -> None:\n    x = 1\n    "explains x"\n',
        "def f() -> None:\n    x = 1\n    f'explains {x}'\n",
        'if True:\n    "first statement of an if is not a docstring"\n',
        'class A:\n    """Doc."""\n\n    def g(self) -> None: ...\n    """After a method."""\n',
        'class A:\n    """Doc."""\n\n    """Second docstring."""\n',
        'for item in items:\n    "explains the loop"\n',
        'try:\n    x = 1\nexcept ValueError:\n    "swallowed"\n',
        'def f() -> None:\n    """Doc."""\n    x = 1\n    b"explains x"\n',
        'x = 1\nb"bytes statement"\n',
        'class A:\n    """Doc."""\n\n    def g(self) -> None: ...\n    b"after a method"\n',
    ],
)
def test_string_statements_that_are_not_docstrings_are_rejected(snippet: str) -> None:
    assert "H1" in rules_found(snippet)


@pytest.mark.parametrize("snippet", ["while True:\n    pass\n", "while 1:\n    pass\n"])
def test_while_true_is_rejected(snippet: str) -> None:
    assert rules_found(snippet) == ["H4"]


@pytest.mark.parametrize(
    "snippet",
    [
        "while pending:\n    pass\n",
        "for attempt in range(3):\n    pass\n",
        "while not done():\n    pass\n",
    ],
)
def test_bounded_loops_pass(snippet: str) -> None:
    assert rules_found(snippet) == []


FUNCTIONS_ONLY = '"""Metrics."""\n\n\ndef qwk() -> float:\n    """Kappa."""\n    return 0.0\n'


def test_module_of_public_functions_without_the_functional_line_is_rejected() -> None:
    assert rules_found(FUNCTIONS_ONLY, module="banhammer.evaluation.metrics") == ["H3"]


def test_async_public_function_counts_as_a_public_function() -> None:
    snippet = '"""Fetch."""\n\n\nasync def fetch() -> None:\n    """Fetch."""\n'
    assert rules_found(snippet, module="banhammer.labeling.client") == ["H3"]


def test_functional_line_with_a_reason_passes() -> None:
    snippet = FUNCTIONS_ONLY.replace(
        '"""Metrics."""', '"""Metrics.\n\nFunctional module: stateless metric formulas.\n"""'
    )
    assert rules_found(snippet, module="banhammer.evaluation.metrics") == []


def test_functional_line_without_a_reason_is_rejected() -> None:
    snippet = FUNCTIONS_ONLY.replace('"""Metrics."""', '"""Metrics.\n\nFunctional module:\n"""')
    assert rules_found(snippet, module="banhammer.evaluation.metrics") == ["H3"]


@pytest.mark.parametrize(
    "wrapper",
    [
        "if TYPE_CHECKING:\n{body}",
        "if sys.platform == 'linux':\n    pass\nelse:\n{body}",
        "try:\n{body}\nexcept ImportError:\n    pass",
        "try:\n    pass\nexcept ImportError:\n{body}",
        "try:\n    pass\nfinally:\n{body}",
    ],
)
def test_public_function_inside_a_top_level_if_or_try_counts(wrapper: str) -> None:
    body = '    def metric() -> float:\n        """Metric."""\n        return 0.0'
    snippet = '"""Metrics."""\n\n' + wrapper.format(body=body) + "\n"
    assert rules_found(snippet, module="banhammer.evaluation.metrics") == ["H3"]


def test_class_inside_a_top_level_if_satisfies_the_class_rule() -> None:
    snippet = (
        '"""Metrics."""\n\n'
        "if TYPE_CHECKING:\n"
        "    class Scorer:\n"
        '        """Scores."""\n\n'
        "    def metric() -> float:\n"
        '        """Metric."""\n'
        "        return 0.0\n"
    )
    assert rules_found(snippet, module="banhammer.evaluation.metrics") == []


def test_function_nested_in_a_function_is_not_top_level() -> None:
    snippet = (
        '"""Helpers."""\n\n\n'
        "def _outer() -> None:\n"
        '    """Outer."""\n\n'
        "    def inner() -> None:\n"
        '        """Inner."""\n'
    )
    assert rules_found(snippet, module="banhammer.evaluation.helpers") == []


def test_module_with_a_class_needs_no_functional_line() -> None:
    snippet = FUNCTIONS_ONLY + '\n\nclass Scorer:\n    """Scores."""\n'
    assert rules_found(snippet, module="banhammer.evaluation.metrics") == []


def test_module_with_only_private_functions_needs_no_functional_line() -> None:
    snippet = '"""Helpers."""\n\n\ndef _helper() -> None:\n    """Help."""\n'
    assert rules_found(snippet, module="banhammer.evaluation.helpers") == []


def test_package_init_is_exempt_from_the_functional_line() -> None:
    assert rules_found(FUNCTIONS_ONLY, module="banhammer.evaluation", is_package_init=True) == []


def test_functional_line_is_not_required_outside_src() -> None:
    assert rules_found(FUNCTIONS_ONLY, module=None) == []


@pytest.mark.parametrize(
    "statement",
    [
        "from banhammer.core.constants import Verdict",
        "from banhammer.messages import Message",
        "from banhammer import messages",
        "from banhammer.rules.constants import RULE_FIELD",
        "from banhammer.rules import constants",
        "import banhammer.core.constants",
        "from . import atomize",
        "from .constants import RULE_FIELD",
        "import torch",
    ],
)
def test_public_and_own_imports_pass(statement: str) -> None:
    assert rules_found(statement + "\n", module="banhammer.rules.canonicalize") == []


@pytest.mark.parametrize(
    "statement",
    [
        "from banhammer.messages.constants import MESSAGE_FIELD",
        "from banhammer.messages import constants",
        "from banhammer.messages import dedup as dedupe",
        "import banhammer.messages.dedup",
        "from banhammer.messages._private import helper",
    ],
)
def test_importing_a_private_module_of_another_subpackage_is_rejected(statement: str) -> None:
    assert rules_found(statement + "\n", module="banhammer.rules.canonicalize") == ["H16"]


def test_package_init_of_a_subpackage_is_held_to_the_same_rule() -> None:
    snippet = "from banhammer.messages.constants import MESSAGE_FIELD\n"
    assert rules_found(snippet, module="banhammer.rules", is_package_init=True) == ["H16"]


def test_import_under_type_checking_is_held_to_the_same_rule() -> None:
    snippet = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from banhammer.messages.constants import MESSAGE_FIELD\n"
    )
    assert rules_found(snippet, module="banhammer.rules.canonicalize") == ["H16"]


def test_import_rule_does_not_apply_to_a_test_outside_a_subpackage_directory() -> None:
    snippet = "from banhammer.messages.constants import MESSAGE_FIELD\n"
    assert rules_found(snippet, module=None) == []


def test_test_of_a_subpackage_may_not_reach_into_another_subpackage_privates() -> None:
    snippet = "from banhammer.messages.constants import MESSAGE_FIELD\n"
    assert rules_found(snippet, test_package="rules") == ["H16"]


@pytest.mark.parametrize(
    "statement",
    [
        "from banhammer.rules.constants import RULE_FIELD",
        "from banhammer.rules import constants",
        "from banhammer.core.constants import Verdict",
        "from banhammer.messages import Message",
    ],
)
def test_test_of_a_subpackage_may_import_its_own_modules_core_and_public_exports(
    statement: str,
) -> None:
    assert rules_found(statement + "\n", test_package="rules") == []


def test_violation_reports_file_line_and_rule() -> None:
    file = SourceFile(path="a.py", text="x = 1\nwhile True:\n    pass\n")
    (violation,) = HygieneChecker(KNOWN_MODULES).check(file)
    assert str(violation).startswith("a.py:2: H4: ")


def write(root: Path, relative: str, text: str, *, mode: int = 0o644) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def test_scanner_names_modules_and_flags_executables(tmp_path: Path) -> None:
    write(tmp_path, "src/banhammer/__init__.py", '"""Root."""\n')
    write(tmp_path, "src/banhammer/core/__init__.py", '"""Core."""\n')
    write(tmp_path, "src/banhammer/core/seeding.py", '"""Seeding."""\n')
    write(tmp_path, "tests/core/test_seeding.py", "x = 1\n")
    write(tmp_path, "tests/tool.py", "#!/usr/bin/env python\n", mode=0o755)
    write(tmp_path, "scripts/hook.py", "x = 1\n")
    write(tmp_path, "conftest.py", "x = 1\n")
    write(tmp_path, "docs/outside.py", "# not scanned\n")
    write(tmp_path, ".venv/lib/outside.py", "# not scanned\n")
    write(tmp_path, "src/banhammer/core/notes.txt", "not python\n")

    by_path = {file.path: file for file in RepositoryScanner(tmp_path).files()}

    assert sorted(by_path) == [
        "conftest.py",
        "scripts/hook.py",
        "src/banhammer/__init__.py",
        "src/banhammer/core/__init__.py",
        "src/banhammer/core/seeding.py",
        "tests/core/test_seeding.py",
        "tests/tool.py",
    ]
    assert by_path["src/banhammer/__init__.py"].module == "banhammer"
    assert by_path["src/banhammer/core/__init__.py"].module == "banhammer.core"
    assert by_path["src/banhammer/core/__init__.py"].is_package_init
    assert by_path["src/banhammer/core/seeding.py"].module == "banhammer.core.seeding"
    assert by_path["src/banhammer/core/seeding.py"].subpackage == "core"
    assert by_path["tests/core/test_seeding.py"].module is None
    assert by_path["tests/core/test_seeding.py"].subpackage == "core"
    assert by_path["tests/tool.py"].subpackage is None
    assert by_path["scripts/hook.py"].subpackage is None
    assert by_path["conftest.py"].subpackage is None
    assert by_path["tests/tool.py"].executable
    assert not by_path["tests/core/test_seeding.py"].executable


def test_scanner_and_checker_flag_a_bad_comment_in_scripts_and_root_conftest(
    tmp_path: Path,
) -> None:
    write(tmp_path, "scripts/hook.py", "x = 1  # explanation\n")
    write(tmp_path, "conftest.py", "while True:\n    pass\n")
    files = RepositoryScanner(tmp_path).files()
    checker = HygieneChecker.for_files(files)
    found = [(v.path, v.rule) for file in files for v in checker.check(file)]
    assert found == [("conftest.py", "H4"), ("scripts/hook.py", "H1")]


def test_scanner_and_checker_flag_a_test_importing_another_subpackage_private_module(
    tmp_path: Path,
) -> None:
    write(tmp_path, "src/banhammer/messages/__init__.py", '"""Messages."""\n')
    write(tmp_path, "src/banhammer/messages/constants.py", '"""Constants."""\n')
    write(
        tmp_path,
        "tests/rules/test_atomize.py",
        "from banhammer.messages import constants\n",
    )
    files = RepositoryScanner(tmp_path).files()
    checker = HygieneChecker.for_files(files)
    found = [(v.path, v.rule) for file in files for v in checker.check(file)]
    assert found == [("tests/rules/test_atomize.py", "H16")]


def test_checker_built_from_scanned_files_catches_a_cross_subpackage_import(
    tmp_path: Path,
) -> None:
    write(tmp_path, "src/banhammer/rules/__init__.py", '"""Rules."""\n')
    write(tmp_path, "src/banhammer/messages/__init__.py", '"""Messages."""\n')
    write(tmp_path, "src/banhammer/messages/constants.py", '"""Constants."""\n')
    write(
        tmp_path,
        "src/banhammer/rules/atomize.py",
        '"""Atomize."""\n\nfrom banhammer.messages import constants\n',
    )
    files = RepositoryScanner(tmp_path).files()
    checker = HygieneChecker.for_files(files)
    found = [v for file in files for v in checker.check(file)]
    assert [(v.path, v.rule) for v in found] == [("src/banhammer/rules/atomize.py", "H16")]
