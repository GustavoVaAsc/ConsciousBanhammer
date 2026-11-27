"""Static checks behind the AGENTS.md hygiene rules H1, H3, H4 and H16."""

import ast
import io
import os
import re
import tokenize
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

CITED_WHY = re.compile(r"^# (Proposal \d+(\.\d+)*|D\d+|arXiv:\S+|https?://\S+): \S.*$")
PRAGMAS = (
    re.compile(r"^# noqa: [A-Z]+\d+(, [A-Z]+\d+)*$"),
    re.compile(r"^# type: ignore\[[a-z-]+(, [a-z-]+)*\]$"),
    re.compile(r"^# pragma: (no cover|allowlist secret)$"),
)
PRAGMA_SEPARATOR = re.compile(r"\s+(?=# )")
FURTHER_COMMENT = " # "
FUNCTIONAL_MODULE_LINE = re.compile(r"^Functional module: \S")
SHEBANG = "#!"
ROOT_PACKAGE = "banhammer"
CORE_SUBPACKAGE = "core"
SCANNED_DIRECTORIES = ("src", "tests", "scripts")
SCANNED_ROOT_PATTERN = "*.py"
STATEMENT_LISTS = ("body", "orelse", "finalbody")
TRY_NODES = (ast.Try, ast.TryStar)
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
TEST_PACKAGE_DEPTH = 2
TORCH_LOADERS = frozenset({"torch.load", "torch.serialization.load"})
WEIGHTS_ONLY_KEYWORD = "weights_only"


@dataclass(frozen=True)
class Violation:
    """One breach of a hygiene rule, located by file and line."""

    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.message}"


@dataclass(frozen=True)
class SourceFile:
    """A Python file under check; `module` is its dotted name when it lives in `src/`."""

    path: str
    text: str
    executable: bool = False
    module: str | None = None
    is_package_init: bool = False
    test_package: str | None = None

    @property
    def subpackage(self) -> str | None:
        """The first-level subpackage of `banhammer` this file belongs to, if any.

        A test under `tests/<pkg>/` belongs to `<pkg>`.
        """
        if self.module is None:
            return self.test_package
        parts = self.module.split(".")
        package = parts if self.is_package_init else parts[:-1]
        return package[1] if len(package) >= 2 else None


class HygieneChecker:
    """Applies the H1, H3, H4 and H16 checks to source files."""

    def __init__(self, known_modules: frozenset[str]) -> None:
        self._known_modules = known_modules

    @classmethod
    def for_files(cls, files: list[SourceFile]) -> "HygieneChecker":
        return cls(frozenset(file.module for file in files if file.module is not None))

    def check(self, file: SourceFile) -> list[Violation]:
        tree = ast.parse(file.text, filename=file.path)
        return [
            *self._comments(file),
            *self._string_statements(file, tree),
            *self._unbounded_loops(file, tree),
            *self._functional_module(file, tree),
            *self._subpackage_imports(file, tree),
            *self._unsafe_torch_loads(file, tree),
        ]

    def _comments(self, file: SourceFile) -> Iterator[Violation]:
        for token in tokenize.generate_tokens(io.StringIO(file.text).readline):
            if token.type == tokenize.COMMENT and not self._is_allowed_comment(token, file):
                message = f"comment is neither a cited why nor a coded pragma: {token.string}"
                yield Violation(file.path, token.start[0], "H1", message)

    def _is_allowed_comment(self, token: tokenize.TokenInfo, file: SourceFile) -> bool:
        text = token.string.rstrip()
        if token.start == (1, 0) and text.startswith(SHEBANG):
            return file.executable
        if CITED_WHY.match(text):
            return FURTHER_COMMENT not in text
        return all(
            any(pragma.match(part) for pragma in PRAGMAS) for part in PRAGMA_SEPARATOR.split(text)
        )

    def _string_statements(self, file: SourceFile, tree: ast.Module) -> Iterator[Violation]:
        for owner in ast.walk(tree):
            for field in STATEMENT_LISTS:
                statements = getattr(owner, field, None)
                if isinstance(statements, list):
                    yield from self._stray_strings(file, owner, field, statements)

    def _stray_strings(
        self, file: SourceFile, owner: ast.AST, field: str, statements: list[ast.stmt]
    ) -> Iterator[Violation]:
        for index, statement in enumerate(statements):
            if _is_string_statement(statement) and not _is_documentation(
                owner, field, statements, index
            ):
                message = "string statement is neither a docstring nor an attribute docstring"
                yield Violation(file.path, statement.lineno, "H1", message)

    def _unbounded_loops(self, file: SourceFile, tree: ast.Module) -> Iterator[Violation]:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.While)
                and isinstance(node.test, ast.Constant)
                and node.test.value
            ):
                message = "`while True` is not allowed; bound the loop or use an iterator"
                yield Violation(file.path, node.lineno, "H4", message)

    def _functional_module(self, file: SourceFile, tree: ast.Module) -> Iterator[Violation]:
        if file.module is None or file.is_package_init:
            return
        definitions = list(_top_level_statements(tree.body))
        has_class = any(isinstance(node, ast.ClassDef) for node in definitions)
        public_functions = [
            node
            for node in definitions
            if isinstance(node, FUNCTION_NODES) and not node.name.startswith("_")
        ]
        docstring_lines = (ast.get_docstring(tree) or "").splitlines()
        has_line = any(FUNCTIONAL_MODULE_LINE.match(line) for line in docstring_lines)
        if public_functions and not has_class and not has_line:
            message = "public functions without classes need a `Functional module: <reason>.` line"
            yield Violation(file.path, public_functions[0].lineno, "H3", message)

    def _subpackage_imports(self, file: SourceFile, tree: ast.Module) -> Iterator[Violation]:
        own = file.subpackage
        if own is None:
            return
        for node in ast.walk(tree):
            for target in self._imported_modules(node):
                if _is_foreign_private_module(own, target):
                    message = f"`{target}` is not public: import from `{_subpackage_of(target)}`"
                    yield Violation(file.path, node.lineno, "H16", message)

    def _unsafe_torch_loads(self, file: SourceFile, tree: ast.Module) -> Iterator[Violation]:
        bindings = _import_bindings(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and _resolved_name(node.func, bindings) in TORCH_LOADERS
                and not _passes_weights_only_true(node)
            ):
                message = "`torch.load` must be called with the literal `weights_only=True`"
                yield Violation(file.path, node.lineno, "H16", message)

    def _imported_modules(self, node: ast.AST) -> Iterator[str]:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module
            for alias in node.names:
                submodule = f"{node.module}.{alias.name}"
                if submodule in self._known_modules:
                    yield submodule


def _import_bindings(tree: ast.Module) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                bindings[alias.asname or root] = alias.name if alias.asname else root
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bindings


def _resolved_name(expression: ast.expr, bindings: dict[str, str]) -> str | None:
    attributes: list[str] = []
    while isinstance(expression, ast.Attribute):
        attributes.append(expression.attr)
        expression = expression.value
    if not isinstance(expression, ast.Name) or expression.id not in bindings:
        return None
    return ".".join([bindings[expression.id], *reversed(attributes)])


def _passes_weights_only_true(call: ast.Call) -> bool:
    return any(
        keyword.arg == WEIGHTS_ONLY_KEYWORD
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in call.keywords
    )


def _top_level_statements(statements: list[ast.stmt]) -> Iterator[ast.stmt]:
    for statement in statements:
        yield statement
        if isinstance(statement, ast.If):
            yield from _top_level_statements(statement.body)
            yield from _top_level_statements(statement.orelse)
        elif isinstance(statement, TRY_NODES):
            yield from _top_level_statements(statement.body)
            for handler in statement.handlers:
                yield from _top_level_statements(handler.body)
            yield from _top_level_statements(statement.orelse)
            yield from _top_level_statements(statement.finalbody)


def _is_string_statement(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr):
        return False
    value = statement.value
    return isinstance(value, ast.JoinedStr) or (
        isinstance(value, ast.Constant) and isinstance(value.value, str | bytes)
    )


def _is_documentation(owner: ast.AST, field: str, statements: list[ast.stmt], index: int) -> bool:
    if field != "body":
        return False
    if index == 0 and isinstance(owner, DOCSTRING_OWNERS):
        return True
    return (
        isinstance(owner, ast.ClassDef)
        and index > 0
        and isinstance(statements[index - 1], ast.Assign | ast.AnnAssign)
    )


def _is_foreign_private_module(own: str, target: str) -> bool:
    parts = target.split(".")
    return parts[0] == ROOT_PACKAGE and len(parts) > 2 and parts[1] not in (own, CORE_SUBPACKAGE)


def _subpackage_of(target: str) -> str:
    return ".".join(target.split(".")[:2])


class RepositoryScanner:
    """Reads the `.py` files of `src/`, `tests/`, `scripts/` and the repository root."""

    def __init__(self, repository_root: Path) -> None:
        self._root = repository_root

    def files(self) -> list[SourceFile]:
        paths = sorted(
            [
                *self._root.glob(SCANNED_ROOT_PATTERN),
                *(
                    path
                    for name in SCANNED_DIRECTORIES
                    for path in (self._root / name).rglob("*.py")
                ),
            ]
        )
        return [self._read(path) for path in paths]

    def _read(self, path: Path) -> SourceFile:
        source_root = self._root / "src"
        in_source = path.is_relative_to(source_root)
        parts = path.relative_to(source_root).with_suffix("").parts if in_source else ()
        is_init = bool(parts) and parts[-1] == "__init__"
        module_parts = parts[:-1] if is_init else parts
        return SourceFile(
            path=str(path.relative_to(self._root)),
            text=path.read_text(encoding="utf-8"),
            executable=os.access(path, os.X_OK),
            module=".".join(module_parts) if in_source else None,
            is_package_init=is_init,
            test_package=self._test_package(path),
        )

    def _test_package(self, path: Path) -> str | None:
        relative = path.relative_to(self._root).parts
        in_package_directory = len(relative) > TEST_PACKAGE_DEPTH and relative[0] == "tests"
        return relative[1] if in_package_directory else None
