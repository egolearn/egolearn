"""Generate reference pages from source without importing the documented package."""

from __future__ import annotations

import ast
import html
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = "egolearn.config.registry"
GITHUB = "https://github.com/egolearn/egolearn"

# Curated entry points use their definitions, never the package's lazy exports.
API = {
    "Pipeline": (
        "egolearn.pipeline.build",
        "egolearn.pipeline.validate",
        "egolearn.pipeline.Pipeline",
        "egolearn.pipeline.PipelineError",
        "egolearn.core.context.Context",
    ),
    "Configuration": (
        "egolearn.config.instantiate.load_config",
        "egolearn.config.instantiate.instantiate",
        "egolearn.config.registry.Registry",
        "egolearn.config.registry.lookup",
    ),
    "Evaluation": (
        "egolearn.evaluation.checkpoint.evaluate_checkpoint",
        "egolearn.evaluation.checkpoint.load_model",
        "egolearn.evaluation.evaluator.evaluate",
        "egolearn.evaluation.evaluator.EvaluationResult",
        "egolearn.evaluation.evaluator.EvaluationError",
        "egolearn.evaluation.benchmark.Benchmark",
        "egolearn.evaluation.probe.Probe",
    ),
}


class Definition(NamedTuple):
    module: str
    path: Path
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    source: str


class Component(NamedTuple):
    target: str
    definition: Definition
    signature: str
    constructor: str | None


class ReferencePage(NamedTuple):
    content: str
    source: str


class SourceIndex:
    """Index local definitions and imports for registry and constructor lookup."""

    def __init__(self, root: Path):
        self.root = root
        self.modules: dict[str, tuple[Path, str, ast.Module]] = {}
        self.definitions: dict[str, Definition] = {}
        self.imports: dict[str, dict[str, str]] = {}
        for path in sorted((root / "src/egolearn").rglob("*.py")):
            parts = path.relative_to(root / "src").with_suffix("").parts
            module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            self.modules[module] = (path, source, tree)
            imports = self.imports[module] = {}
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    package = module.split(".") if path.name == "__init__.py" else parts[:-1]
                    prefix = list(package[: len(package) - node.level + 1]) if node.level else []
                    if node.module:
                        prefix.extend(node.module.split("."))
                    for alias in node.names:
                        imports[alias.asname or alias.name] = ".".join([*prefix, alias.name])
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports[alias.asname or alias.name.split(".")[0]] = (
                            alias.name if alias.asname else alias.name.split(".")[0]
                        )
                elif isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                    self.definitions[f"{module}.{node.name}"] = Definition(
                        module, path, node, source
                    )

    def resolve(self, module: str, name: str) -> str:
        """Resolve local names and re-export aliases without loading modules."""
        qualified = f"{module}.{name}"
        if (
            name == "object"
            and qualified not in self.definitions
            and name not in self.imports[module]
        ):
            return "builtins.object"
        seen = set()
        while qualified not in seen:
            if qualified in self.definitions:
                return qualified
            seen.add(qualified)
            parts = qualified.split(".")
            for split in range(len(parts) - 1, 0, -1):
                aliases = self.imports.get(".".join(parts[:split]), {})
                if parts[split] in aliases:
                    resolved = ".".join([aliases[parts[split]], *parts[split + 1 :]])
                    if resolved == qualified:
                        return qualified
                    qualified = resolved
                    break
            else:
                return qualified
        raise ValueError(f"cyclic source alias: {qualified}")

    def registries(self) -> dict[str, str]:
        tree = self.modules[REGISTRY][2]
        return {
            f"{REGISTRY}.{node.targets[0].id}": ast.literal_eval(node.value.args[0])
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "Registry"
        }

    def mro(self, name: str, visiting: tuple[str, ...] = ()) -> list[str]:
        """Compute Python's C3 order for local classes, retaining external bases."""
        if name in visiting:
            raise ValueError(f"cyclic class inheritance: {name}")
        definition = self.definitions.get(name)
        if definition is None:
            return [name]
        node = definition.node
        if not isinstance(node, ast.ClassDef):
            raise ValueError(f"base is not a class: {name}")
        bases = [self.resolve(definition.module, ast.unparse(base)) for base in node.bases]
        sequences = [self.mro(base, (*visiting, name)) for base in bases] + [bases.copy()]
        result = [name]
        while sequences := [seq for seq in sequences if seq]:
            candidate = next(
                (seq[0] for seq in sequences if all(seq[0] not in s[1:] for s in sequences)),
                None,
            )
            if candidate is None:
                raise ValueError(f"inconsistent class inheritance: {name}")
            result.append(candidate)
            for seq in sequences:
                if seq[0] == candidate:
                    seq.pop(0)
        return result

    def constructor(self, definition: Definition) -> tuple[str, str | None]:
        node = definition.node
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return signature(node, definition.source), None
        name = f"{definition.module}.{node.name}"
        for parent in self.mro(name):
            inherited = self.definitions.get(parent)
            if inherited is None:
                if parent == "builtins.object":
                    continue
                raise ValueError(f"cannot statically resolve constructor of {name}: {parent}")
            for member in inherited.node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "__init__":
                    return signature(member, inherited.source, method=True), f"{parent}.__init__"
        return "()", None

    def components(self) -> dict[str, list[Component]]:
        registries = self.registries()
        result: dict[str, list[Component]] = {kind: [] for kind in registries.values()}
        seen = set()
        for definition in self.definitions.values():
            for decorator in definition.node.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "register"
                ):
                    continue
                registry = self.resolve(definition.module, ast.unparse(decorator.func.value))
                if registry not in registries:
                    raise ValueError(
                        f"unresolved registry decorator in {definition.path}: {registry}"
                    )
                name = ast.literal_eval(decorator.args[0])
                if not isinstance(name, str):
                    raise ValueError(f"registry entry must have a literal name: {definition.path}")
                target = f"{registries[registry]}:{name}"
                if target in seen:
                    raise ValueError(f"duplicate component: {target}")
                seen.add(target)
                args, constructor = self.constructor(definition)
                result[registries[registry]].append(
                    Component(target, definition, args, constructor)
                )
        return {
            kind: sorted(entries, key=lambda item: item.target) for kind, entries in result.items()
        }


def signature(node, source: str, *, method: bool = False) -> str:
    """Format parameters while preserving source spelling of annotations and defaults."""
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)

    def parameter(arg, default=None):
        value = arg.arg
        if arg.annotation:
            value += ": " + ast.get_source_segment(source, arg.annotation)
        if default is not None:
            value += " = " + ast.get_source_segment(source, default)
        return value

    parts = []
    for i, (arg, default) in enumerate(zip(positional, defaults, strict=True)):
        if not (method and i == 0):
            parts.append(parameter(arg, default))
        if args.posonlyargs and i + 1 == len(args.posonlyargs) and parts:
            parts.append("/")
    if args.vararg:
        parts.append("*" + parameter(args.vararg))
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        parts.append(parameter(arg, default))
    if args.kwarg:
        parts.append("**" + parameter(args.kwarg))
    return "(" + ", ".join(parts) + ")"


def code(value: str) -> str:
    """Escape literal text for an inline code cell in a Markdown table."""
    return "<code>" + html.escape(value).replace("|", "&#124;").replace("\n", " ") + "</code>"


def description(definition: Definition) -> str:
    doc = ast.get_docstring(definition.node)
    return doc.splitlines()[0] if doc else "No description."


def source_link(root: Path, path: Path, line: int | None = None) -> str:
    suffix = f"#L{line}" if line else ""
    return f"{GITHUB}/blob/main/{path.relative_to(root).as_posix()}{suffix}"


def component_page(index: SourceIndex, kind: str, entries: list[Component]) -> str:
    text = (
        f"# {kind.capitalize()} components\n\n"
        f"Use these `{kind}:name` targets in configuration `_target_` fields. "
        "Add the defining module to `_imports_` to register a component explicitly. "
        "Method components and benchmark probes require these imports.\n\n"
    )
    if not entries:
        return text + "No components registered.\n"
    text += "| Target | Description |\n| --- | --- |\n"
    for entry in entries:
        anchor = entry.target.replace(":", "-")
        summary = html.escape(description(entry.definition)).replace("|", "&#124;")
        text += f"| [{code(entry.target)}](#{anchor}) | {summary} |\n"
    for entry in entries:
        definition = entry.definition
        anchor = entry.target.replace(":", "-")
        link = source_link(index.root, definition.path, definition.node.lineno)
        text += (
            f"\n## `{entry.target}` {{#{anchor}}}\n\n"
            f"{description(definition)}\n\n"
            f"Defining module / `_imports_`: [`{definition.module}`]({link}).\n\n"
            f"```python\n{definition.node.name}{entry.signature}\n```\n"
        )
        if entry.constructor and entry.constructor != (
            f"{definition.module}.{definition.node.name}.__init__"
        ):
            text += f"\nInherited constructor: `{entry.constructor}`.\n"
    return text


def config_page(root: Path) -> str:
    text = (
        "# Packaged configs\n\n"
        "These YAML files ship with the package. Use their names with `egl train` "
        "or `egl eval`, or compose them through Hydra defaults.\n\n"
        "| File | Header comment |\n| --- | --- |\n"
    )
    folder = root / "src/egolearn/configs"
    for path in sorted(folder.rglob("*")):
        if path.suffix not in {".yaml", ".yml"}:
            continue
        header = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                header.append(line[1:].removeprefix(" "))
            elif line.strip() or header:
                break
        comment = "\n".join(header) if header else "No header comment."
        name = path.relative_to(folder).as_posix()
        text += f"| [{code(name)}]({source_link(root, path)}) | {code(comment)} |\n"
    return text


def cli_page(index: SourceIndex) -> str:
    tree = index.modules["egolearn.cli"][2]
    metavar = None
    epilog = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "ArgumentParser":
            epilog = next(ast.literal_eval(k.value) for k in node.keywords if k.arg == "epilog")
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "arguments"
        ):
            metavar = next(ast.literal_eval(k.value) for k in node.keywords if k.arg == "metavar")
    if not metavar or not epilog:
        raise ValueError("could not extract train/eval usage from egolearn.cli")
    text = (
        "# Train and eval CLI\n\n"
        f"{epilog}\n\n"
        "| Command | Purpose |\n| --- | --- |\n"
        "| [`egl train`](#egl-train) | Train from a Hydra configuration |\n"
        "| [`egl eval`](#egl-eval) | Evaluate a checkpoint |\n"
    )
    for command in ("train", "eval"):
        text += (
            f"\n## `egl {command}`\n\n```text\negl {command} {metavar}\n```\n\n"
            "Hydra displays the composed configuration and its own command options:\n\n"
            f"```bash\negl {command} --help\negl {command} --hydra-help\n```\n"
        )
    return text + "\nSee the [command-line guide](../cli.md) for configuration examples.\n"


def benchmark_page(index: SourceIndex) -> str:
    text = (
        "# Benchmark data\n\n"
        "Each benchmark package records its data source, license, access tier, "
        "and preparation step. "
        "Public labels do not grant access to restricted images. "
        "See [access tiers](../evaluation/hub.md#access-tiers) before sharing prepared data.\n\n"
        "| Benchmark | Data source | License / terms | Access tier | Preparation |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
    prefix = "egolearn.evaluation.benchmarks."
    for module, (path, _, tree) in index.modules.items():
        if not module.startswith(prefix) or path.name != "__init__.py":
            continue
        values = [
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "BENCHMARK_METADATA" for t in node.targets)
        ]
        if len(values) != 1:
            raise ValueError(f"{path}: expected one literal BENCHMARK_METADATA mapping")
        metadata = values[0]
        keys = {"name", "source", "license", "access", "prepare"}
        if not isinstance(metadata, dict) or set(metadata) != keys:
            raise ValueError(f"{path}: BENCHMARK_METADATA requires {sorted(keys)}")
        if not all(isinstance(v, str) and v for v in metadata.values()):
            raise ValueError(f"{path}: benchmark metadata values must be nonempty strings")
        if metadata["access"] not in {"public", "gated", "IRB", "upstream-licensed"}:
            raise ValueError(f"{path}: invalid benchmark access tier")
        text += (
            f"| [{metadata['name']}]({source_link(index.root, path)}) "
            f"| {metadata['source']} | {metadata['license']} | {metadata['access']} "
            f"| {metadata['prepare']} |\n"
        )
    return text


def reference_pages(root: Path) -> tuple[dict[str, ReferencePage], list[dict]]:
    """Build Markdown and navigation from a fresh static source index."""
    index = SourceIndex(root)
    pages = {}
    components = []
    for kind, entries in index.components().items():
        path = f"reference/components/{kind}.md"
        pages[path] = ReferencePage(
            component_page(index, kind, entries), "src/egolearn/config/registry.py"
        )
        components.append({kind.capitalize(): path})
    api = []
    for group, names in API.items():
        path = f"reference/api/{group.lower()}.md"
        for name in names:
            if name not in index.definitions:
                raise ValueError(f"API entry must name its defining module: {name}")
        content = f"# {group} API\n\n" + "\n\n".join(f"::: {name}" for name in names) + "\n"
        pages[path] = ReferencePage(content, "scripts/docs/reference.py")
        api.append({group: path})
    pages["reference/cli.md"] = ReferencePage(cli_page(index), "src/egolearn/cli.py")
    pages["reference/configs.md"] = ReferencePage(config_page(root), "src/egolearn/configs")
    pages["reference/benchmarks.md"] = ReferencePage(
        benchmark_page(index), "src/egolearn/evaluation/benchmarks"
    )
    nav = [
        {"Components": components},
        {"Python API": api},
        {"Train and eval CLI": "reference/cli.md"},
        {"Packaged configs": "reference/configs.md"},
        {"Benchmark data": "reference/benchmarks.md"},
    ]
    return pages, nav


_sources: dict[str, str] = {}


def on_files(files, config):
    from mkdocs.structure.files import File

    pages, navigation = reference_pages(ROOT)
    _sources.clear()
    for path, page in pages.items():
        if files.get_file_from_path(path):
            raise ValueError(f"generated reference page must not be committed: {path}")
        files.append(File.generated(config, path, content=page.content))
        _sources[path] = page.source
    for section in config.nav:
        if "Reference" in section:
            section["Reference"] = navigation
            break
    else:
        raise ValueError("mkdocs.yml needs a Reference navigation section")
    return files


def on_page_markdown(markdown, page, **kwargs):
    if source := _sources.get(page.file.src_uri):
        # Directories have no edit form. Every catalog row links to its source file.
        page.edit_url = f"{GITHUB}/edit/main/{source}" if Path(source).suffix else None
    return markdown
