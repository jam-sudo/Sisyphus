"""Keep the ODE engine independent of higher layers."""

import ast
from pathlib import Path


def test_engine_imports_only_lower_layers() -> None:
    engine_dir = Path(__file__).resolve().parents[2] / "src" / "sisyphus" / "engine"
    forbidden = {"pk", "regimen", "physiology", "predict", "ml", "pipeline", "mipd", "sbi"}
    violations = []
    for path in engine_dir.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                if node.level >= 2 and (node.module or "").split(".")[0] in forbidden:
                    violations.append(f"{path.name}:{node.lineno} imports relative {node.module}")
            else:
                continue
            for module in modules:
                parts = module.split(".")
                if len(parts) > 1 and parts[0] == "sisyphus" and parts[1] in forbidden:
                    violations.append(f"{path.name}:{node.lineno} imports {module}")
    assert not violations, "\n".join(violations)
