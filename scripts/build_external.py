"""Build a test set of poisoned and benign tool descriptions from public sources other than MCPTox.

Nothing here is used to write rules or to train. It measures whether detection carries over
to attacks written by other people. The data is not redistributed here; get the sources first:

    git clone https://github.com/xjzzzzzzzz/MCPSafety   # MCP-SafetyBench (Zong et al., ICLR 2026), BSD-3-Clause
    git clone https://github.com/dongsenzhang/MSB       # MCP Security Bench (Zhang et al., ICLR 2026), MIT
    curl -LO https://huggingface.co/datasets/fevziegeyurtsevenler/mcp-tool-poisoning/resolve/main/mtp.jsonl  # Apache-2.0

Then:

    python scripts/build_external.py --safetybench MCPSafety --msb MSB --mtp mtp.jsonl -o data/external.jsonl

Each output row has source, category, server, tool, label ("poisoned" or "benign"), instructs and
text. `instructs` is false for attacks that give the model no instruction: advertising a tool as
the best, copying another tool's description, or adding a parameter that asks for the model's
name. A text detector is not expected to catch those.

Python files from the sources are parsed with `ast`, never imported or run.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

NO_INSTRUCTION = {"Preference Manipulation", "Function Overlapping", "preference_manipulation", "out_of_scope_parameter"}


def tool_docstrings(path: Path) -> dict[str, str]:
    """Docstrings of the functions decorated as MCP tools in a Python file."""
    out = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and any("tool" in ast.unparse(d) for d in node.decorator_list):
            doc = ast.get_docstring(node)
            if doc:
                out[node.name] = doc
    return out


def string_constant(path: Path, name: str) -> str:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def row(source: str, category: str, server: str, tool: str, label: str, text: str) -> dict:
    instructs = label == "poisoned" and category not in NO_INSTRUCTION
    return {"source": source, "category": category, "server": server, "tool": tool, "label": label, "instructs": instructs, "text": text}


def safetybench(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("mcpuniverse/benchmark/configs/**/*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(task, dict) or "attack_category" not in task:
            continue
        category, domain = task["attack_category"], path.parent.name
        for mod in task.get("mcp_server_modifications") or []:
            # Modifications that only change what a tool returns leave its description alone.
            if "modification_description" in mod:
                rows.append(row("MCP-SafetyBench", category, domain, mod["tool_name"], "poisoned", mod["modification_description"]))
        added = task.get("mcp_server_additions")
        if added:
            rows.append(row("MCP-SafetyBench", category, domain, added["tool_name"], "poisoned", added["description"]))
    for path in sorted(root.glob("mcpuniverse/mcp/servers/*/*.py")):
        for tool, doc in tool_docstrings(path).items():
            rows.append(row("MCP-SafetyBench", "benign", path.parent.name, tool, "benign", doc))
    return rows


def msb(root: Path) -> list[dict]:
    template = string_constant(root / "data" / "prompt_template.py", "PROMPT_INJECTION_TEMPLATE")
    goals = [json.loads(line)["implementation"] for line in (root / "data" / "attack_task.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for server in sorted(p for p in (root / "data" / "tools" / "attack_tools").glob("*/*") if p.is_dir()):
        for path in sorted(server.glob("*.py")):
            stem, _, variant = path.stem.partition("-")
            # Name collision and tool transfer keep the original text; the combined variant repeats the two below.
            if variant in ("preference_manipulation", "out_of_scope_parameter"):
                base = tool_docstrings(path.with_name(stem + ".py"))
                for tool, doc in tool_docstrings(path).items():
                    if doc != base.get(tool):
                        rows.append(row("MSB", variant, server.name, tool, "poisoned", doc))
            elif not variant:
                for tool, doc in tool_docstrings(path).items():
                    rows.append(row("MSB", "benign", server.name, tool, "benign", doc))
                    # MSB appends one injection template, filled with one of its attack goals, to the tool list.
                    for goal in goals:
                        text = doc.rstrip() + "\n" + template.replace("{goal}", goal.replace("{pid}", "4242"))
                        rows.append(row("MSB", "prompt_injection", server.name, tool, "poisoned", text))
    return rows


def mtp(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.append(row("mcp-tool-poisoning", r["technique"], "", r["tool_name"], "poisoned", r["poisoned_description"]))
            rows.append(row("mcp-tool-poisoning", "benign", "", r["tool_name"], "benign", r["benign_description"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safetybench", type=Path, help="Clone of github.com/xjzzzzzzzz/MCPSafety")
    ap.add_argument("--msb", type=Path, help="Clone of github.com/dongsenzhang/MSB")
    ap.add_argument("--mtp", type=Path, help="mtp.jsonl from fevziegeyurtsevenler/mcp-tool-poisoning")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for fn, src in ((safetybench, args.safetybench), (msb, args.msb), (mtp, args.mtp)):
        if src:
            rows += fn(src)
    # A text listed twice in one source counts once.
    seen, unique = set(), []
    for r in rows:
        key = (r["source"], r["label"], r["text"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in unique), encoding="utf-8")
    print(f"wrote {len(unique)} rows to {args.out}")


if __name__ == "__main__":
    main()
