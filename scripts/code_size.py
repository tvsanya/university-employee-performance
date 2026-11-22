"""Transparent source-only statement count; data and weights are excluded."""
import argparse
import ast
import json
from pathlib import Path


def count_file(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    doc_nodes = set()
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str):
                doc_nodes.add(id(first))
                doc_lines.update(range(first.lineno,first.end_lineno+1))
    logical = sum(isinstance(node,ast.stmt) and id(node) not in doc_nodes for node in ast.walk(tree))
    physical = sum(bool(line.strip()) and not line.lstrip().startswith("#") and number not in doc_lines for number,line in enumerate(source.splitlines(),1))
    return {"file":path.name,"logical_statements":logical,"nonempty_code_lines":physical}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    files = [root/"main.py",*sorted((root/"university_analytics").glob("*.py"))]
    counts = [count_file(path) for path in files]
    report = {"method":"AST statements excluding docstrings; definitions and control statements included", "excluded":["tests","scripts","notebooks","HTML","JavaScript","data","serialized weights","comments","blank lines"], "files":counts,
        "logical_statements":sum(item["logical_statements"] for item in counts),"nonempty_code_lines":sum(item["nonempty_code_lines"] for item in counts)}
    serialized = json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(serialized,encoding="utf-8")
    print(serialized,end="")


if __name__ == "__main__":
    main()
