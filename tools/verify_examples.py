#!/usr/bin/env python
"""Verify docs pages by executing their ```python blocks — one subprocess per page.

For each docs/**/*.md page: extract fenced python blocks (dedented, so
tabbed content works), concatenate, transform bare trailing expressions
into notebook-style displays, and exec in a FRESH subprocess with the Agg
backend. Figures are saved to build/verify_figs/. Captured stdout is
written to build/verify_stdout.txt for cross-checking expected-output
blocks in the docs.

Usage: python tools/verify_examples.py [page_rel_path ...]
Exit 1 on any failure.
"""
import os
import re
import subprocess
import sys
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DOCS = os.path.join(ROOT, "docs")

PAGE_RE = re.compile(r"^(?: {0,8})```python\n(.*?)^(?: {0,8})```", re.DOTALL | re.MULTILINE)

RUNNER = r'''
import ast, io, os, sys, traceback
from contextlib import redirect_stdout
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

page = sys.argv[1]
text = open(page).read()
import re, textwrap
blocks = [textwrap.dedent(b) for b in re.findall(
    r"^(?: {0,8})```python\n(.*?)^(?: {0,8})```", text, re.DOTALL | re.MULTILINE)]
code = "\n\n".join(blocks)

def display(v):
    print(repr(v))

tree = ast.parse(code)
new = []
for node in tree.body:
    if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
        new.append(ast.Expr(value=ast.Call(
            func=ast.Name(id="__display", ctx=ast.Load()),
            args=[node.value], keywords=[])))
    else:
        new.append(node)
tree.body = new
ast.fix_missing_locations(tree)

buf = io.StringIO()
try:
    with redirect_stdout(buf):
        exec(compile(tree, page, "exec"),
             {"__name__": "__main__", "__file__": page, "__display": display})
    sys.stdout.write(buf.getvalue())
except Exception:
    sys.stdout.write(buf.getvalue())
    traceback.print_exc()
    sys.exit(1)
finally:
    figdir = sys.argv[2]
    for i, num in enumerate(plt.get_fignums()):
        plt.figure(num).savefig(os.path.join(
            figdir, os.path.basename(page)[:-3] + f"_{i:02d}.png"),
            dpi=160, bbox_inches="tight")
'''


def main():
    targets = sys.argv[1:]
    pages = []
    for root, _dirs, files in os.walk(DOCS):
        for f in sorted(files):
            if f.endswith(".md"):
                pages.append(os.path.join(root, f))
    pages.sort()
    if targets:
        pages = [p for p in pages if any(t in p for t in targets)]

    figdir = os.path.join(ROOT, "build", "verify_figs")
    os.makedirs(figdir, exist_ok=True)
    runner = os.path.join(ROOT, "build", "_page_runner.py")
    with open(runner, "w") as fh:
        fh.write(RUNNER)

    total_blocks = 0
    failures = []
    records = []
    for path in pages:
        rel = os.path.relpath(path, ROOT)
        blocks = PAGE_RE.findall(open(path).read())
        if not blocks:
            print(f"[skip] {rel}: no python blocks")
            continue
        total_blocks += len(blocks)
        env = dict(os.environ, MPLBACKEND="Agg")
        # The runner lives under build/. Test this checkout, even if an older
        # distribution is installed in the shared Python library.
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        r = subprocess.run([sys.executable, runner, path, figdir],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        ok = r.returncode == 0
        print(f"[{'ok  ' if ok else 'FAIL'}] {rel}: {len(blocks)} block(s)")
        out = r.stdout.strip()
        if out:
            records.append((rel, out))
            for line in out.splitlines()[:40]:
                print(f"  | {line}")
            if len(out.splitlines()) > 40:
                print(f"  | ... ({len(out.splitlines()) - 40} more lines)")
        if not ok:
            failures.append(rel)
            for line in r.stderr.strip().splitlines()[-8:]:
                print(f"  ! {line}")

    print()
    print("=" * 60)
    print(f"pages verified: {sum(1 for p in pages if PAGE_RE.search(open(p).read()))}")
    print(f"total python blocks executed: {total_blocks}")
    print(f"failures: {len(failures)} {failures}")

    with open(os.path.join(ROOT, "build", "verify_stdout.txt"), "w") as fh:
        for rel, out in records:
            fh.write(f"##### {rel}\n{out}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
