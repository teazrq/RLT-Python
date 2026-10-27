#!/usr/bin/env python
"""Cross-check expected-output blocks in docs/ against actual verified stdout.

For each ```python block whose code ends in print(...) or bare display
expressions, the following plain ``` block (if any) is treated as expected
output. Re-executes the page (fresh subprocess, like verify_examples) and
compares normalized text. Reports mismatches.
"""
import os
import re
import subprocess
import sys
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DOCS = os.path.join(ROOT, "docs")

PY_RE = re.compile(r"^( {0,8})```python\n(.*?)^\1```", re.DOTALL | re.MULTILINE)


def expected_blocks(text):
    """Map python-block-index to expected output in plain or text fences."""
    out = {}
    pos = 0
    idx = 0
    lines = text.splitlines(keepends=True)
    # find all fences in order
    fence_re = re.compile(r"^( {0,8})```(\w*)\n(.*?)^\1```", re.DOTALL | re.MULTILINE)
    last_was_python = False
    py_i = 0
    for m in fence_re.finditer(text):
        # A section heading starts a different object, such as an API signature.
        if re.search(r"^#{1,6} ", text[pos:m.start()], re.MULTILINE):
            last_was_python = False
        pos = m.end()
        lang = m.group(2)
        if lang == "python":
            last_was_python = True
            py_i = idx
            idx += 1
        else:
            if last_was_python and lang in ("", "text"):
                out[py_i] = m.group(3)
            last_was_python = False
    return out


def main():
    pages = []
    for root, _d, files in os.walk(DOCS):
        for f in sorted(files):
            if f.endswith(".md"):
                pages.append(os.path.join(root, f))
    pages.sort()

    # regenerate the runner (same one verify_examples writes)
    runner = os.path.join(ROOT, "build", "_page_runner.py")
    os.makedirs(os.path.dirname(runner), exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from verify_examples import RUNNER
    with open(runner, "w") as fh:
        fh.write(RUNNER)

    n_checked = 0
    n_mismatch = 0
    for path in pages:
        rel = os.path.relpath(path, ROOT)
        text = open(path).read()
        exp = expected_blocks(text)
        if not exp:
            continue
        env = dict(os.environ, MPLBACKEND="Agg")
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        figdir = os.path.join(ROOT, "build", "verify_figs")
        os.makedirs(figdir, exist_ok=True)
        r = subprocess.run([sys.executable, runner, path, figdir],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        if r.returncode != 0:
            print(f"[FAIL] {rel}: page failed to execute")
            print(r.stderr.strip())
            n_mismatch += 1
            continue
        # actual stdout, normalized: drop matplotlib repr lines and None
        actual_lines = [l for l in r.stdout.splitlines()
                        if l.strip() and l.strip() != "None"
                        and "matplotlib" not in l]
        actual = "\n".join(actual_lines)
        for py_i, expected in exp.items():
            exp_norm = "\n".join(l.rstrip() for l in expected.splitlines()
                                 if l.strip())
            act_norm = "\n".join(l.rstrip() for l in actual.splitlines()
                                 if l.strip())
            if exp_norm in act_norm:
                n_checked += 1
            else:
                n_checked += 1
                n_mismatch += 1
                print(f"[MISMATCH] {rel} python-block #{py_i + 1}")
                print(f"  expected: {exp_norm[:120]!r}")
                # find best guess location in actual
                first = exp_norm.splitlines()[0]
                hits = [l for l in actual_lines if first[:30] in l]
                if hits:
                    i = actual_lines.index(hits[0])
                    ctx = "\n".join(actual_lines[i:i + len(exp_norm.splitlines())])
                    print(f"  actual:   {ctx[:240]!r}")
                else:
                    print(f"  (no line matching {first[:40]!r} in output)")

    print(f"\nexpected-output blocks checked: {n_checked}, mismatches: {n_mismatch}")
    return 1 if n_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
