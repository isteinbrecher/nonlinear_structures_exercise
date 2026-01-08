"""Create "student" versions from master files by stripping SOLUTION blocks and
keeping STUDENT blocks.

Supports:
- .ipynb (via nbformat)
- .py / .md / .markdown / .txt (plain text processing)

Markers:

Python/Jupytext (supports trailing extras like "# fmt: skip"):
    # === BLOCK SOLUTION ===
    # === BLOCK STUDENT === # fmt: skip
    # === BLOCK END ===

Markdown:
    <!-- === BLOCK SOLUTION === -->
    <!-- === BLOCK STUDENT === -->
    <!-- === BLOCK END === -->

Behavior:
- Inside a SOLUTION block: removed
- Inside a STUDENT block: kept
- Marker lines themselves: removed
- Anything outside blocks: unchanged
- If a SOLUTION block ends without any STUDENT block appearing before END:
    replace that SOLUTION content with a placeholder (no extra blank line)

Notes on newlines:
- Placeholders are stored WITHOUT trailing newline.
- When inserting a placeholder we insert exactly ONE newline after it.
- If you provide a custom placeholder via CLI, it is normalized the same way (no trailing newlines).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

# =========================
# Defaults
# =========================
DEFAULT_SUFFIX = "_student"

DEFAULT_CODE_PLACEHOLDER = "# TODO: Implement solution"
DEFAULT_MD_PLACEHOLDER = "*TODO: Write your answer here.*"

# =========================
# Marker regexes
# =========================
# For Python/Jupytext markers, allow trailing stuff (e.g. "# fmt: skip") after the final ===
# Example: "# === BLOCK STUDENT === # fmt: skip"
PY_SOLUTION_RE = re.compile(r"^\s*#\s*===\s*BLOCK\s+SOLUTION\s*===\s*(?:#.*)?$")
PY_STUDENT_RE = re.compile(r"^\s*#\s*===\s*BLOCK\s+STUDENT\s*===\s*(?:#.*)?$")
PY_END_RE = re.compile(r"^\s*#\s*===\s*BLOCK\s+END\s*===\s*(?:#.*)?$")

MD_SOLUTION_RE = re.compile(r"^\s*<!--\s*===\s*BLOCK\s+SOLUTION\s*===\s*-->\s*$")
MD_STUDENT_RE = re.compile(r"^\s*<!--\s*===\s*BLOCK\s+STUDENT\s*===\s*-->\s*$")
MD_END_RE = re.compile(r"^\s*<!--\s*===\s*BLOCK\s+END\s*===\s*-->\s*$")


def _normalize_placeholder(s: str) -> str:
    """Normalize placeholder so it never ends with trailing newlines.

    We will control line breaks at insertion time.
    """
    return s.rstrip("\r\n")


def _extract_indent(sample_line: str) -> str:
    """TMP."""
    m = re.match(r"\s*", sample_line)
    return m.group(0) if m else ""


def strip_blocks(
    text: str,
    mode: str,
    code_placeholder: str,
    md_placeholder: str,
    preserve_indent: bool = True,
) -> str:
    """Removes SOLUTION blocks; keeps STUDENT blocks. If SOLUTION exists but
    STUDENT is missing before END, insert a placeholder.

    mode: "py" or "md"
    """
    if mode == "py":
        sol_re, stu_re, end_re = PY_SOLUTION_RE, PY_STUDENT_RE, PY_END_RE
        placeholder = _normalize_placeholder(code_placeholder)
    elif mode == "md":
        sol_re, stu_re, end_re = MD_SOLUTION_RE, MD_STUDENT_RE, MD_END_RE
        placeholder = _normalize_placeholder(md_placeholder)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    lines = text.splitlines(keepends=True)
    out: List[str] = []

    state = "OUT"  # OUT, IN_SOLUTION, IN_STUDENT
    saw_student_anywhere = False

    # We'll buffer the first line (and all lines) of the solution region so we can infer indentation
    solution_buffer: List[str] = []

    for line in lines:
        if state == "OUT":
            if sol_re.match(line):
                state = "IN_SOLUTION"
                solution_buffer = []
                continue  # drop marker
            if stu_re.match(line):
                state = "IN_STUDENT"
                saw_student_anywhere = True
                continue  # drop marker
            if end_re.match(line):
                continue  # stray END marker, drop
            out.append(line)

        elif state == "IN_SOLUTION":
            if stu_re.match(line):
                state = "IN_STUDENT"
                saw_student_anywhere = True
                continue  # drop marker
            if end_re.match(line):
                # SOLUTION ended without STUDENT (at least for this block)
                if not saw_student_anywhere:
                    # Insert placeholder with exactly one newline
                    indent = (
                        _extract_indent(solution_buffer[0])
                        if (preserve_indent and solution_buffer)
                        else ""
                    )
                    out.append(f"{indent}{placeholder}\n")
                # If STUDENT exists somewhere else in file, we still remove SOLUTION silently.
                state = "OUT"
                continue  # drop marker

            # collect for indentation inference; content is removed
            solution_buffer.append(line)

        elif state == "IN_STUDENT":
            if end_re.match(line):
                state = "OUT"
                continue  # drop marker
            if sol_re.match(line):
                # handle weird nesting: start solution inside student
                state = "IN_SOLUTION"
                solution_buffer = []
                continue  # drop marker
            if stu_re.match(line):
                # ignore nested student marker
                saw_student_anywhere = True
                continue
            out.append(line)

    return "".join(out)


def process_text_file(
    src: Path,
    dst: Path,
    code_placeholder: str,
    md_placeholder: str,
) -> None:
    """TMP."""
    raw = src.read_text(encoding="utf-8")
    ext = src.suffix.lower()

    if ext == ".py":
        cleaned = strip_blocks(
            raw,
            mode="py",
            code_placeholder=code_placeholder,
            md_placeholder=md_placeholder,
        )
    elif ext in {".md", ".markdown"}:
        cleaned = strip_blocks(
            raw,
            mode="md",
            code_placeholder=code_placeholder,
            md_placeholder=md_placeholder,
        )
    else:
        # default to python markers for unknown text types
        cleaned = strip_blocks(
            raw,
            mode="py",
            code_placeholder=code_placeholder,
            md_placeholder=md_placeholder,
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(cleaned, encoding="utf-8")


def process_ipynb(
    src: Path,
    dst: Path,
    code_placeholder: str,
    md_placeholder: str,
    drop_empty_cells: bool = True,
) -> None:
    """TMP."""
    try:
        import nbformat  # type: ignore
    except ImportError:
        raise SystemExit(
            "Missing dependency: nbformat. Install with: pip install nbformat"
        )

    nb = nbformat.read(str(src), as_version=4)
    new_cells = []

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = strip_blocks(
                cell.source,
                mode="py",
                code_placeholder=code_placeholder,
                md_placeholder=md_placeholder,
            )
        elif cell.cell_type == "markdown":
            cell.source = strip_blocks(
                cell.source,
                mode="md",
                code_placeholder=code_placeholder,
                md_placeholder=md_placeholder,
            )

        if (
            drop_empty_cells
            and isinstance(cell.source, str)
            and cell.source.strip() == ""
        ):
            continue

        new_cells.append(cell)

    nb.cells = new_cells
    dst.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(dst))


def build_dst_path(src: Path, out: Optional[Path]) -> Path:
    """If out is a directory, place file there.

    If out is a file path, use it. If out is None, create alongside src
    with DEFAULT_SUFFIX inserted before extension.
    """
    if out is None:
        return src.with_name(src.stem + DEFAULT_SUFFIX + src.suffix)

    if out.exists() and out.is_dir():
        return out / src.name

    # If out doesn't exist and has no suffix, treat as directory.
    if out.suffix == "":
        out.mkdir(parents=True, exist_ok=True)
        return out / src.name

    return out


def iter_inputs(paths: List[str]) -> Iterable[Path]:
    """TMP."""
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in ("*.ipynb", "*.py", "*.md", "*.markdown", "*.txt"):
                yield from path.rglob(ext)
        else:
            yield path


def main() -> None:
    """TMP."""
    ap = argparse.ArgumentParser(
        description="Strip solution blocks and create student versions."
    )
    ap.add_argument("inputs", nargs="+", help="Input file(s) or directories.")
    ap.add_argument(
        "-o", "--out", default=None, help="Output file or directory (optional)."
    )
    ap.add_argument(
        "--keep-empty-cells",
        action="store_true",
        help="Do not drop cells that become empty.",
    )
    ap.add_argument(
        "--code-placeholder",
        default=DEFAULT_CODE_PLACEHOLDER,
        help="Placeholder inserted when STUDENT block is missing (code).",
    )
    ap.add_argument(
        "--md-placeholder",
        default=DEFAULT_MD_PLACEHOLDER,
        help="Placeholder inserted when STUDENT block is missing (markdown).",
    )
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else None

    # Normalize placeholders (avoid accidental extra blank lines)
    code_placeholder = _normalize_placeholder(args.code_placeholder)
    md_placeholder = _normalize_placeholder(args.md_placeholder)

    for src in iter_inputs(args.inputs):
        if not src.exists():
            print(f"[skip] not found: {src}", file=sys.stderr)
            continue

        dst = build_dst_path(src, out_path)

        try:
            if src.suffix.lower() == ".ipynb":
                process_ipynb(
                    src,
                    dst,
                    code_placeholder=code_placeholder,
                    md_placeholder=md_placeholder,
                    drop_empty_cells=not args.keep_empty_cells,
                )
            else:
                process_text_file(
                    src,
                    dst,
                    code_placeholder=code_placeholder,
                    md_placeholder=md_placeholder,
                )
            print(f"[ok] {src} -> {dst}")
        except Exception as e:
            print(f"[error] {src}: {e}", file=sys.stderr)


if __name__ == "__main__":
    """TMP."""
    main()
