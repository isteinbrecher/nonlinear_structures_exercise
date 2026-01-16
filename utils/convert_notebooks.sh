#!/usr/bin/env bash
set -euo pipefail

# 0. Remove all existing _student and _teacher notebooks (recursively)
find ../ -name "*_student.ipynb" -print0 | xargs -0 rm -f
find ../ -name "*_teacher.ipynb" -print0 | xargs -0 rm -f

# 1. Strip output from all notebooks (recursively)
find ../ -name "*.ipynb" -print0 | xargs -0 nbstripout

# 2. Loop over cleaned notebooks and generate _student versions
find ../ -name "*.ipynb" -print0 |
while IFS= read -r -d '' nb; do
  python utils/make_student.py "$nb"
  python utils/make_student.py --teacher "$nb"
done
