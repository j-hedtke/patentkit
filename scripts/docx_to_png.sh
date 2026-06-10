#!/usr/bin/env bash
# Render the first page of a .docx to PNG: Word (AppleScript) -> PDF -> pymupdf.
# Usage: scripts/docx_to_png.sh in.docx out.png [dpi]
set -euo pipefail
IN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
OUT=$2
DPI=${3:-180}
PDF="${IN%.docx}.pdf"

osascript <<EOF
tell application "Microsoft Word"
  set wasRunning to running
  open POSIX file "$IN"
  set doc to active document
  save as doc file name (POSIX file "$PDF") file format format PDF
  close doc saving no
  if not wasRunning then quit
end tell
EOF

.venv/bin/python - "$PDF" "$OUT" "$DPI" <<'PY'
import sys
import fitz

pdf, out, dpi = sys.argv[1], sys.argv[2], int(sys.argv[3])
doc = fitz.open(pdf)
doc[0].get_pixmap(dpi=dpi).save(out)
print(f"{out}: page 1 of {len(doc)} at {dpi} dpi")
PY
