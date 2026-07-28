"""Compile protos/med_session.proto into Python modules.

Equivalent of ``make proto`` but portable (Windows/Linux/macOS).
Generated files are written to ``src/med_langchain_memory/serde/``
and are committed to the repository, so end users never need protoc.

Usage:
    python scripts/gen_proto.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = ROOT / "protos"
PROTO_FILE = PROTO_DIR / "med_session.proto"
OUT_DIR = ROOT / "src" / "med_langchain_memory" / "serde"


def main() -> int:
    """Run protoc via grpcio-tools and report generated files."""
    try:
        from grpc_tools import protoc
    except ImportError:
        print("error: grpcio-tools is required: pip install grpcio-tools", file=sys.stderr)
        return 1

    if not PROTO_FILE.is_file():
        print(f"error: proto file not found: {PROTO_FILE}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = [
        "protoc",
        f"--proto_path={PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--pyi_out={OUT_DIR}",
        str(PROTO_FILE),
    ]
    code = int(protoc.main(args))
    if code != 0:
        print(f"error: protoc exited with code {code}", file=sys.stderr)
        return code

    for name in ("med_session_pb2.py", "med_session_pb2.pyi"):
        print(f"generated: {(OUT_DIR / name).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
