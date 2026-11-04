"""
This script converts rosbags-convert output (ROS 2 Iron/Jazzy format)
into a ROS 2 Humble-compatible metadata.yaml.

Changes applied:
  1. version: <N> -> version: 5
  2. remove the 'ros_distro' key
  3. remove 'type_description_hash' from every topic
  4. replace every 'offered_qos_profiles' value with "" (empty string)

Usage:
  python fix_metadata.py path/to/metadata.yaml
  python fix_metadata.py path/to/metadata.yaml --dry-run
  python fix_metadata.py path/to/metadata.yaml --backup

Exit codes:
  0 - success (or nothing to do)
  1 - path is not a file
  2 - YAML parse error
  3 - structural error (missing 'rosbag2_bagfile_information')
"""


import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write("PyYAML is required: pip install pyyaml\n")
    sys.exit(1)


def fix_metadata(data: dict) -> list[str]:
    """Mutate `data` in place and return a list of human-readable changes."""
    if not isinstance(data, dict) or "rosbag2_bagfile_information" not in data:
        raise ValueError("Missing top-level key 'rosbag2_bagfile_information'")

    root = data["rosbag2_bagfile_information"]
    changes: list[str] = []

    # 1) version -> 5
    if root.get("version") != 5:
        changes.append(f"version: {root.get('version')!r} -> 5")
        root["version"] = 5

    # 2) drop ros_distro
    if "ros_distro" in root:
        changes.append(f"removed ros_distro: {root['ros_distro']!r}")
        del root["ros_distro"]

    # 3–4) per-topic fixes
    for entry in root.get("topics_with_message_count") or []:
        meta = entry.get("topic_metadata") or {}
        name = meta.get("name", "<unknown>")

        if "type_description_hash" in meta:
            changes.append(f"[{name}] removed type_description_hash")
            del meta["type_description_hash"]

        if "offered_qos_profiles" in meta:
            qos = meta["offered_qos_profiles"]
            if qos != "" and qos is not None:
                changes.append(f"[{name}] offered_qos_profiles -> ''")
            meta["offered_qos_profiles"] = ""

    return changes


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fix metadata.yaml for ROS 2 Humble compatibility."
    )
    ap.add_argument("path", type=Path, help="Path to metadata.yaml")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes but do not modify the file",
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        help="Create metadata.yaml.bak before overwriting",
    )
    args = ap.parse_args()

    if not args.path.is_file():
        print(f"ERROR: {args.path} is not a file", file=sys.stderr)
        return 1

    try:
        with args.path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR parsing {args.path}: {e}", file=sys.stderr)
        return 2

    try:
        changes = fix_metadata(data)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    if not changes:
        print("No changes needed — file is already Humble-compatible.")
        return 0

    print(f"Changes ({len(changes)}):")
    for c in changes:
        print(f"  - {c}")

    if args.dry_run:
        print("\n[dry-run] File NOT modified.")
        return 0

    if args.backup:
        bak = args.path.with_suffix(args.path.suffix + ".bak")
        bak.write_bytes(args.path.read_bytes())
        print(f"\nBackup: {bak}")

    with args.path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,          # preserve original key order (pyyaml >= 5.1)
            default_flow_style=False, # block style, same as original
            allow_unicode=True,
        )

    print(f"\nOK, written: {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())