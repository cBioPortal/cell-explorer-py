#!/usr/bin/env python3
"""
Script to find all h5ad files in a directory and create symlinks to them.
"""
import argparse
from pathlib import Path
import sys


def find_h5ad_files(search_dir: Path) -> list[Path]:
    """Recursively find all .h5ad files in the given directory."""
    return list(search_dir.rglob("*.h5ad"))


def create_symlinks(h5ad_files: list[Path], output_dir: Path) -> None:
    """Create symlinks to h5ad files in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    created_count = 0
    skipped_count = 0

    for h5ad_file in h5ad_files:
        # Create symlink with the same filename in output directory
        symlink_path = output_dir / h5ad_file.name

        if symlink_path.exists():
            print(f"⚠️  Skipping {h5ad_file.name} (already exists)")
            skipped_count += 1
            continue

        try:
            symlink_path.symlink_to(h5ad_file.resolve())
            print(f"✓ Created symlink: {symlink_path} -> {h5ad_file}")
            created_count += 1
        except Exception as e:
            print(f"✗ Error creating symlink for {h5ad_file.name}: {e}", file=sys.stderr)

    print(f"\nSummary: Created {created_count} symlink(s), skipped {skipped_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Find h5ad files and create symlinks to them"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory to search for h5ad files"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where symlinks will be created"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without creating symlinks"
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    if not args.input_dir.is_dir():
        print(f"Error: '{args.input_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find h5ad files
    print(f"Searching for .h5ad files in {args.input_dir}...")
    h5ad_files = find_h5ad_files(args.input_dir)

    if not h5ad_files:
        print("No .h5ad files found")
        sys.exit(0)

    print(f"Found {len(h5ad_files)} .h5ad file(s):")
    for f in h5ad_files:
        print(f"  - {f}")
    print()

    if args.dry_run:
        print(f"[DRY RUN] Would create symlinks in: {args.output_dir}")
        sys.exit(0)

    # Create symlinks
    create_symlinks(h5ad_files, args.output_dir)


if __name__ == "__main__":
    main()
