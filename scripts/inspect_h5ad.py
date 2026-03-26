"""Inspect top-level keys and their contents in an h5ad file."""
import sys
import h5py


def inspect(path: str):
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if hasattr(f[key], "keys"):
                print(f"{key}: {list(f[key].keys())}")
            else:
                print(f"{key}: {f[key].shape} {f[key].dtype}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <file.h5ad>", file=sys.stderr)
        sys.exit(1)
    inspect(sys.argv[1])
