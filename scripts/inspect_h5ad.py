"""Inspect top-level keys and their contents in an h5ad file."""
import sys
import h5py


def _infer_x_state(sample):
    """Infer preprocessing state of X from a sample of values."""
    import numpy as np

    nonzero = sample[sample != 0]
    if len(nonzero) == 0:
        return "empty (all zeros)"

    has_negatives = (nonzero < 0).any()
    is_integer = np.allclose(nonzero, np.round(nonzero))
    max_val = np.abs(nonzero).max()
    mean_val = nonzero.mean()
    std_val = nonzero.std()

    if has_negatives:
        if 0.5 < std_val < 2.0 and abs(mean_val) < 1.0:
            return "z-scored / scaled (negative values, std ~1)"
        return "z-scored / scaled (negative values present)"
    elif is_integer and max_val > 20:
        return "raw counts (integers)"
    elif is_integer and max_val <= 20:
        return "likely raw counts (small integers)"
    elif max_val < 15:
        return "log-normalized (floats, range 0-15)"
    else:
        return "normalized (floats, not log-transformed)"


def inspect(path: str):
    import numpy as np

    with h5py.File(path, "r") as f:
        for key in f.keys():
            if hasattr(f[key], "keys"):
                print(f"{key}: {list(f[key].keys())}")
            else:
                print(f"{key}: {f[key].shape} {f[key].dtype}")

        # Sample X values to infer preprocessing state
        if "X" in f:
            x = f["X"]
            if hasattr(x, "keys") and "data" in x:
                sample = np.array(x["data"][:1000])
            else:
                sample = np.array(x[:5, :5]).flatten()

            print(f"\nX sample (first nonzero values): {sample[sample != 0][:10]}")
            print(f"X state: {_infer_x_state(sample)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <file.h5ad>", file=sys.stderr)
        sys.exit(1)
    inspect(sys.argv[1])
