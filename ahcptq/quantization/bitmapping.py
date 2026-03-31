# map_bits_from_errors.py
import json
from pathlib import Path
from typing import List, Tuple

# ---------- I/O ----------
def read_errors_jsonl(path: str) -> List[float]:
    """
    Accepts any of:
      - each line is a number (e.g., 0.0123)
      - each line is an object with an error under one of: error,mse,loss,quant_error,err
      - a single line that is a list of numbers [e1, e2, ...]  (entire file is one list)
    Returns a flat list of errors in layer order.
    """
    p = Path(path)
    errors: List[float] = []
    lines = p.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, (int, float)):
            errors.append(float(obj))
        elif isinstance(obj, list):
            if not all(isinstance(x, (int, float)) for x in obj):
                raise ValueError(f"Line {i+1}: list contains non-numerics.")
            if errors:
                raise ValueError("Mixed formats (list + per-line). Put all errors in one form.")
            return [float(x) for x in obj]
        elif isinstance(obj, dict):
            for k in ("error", "mse", "loss", "quant_error", "err"):
                if k in obj and isinstance(obj[k], (int, float)):
                    errors.append(float(obj[k]))
                    break
            else:
                raise ValueError(f"Line {i+1}: dict lacks a numeric error field.")
        else:
            raise ValueError(f"Line {i+1}: unsupported JSON type {type(obj)}.")
    return errors

def write_mapping_jsonl(out_path: str, errors: List[float], bits: List[int],ranks: List[int]) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for idx, (e, b) in enumerate(zip(errors, bits)):
            json.dump({"index": idx, "error": e, "bit": int(b),"rank":ranks[idx]}, f)
            f.write("\n")

# ---------- Core ----------
def assign_bits_by_error(errors: List[float],
                         target_avg: float = 4.3,
                         low_bit: int = 4,
                         high_bit: int = 5,
                         mode: str = "closest") -> Tuple[List[int], float, int]:
    """
    Assign 'high_bit' to the largest errors and 'low_bit' to the rest to hit an average ≈ target_avg.
    mode:
      - "closest": round to the nearest achievable average
      - "le": never exceed the target average (floor)
      - "ge": never go below the target average (ceil)
    Returns (bits_per_layer, achieved_avg, k_high).
    """
    n = len(errors)
    if n == 0:
        return [], float(low_bit), 0

    step = high_bit - low_bit
    if step <= 0:
        raise ValueError("high_bit must be greater than low_bit.")

    p = (target_avg - low_bit) / step
    p = max(0.0, min(1.0, p))  # clamp to [0,1]

    want = p * n
    if mode == "closest":
        k_high = int(round(want))
    elif mode == "le":
        k_high = int(want // 1)          # floor
    elif mode == "ge":
        k_high = int(-(-want // 1))      # ceil
    else:
        raise ValueError("mode must be one of {'closest','le','ge'}")

    # stable sort: by error descending, tie-break by original index
    order = sorted(range(n), key=lambda i: (errors[i], -i), reverse=True)
    high_set = set(order[:k_high])

    bits = [high_bit if i in high_set else low_bit for i in range(n)]
    achieved_avg = low_bit + (k_high / n) * step
    return bits, achieved_avg, k_high
def _dense_ranks_desc(values: List[float]) -> List[int]:
    """
    Dense ranks for descending sort (largest value gets rank=1).
    Ties share the same rank; next distinct value increments by 1.
    """
    n = len(values)
    idx_sorted = sorted(range(n), key=lambda i: (values[i], -i), reverse=True)  # by value desc, then index
    ranks = [0] * n
    current_rank = 0
    last_val = None
    for i in idx_sorted:
        v = values[i]
        if last_val is None or v != last_val:
            current_rank += 1
            last_val = v
        ranks[i] = current_rank
    return ranks


# ---------- Convenience wrapper ----------
def map_jsonl_to_bits(in_path: str,
                      out_path: str = "layer_bits.jsonl",
                      target_avg: float = 4.3,
                      mode: str = "closest") -> None:
    errors = read_errors_jsonl(in_path)
    bits, achieved, k = assign_bits_by_error(errors, target_avg=target_avg, mode=mode)
    ranks = _dense_ranks_desc(errors) 
    write_mapping_jsonl(out_path, errors, bits,ranks)
    print(f"Wrote {len(errors)} rows to {out_path}. "
          f"target_avg={target_avg}, achieved_avg={achieved:.4f}, "
          f"k_high={k} (set to {5}-bit), k_low={len(errors)-k} (set to {4}-bit).")

# Example:
import json, linecache

def get_bit_for_index(mapping_path: str, idx: int) -> int:
    """
    Return the bit for layer at position `idx` (0-based) from a JSONL mapping.
    """
    line = linecache.getline(mapping_path, idx + 1)  # linecache is 1-based
    if not line:
        raise IndexError(f"No line for index {idx} in {mapping_path}")
    rec = json.loads(line)
    return int(rec["bit"])

def get_rank_for_index(mapping_path: str, idx: int) -> int:
    """
    Return the bit for layer at position `idx` (0-based) from a JSONL mapping.
    """
    line = linecache.getline(mapping_path, idx + 1)  # linecache is 1-based
    if not line:
        raise IndexError(f"No line for index {idx} in {mapping_path}")
    rec = json.loads(line)
    return int(rec["rank"])

def get_record_for_index(mapping_path: str, idx: int):
    """If you want both error and bit."""
    line = linecache.getline(mapping_path, idx + 1)
    if not line:
        raise IndexError(f"No line for index {idx}")
    return json.loads(line)
import json, math
from typing import Iterable, Tuple

def average_error_jsonl(path: str,
                        *,
                        keys=("error", "mse", "loss", "quant_error", "err"),
                        skip_nonfinite: bool = True) -> float:
    """
    Compute the average of errors stored in a JSONL file.

    Accepted per-line formats:
      - number                 -> counts as one error
      - {"error": <number>, ...} (or key in `keys`)
      - [e1, e2, ...]          -> counts as many errors (entire line is a list)

    Returns:
        float: mean error across all parsed numbers (NaN if no numbers found).
    """
    total = 0.0
    count = 0

    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)

            def add_val(v):
                nonlocal total, count
                v = float(v)
                if skip_nonfinite and not math.isfinite(v):
                    return
                total += v
                count += 1

            if isinstance(obj, (int, float)):
                add_val(obj)

            elif isinstance(obj, list):
                for v in obj:
                    if not isinstance(v, (int, float)):
                        raise ValueError(f"Line {line_no}: list contains non-numeric value {v!r}")
                    add_val(v)

            elif isinstance(obj, dict):
                val = None
                for k in keys:
                    if k in obj and isinstance(obj[k], (int, float)):
                        val = obj[k]; break
                if val is None:
                    raise ValueError(f"Line {line_no}: dict lacks a numeric field in {keys}")
                add_val(val)

            else:
                raise ValueError(f"Line {line_no}: unsupported JSON type {type(obj)}")

    return (total / count) if count else float("nan")
import json, math
from typing import Tuple

def average_error_jsonl_upto(path: str,
                             max_lines: int,
                             *,
                             keys=("error", "mse", "loss", "quant_error", "err"),
                             skip_nonfinite: bool = True,
                             return_stats: bool = False):
    """
    Compute the average error from the first `max_lines` lines of a JSONL file.

    Each considered line may be:
      - a number                      -> counts as one error
      - a dict with a numeric field in `keys`
      - a list of numbers             -> counts all its elements (if the line is within max_lines)

    Args:
        path: JSONL file path.
        max_lines: only parse lines 1..max_lines (<=0 means no lines).
        keys: candidate field names for dict-style lines.
        skip_nonfinite: ignore NaN/±Inf values if encountered.
        return_stats: if True, return (total, count, mean); else return mean.

    Returns:
        mean error (float) or (total, count, mean) if return_stats=True.
        If no numbers are found, returns NaN (or (0.0, 0, NaN)).
    """
    if max_lines <= 0:
        return (0.0, 0, float("nan")) if return_stats else float("nan")

    total = 0.0
    count = 0

    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            if line_no > max_lines:
                break
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)

            def add_val(v):
                nonlocal total, count
                v = float(v)
                if skip_nonfinite and not math.isfinite(v):
                    return
                total += v
                count += 1

            if isinstance(obj, (int, float)):
                add_val(obj)

            elif isinstance(obj, list):
                for v in obj:
                    if not isinstance(v, (int, float)):
                        raise ValueError(f"Line {line_no}: list contains non-numeric value {v!r}")
                    add_val(v)

            elif isinstance(obj, dict):
                val = None
                for k in keys:
                    if k in obj and isinstance(obj[k], (int, float)):
                        val = obj[k]; break
                if val is None:
                    raise ValueError(f"Line {line_no}: dict lacks a numeric field in {keys}")
                add_val(val)

            else:
                raise ValueError(f"Line {line_no}: unsupported JSON type {type(obj)}")

    mean = (total / count) if count else float("nan")
    return (total, count, mean) if return_stats else mean


###USAGE 
# map_jsonl_to_bits("errors.jsonl", out_path="layer_bits.jsonl", target_avg=4.3, mode="le")