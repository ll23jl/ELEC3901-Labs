"""
labcheck.py - self-checks for ELEC3901 lab tasks.

    from labcheck import check
    check('fk at test pose', lambda: my_fk(q_test), 'a3f9c1e2b0', decimals=3)

The expected value is stored as a short hash of the rounded result, so a passing check tells you your
function agrees with the reference to the stated number of decimals without revealing the number.
Shapes and ordering matter: return exactly what the task's docstring asks for.
"""
import hashlib
import numpy as np


def digest(value, decimals=3):
    a = np.round(np.asarray(value, dtype=float), decimals) + 0.0     # +0.0 turns -0.0 into 0.0
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:10]


def check(label, value, expected, decimals=3):
    """value may be the result itself or a zero-argument callable (evaluated here so errors are caught)."""
    try:
        if callable(value):
            value = value()
        got = digest(value, decimals)
    except Exception as e:                      # NotImplementedError, wrong type, etc.
        print(f'  x  {label}: could not evaluate ({type(e).__name__}: {e})')
        return False
    ok = got == expected
    print(('  OK ' if ok else '  x  ') + label + ('' if ok else '   -> result does not match the reference (check shape, units, sign, rounding)'))
    return ok
