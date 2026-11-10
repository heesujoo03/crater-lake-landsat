# -*- coding: utf-8 -*-
"""
Satellite thermal-anomaly pipeline for volcanic crater lakes.

Importing the package first makes console output encoding-safe: the code
prints non-ASCII symbols (e.g. —, ΔT, σ, ≥), and on consoles whose default
code page cannot represent them a final ``print`` would otherwise raise
``UnicodeEncodeError`` after all computation has finished. stdout/stderr are
reconfigured to UTF-8 with ``errors="replace"``. File outputs (CSV/MD/JSON)
are always written as UTF-8 and are unaffected by this setting.
"""

import sys as _sys


def _setup_console():
    for _name in ("stdout", "stderr"):
        _s = getattr(_sys, _name, None)
        if _s is None or not hasattr(_s, "reconfigure"):
            continue
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            try:
                _s.reconfigure(errors="replace")
            except Exception:
                pass


_setup_console()
