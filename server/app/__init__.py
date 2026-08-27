"""Package init.

Rocky Linux 9 ships sqlite 3.34, but Chroma requires >= 3.35. We shim the
stdlib name with `pysqlite3` (bundled newer sqlite) BEFORE any chromadb
import elsewhere in the package.
"""
import sys

try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2  # type: ignore[attr-defined]
except ImportError:
    # pysqlite3 not installed - fall back to system sqlite3 (may fail on Chroma)
    pass
