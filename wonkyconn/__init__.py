__packagename__ = "wonkyconn"
__copyright__ = "2024, SIMEXP"

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

__all__ = [
    "__copyright__",
    "__packagename__",
    "__version__",
]
