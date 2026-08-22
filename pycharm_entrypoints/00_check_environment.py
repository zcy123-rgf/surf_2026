"""Open this file in PyCharm and run it to validate the workstation."""

from __future__ import annotations

from common import run_main
from scripts import check_windows_env


if __name__ == "__main__":
    run_main(check_windows_env.main, ["--device", "cuda", "--run-clrnet"])
