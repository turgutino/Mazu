import sys

_RED = "\x1b[31m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"

LOGO = r"""
  ███╗   ███╗
  ████╗ ████║   A Z U
  ██╔████╔██║   ══════════════════════════════
  ██║╚██╔╝██║   memory-augmented coding agent
  ██║ ╚═╝ ██║   persistent memory · checkpoints · skills · multi-model
  ╚═╝     ╚═╝
"""

# Plain-ASCII fallback for consoles that can't encode the box-drawing characters
# above (mazu.cli forces UTF-8 on stdout, so this shouldn't trigger for the real
# `mazu` command, but print_banner() may also be called by code/tests that import
# mazu.agent.* directly without going through that entry point).
LOGO_ASCII = r"""
   __  __
  |  \/  |  A Z U
  | |\/| |  ------------------------------
  |_|  |_|  memory-augmented coding agent
"""


def print_banner() -> None:
    text = f"{_BOLD}{_RED}{LOGO}{_RESET}" if sys.stdout.isatty() else LOGO
    try:
        print(text)
    except UnicodeEncodeError:
        print(LOGO_ASCII)
