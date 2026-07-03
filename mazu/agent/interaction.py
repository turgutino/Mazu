def safe_confirm(prompt: str) -> bool:
    """input()-based y/N confirmation that defaults to declining (False) instead of
    crashing the whole process if stdin closes or runs out mid-prompt (piped input
    exhausted, terminal closed, etc.) — found via live testing: an EOFError here
    previously propagated uncaught all the way up and killed `mazu chat`/`mazu run`
    even after a destructive action had already been safely approved earlier in the
    same session.
    """
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n(no input available — declining by default)")
        return False
    return answer == "y"
