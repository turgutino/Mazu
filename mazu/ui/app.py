"""Terminal UI (`mazu ui`) -- a read-mostly, interactive view over the same data
`mazu timeline`/`mazu memory list`/`mazu log` already expose, plus the two actions
that matter most from an interactive view: rolling back to a checkpoint and
pinning/unpinning a memory. Every mutation goes through the exact same store methods
the CLI commands use (`CheckpointManager.restore`, `MemoryStore.pin`/`unpin`) --
this is a different way to reach the same operations, not a second implementation
of them.
"""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Label, TabbedContent, TabPane

from mazu.action_log.store import ActionLogStore
from mazu.checkpoint.manager import CheckpointManager
from mazu.memory.store import MemoryStore
from mazu.ui.data import load_actions, load_checkpoints, load_memories, load_sessions


class ConfirmModal(ModalScreen[bool]):
    """A small Yes/Cancel dialog. Dismisses with True (confirmed) or False
    (cancelled) -- callers pass a callback to push_screen() to react to the result.
    """

    DEFAULT_CSS = """
    ConfirmModal {
        align: center middle;
    }
    #confirm-dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        border: thick $error;
        background: $surface;
    }
    #confirm-buttons {
        height: auto;
        align: right middle;
        padding-top: 1;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(self.message)
            with Horizontal(id="confirm-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Confirm", id="confirm", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")


class MazuApp(App):
    """Three tabs: Checkpoints, Memory, Action Log. `r` on a checkpoint rolls back
    to it (confirmed first); `p` on a memory toggles its pinned state; selecting a
    session in Action Log loads that session's individual tool calls below it.
    """

    TITLE = "Mazu"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "rollback_selected", "Rollback"),
        ("p", "toggle_pin_selected", "Pin/Unpin"),
    ]
    CSS = """
    DataTable {
        height: 1fr;
    }
    """

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.checkpoint_manager = CheckpointManager(root)
        self.memory_store = MemoryStore(root / ".mazu" / "memory.db")
        self.action_log_store = ActionLogStore(root / ".mazu" / "action_log.db")
        self._checkpoints: list = []
        self._memories: list = []
        self._sessions: list = []

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="checkpoints-tab"):
            with TabPane("Checkpoints", id="checkpoints-tab"):
                yield DataTable(id="checkpoint_table", cursor_type="row")
            with TabPane("Memory", id="memory-tab"):
                yield DataTable(id="memory_table", cursor_type="row")
            with TabPane("Action Log", id="actionlog-tab"):
                with Vertical():
                    yield DataTable(id="session_table", cursor_type="row")
                    yield DataTable(id="action_table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.reload_checkpoints()
        self.reload_memory()
        self.reload_sessions()

    # -- Checkpoints -----------------------------------------------------------

    def reload_checkpoints(self) -> None:
        table = self.query_one("#checkpoint_table", DataTable)
        table.clear(columns=True)
        table.add_columns("ID", "Created", "Trigger", "Files changed")
        self._checkpoints = load_checkpoints(self.checkpoint_manager)
        for cp in self._checkpoints:
            table.add_row(
                cp.id, cp.created_at, cp.trigger, ", ".join(cp.files_changed) or "-", key=cp.id
            )

    def action_rollback_selected(self) -> None:
        table = self.query_one("#checkpoint_table", DataTable)
        if not table.has_focus or table.cursor_row is None or not self._checkpoints:
            return
        checkpoint_id = self._checkpoints[table.cursor_row].id

        def _handle(confirmed: bool | None) -> None:
            if not confirmed:
                return
            self.checkpoint_manager.restore(checkpoint_id)
            self.reload_checkpoints()
            self.notify(f"Rolled back to {checkpoint_id}.")

        self.push_screen(
            ConfirmModal(
                f"Roll back to {checkpoint_id}? This discards code and memory changes since then."
            ),
            _handle,
        )

    # -- Memory ------------------------------------------------------------

    def reload_memory(self) -> None:
        table = self.query_one("#memory_table", DataTable)
        table.clear(columns=True)
        table.add_columns("ID", "Category", "Pinned", "Title")
        self._memories = load_memories(self.memory_store)
        for mem in self._memories:
            table.add_row(
                str(mem.id), mem.category, "*" if mem.pinned else "", mem.title, key=str(mem.id)
            )

    def action_toggle_pin_selected(self) -> None:
        table = self.query_one("#memory_table", DataTable)
        if not table.has_focus or table.cursor_row is None or not self._memories:
            return
        mem = self._memories[table.cursor_row]
        if mem.pinned:
            self.memory_store.unpin(mem.id)
            self.notify(f"Unpinned memory {mem.id}.")
        else:
            self.memory_store.pin(mem.id)
            self.notify(f"Pinned memory {mem.id}.")
        self.reload_memory()

    # -- Action log ----------------------------------------------------------

    def reload_sessions(self) -> None:
        table = self.query_one("#session_table", DataTable)
        table.clear(columns=True)
        table.add_columns("Session", "Command", "Actions", "Errors", "Started")
        self._sessions = load_sessions(self.action_log_store)
        for s in self._sessions:
            table.add_row(
                s.session_id, s.command, str(s.action_count), str(s.error_count), s.started_at,
                key=s.session_id,
            )
        action_table = self.query_one("#action_table", DataTable)
        action_table.clear(columns=True)
        action_table.add_columns("Time", "Tool", "Outcome", "Output")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "session_table":
            return
        session_id = str(event.row_key.value)
        action_table = self.query_one("#action_table", DataTable)
        action_table.clear(columns=True)
        action_table.add_columns("Time", "Tool", "Outcome", "Output")
        for a in load_actions(self.action_log_store, session_id):
            action_table.add_row(a.created_at, a.tool_name, a.outcome, a.output_summary)

    def on_unmount(self) -> None:
        self.memory_store.close()
        self.action_log_store.close()
