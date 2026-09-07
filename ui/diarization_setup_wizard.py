"""Step-by-step guided setup for speaker diarization's Hugging Face access.

Walks through: paste a token (verified via whoami), then each required
gated repo one at a time (open its page / test it / move on), ending in
a confirmation screen. Replaces a single wall of instructions with one
concrete thing to do at a time -- each step only unlocks "Next" once
that step's own check has actually passed.
"""
from __future__ import annotations

import webbrowser

from PySide6.QtCore import QSettings, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from core.diarization import REQUIRED_REPOS, BackgroundCheckWorker, check_repo_access, check_whoami, verification_key


class DiarizationSetupWizard(QDialog):
    setup_completed = Signal(str)  # emits the verified token

    def __init__(self, parent, existing_token: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Speaker Identification Setup")
        self.setMinimumWidth(480)
        self._settings = QSettings("TranscriberGUI", "TranscriberGUI")
        self._token = existing_token
        # step 0 = token entry, 1..len(REQUIRED_REPOS) = one per repo,
        # len(REQUIRED_REPOS)+1 = final confirmation screen
        self._step = 0
        self._worker = None
        self._thread = None
        self._current_status_label = None
        self._current_retry_button = None
        self._current_on_success = None

        self._layout = QVBoxLayout(self)
        self._content = QVBoxLayout()
        self._layout.addLayout(self._content)

        self._render_step()

    # ------------------------------------------------------------------
    # Step rendering
    # ------------------------------------------------------------------
    def _clear_content(self) -> None:
        while self._content.count():
            item = self._content.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                layout = item.layout()
                if layout is not None:
                    self._clear_layout(layout)

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _render_step(self) -> None:
        self._clear_content()
        total_repo_steps = len(REQUIRED_REPOS)
        if self._step == 0:
            self._render_token_step()
        elif self._step <= total_repo_steps:
            self._render_repo_step(REQUIRED_REPOS[self._step - 1], self._step, total_repo_steps)
        else:
            self._render_done_step()

    def _render_token_step(self) -> None:
        self._content.addWidget(QLabel(
            "<b>Step 1: Add your Hugging Face token</b>"
        ))
        info = QLabel(
            "Speaker identification needs a free Hugging Face token, one-time:<br>"
            "1. Create a free account: <a href=\"https://huggingface.co/join\">huggingface.co/join</a><br>"
            "2. Create a token (\"Read\" access is enough): "
            "<a href=\"https://huggingface.co/settings/tokens\">huggingface.co/settings/tokens</a><br>"
            "3. Paste it below."
        )
        info.setOpenExternalLinks(True)
        info.setWordWrap(True)
        self._content.addWidget(info)

        token_input = QLineEdit()
        token_input.setEchoMode(QLineEdit.EchoMode.Password)
        token_input.setPlaceholderText("hf_...")
        token_input.setText(self._token)
        self._content.addWidget(token_input)

        status = QLabel("")
        status.setWordWrap(True)
        self._content.addWidget(status)

        next_button = QPushButton("Next")

        def on_next() -> None:
            token = token_input.text().strip()
            if not token:
                status.setText("Paste a token first.")
                return
            self._token = token
            next_button.setEnabled(False)
            status.setText("Checking token...")
            self._run_check(check_whoami, (token,), status, next_button, on_success=self._advance)

        next_button.clicked.connect(on_next)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(next_button)
        self._content.addLayout(button_row)

    def _render_repo_step(self, repo_id: str, step_num: int, total: int) -> None:
        self._content.addWidget(QLabel(
            f"<b>Step {step_num + 1} of {total + 1}: {repo_id}</b>"
        ))
        info = QLabel(
            "This model is gated. If you haven't already, open its page while "
            "logged in, scroll down to the access-request form, fill it in "
            "(e.g. company/university -- anything reasonable works), and click "
            "\"Agree and access repository\". Then come back and click Test."
        )
        info.setWordWrap(True)
        self._content.addWidget(info)

        status = QLabel("")
        status.setWordWrap(True)
        self._content.addWidget(status)

        open_button = QPushButton("Open model page")
        open_button.clicked.connect(lambda: webbrowser.open(f"https://huggingface.co/{repo_id}"))

        test_button = QPushButton("Test")
        next_button = QPushButton("Next")
        next_button.setEnabled(False)

        def on_test() -> None:
            test_button.setEnabled(False)
            status.setText("Checking access...")
            self._run_check(
                check_repo_access, (repo_id, self._token), status, test_button,
                on_success=lambda: (status.setText("Access confirmed."), next_button.setEnabled(True)),
            )

        test_button.clicked.connect(on_test)
        next_button.clicked.connect(self._advance)

        button_row = QHBoxLayout()
        button_row.addWidget(open_button)
        button_row.addStretch(1)
        button_row.addWidget(test_button)
        button_row.addWidget(next_button)
        self._content.addLayout(button_row)

    def _render_done_step(self) -> None:
        self._content.addWidget(QLabel("<b>All set!</b>"))
        self._content.addWidget(QLabel("Speaker identification is ready to use."))

        done_button = QPushButton("Done")
        done_button.setStyleSheet("font-weight: bold; padding: 8px 24px;")
        done_button.clicked.connect(self._finish)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(done_button)
        button_row.addStretch(1)
        self._content.addLayout(button_row)

    # ------------------------------------------------------------------
    # Background check plumbing
    # ------------------------------------------------------------------
    def _run_check(self, fn, args, status_label, retry_button, on_success) -> None:
        # Context for whichever check is currently running, read back by
        # the two handlers below -- NOT passed via a lambda closure on the
        # signal connection itself. A signal connected to a bare lambda
        # (confirmed directly, the hard way, earlier in this app) breaks
        # both Qt's cross-thread auto-connection detection AND
        # self.sender() tracking inside the called function, since the
        # lambda -- not `self` -- is what Qt actually sees as connected.
        # Real bound methods of `self` (a QObject) are what make both of
        # those work correctly.
        self._current_status_label = status_label
        self._current_retry_button = retry_button
        self._current_on_success = on_success

        self._worker = BackgroundCheckWorker(lambda: fn(*args))
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        worker = self._worker
        thread = self._thread

        thread.started.connect(worker.run)
        worker.succeeded.connect(self._on_check_succeeded)
        worker.failed.connect(self._on_check_failed)
        worker.succeeded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._cleanup_check_thread)

        thread.start()

    def _on_check_succeeded(self) -> None:
        if self.sender() is not self._worker:
            return
        self._current_retry_button.setEnabled(True)
        self._current_on_success()

    def _on_check_failed(self, message: str) -> None:
        if self.sender() is not self._worker:
            return
        self._current_status_label.setText(message)
        self._current_retry_button.setEnabled(True)

    def _cleanup_check_thread(self) -> None:
        if self.sender() is not self._thread:
            return
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _advance(self) -> None:
        self._step += 1
        self._render_step()

    def _finish(self) -> None:
        self._settings.setValue("hf_token", self._token)
        self._settings.setValue("hf_token_verified_key", verification_key(self._token))
        self.setup_completed.emit(self._token)
        self.accept()
