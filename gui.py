import io
import queue
import re
import signal
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, TextIO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path("/tmp/cat_dog_cnn.pth")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


class GuiOutput(io.TextIOBase):
    def __init__(
        self,
        messages: queue.Queue[tuple[str, object]],
        cli_stream: TextIO | None,
    ) -> None:
        self.messages = messages
        self.cli_stream = cli_stream
        self.buffer = ""

    def write(self, text: str) -> int:
        if self.cli_stream is not None:
            self.cli_stream.write(text)
            self.cli_stream.flush()
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.messages.put(("output", line.strip()))
        return len(text)

    def flush(self) -> None:
        if self.buffer.strip():
            self.messages.put(("output", self.buffer.strip()))
            self.buffer = ""


class ImageBrowser(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        folder: Path,
        predict_callback: Callable[[Path], None],
    ) -> None:
        super().__init__(parent, padding=10)
        self.folder = folder
        self.title = title
        self.predict_callback = predict_callback
        self.paths: list[Path] = []
        self.index = -1
        self.photo: ImageTk.PhotoImage | None = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        ttk.Label(self, text=title, font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        self.preview = ttk.Label(self, anchor="center", text="No image selected")
        self.preview.grid(row=1, column=0, sticky="nsew")
        self.preview.bind("<Configure>", self._refresh_preview)

        controls = ttk.Frame(self)
        controls.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        controls.columnconfigure(1, weight=1)
        self.previous_button = ttk.Button(
            controls, text="Previous", command=self.previous
        )
        self.previous_button.grid(row=0, column=0, padx=(0, 8))
        self.position = ttk.Label(controls, text="0 / 0")
        self.position.grid(row=0, column=1)
        self.next_button = ttk.Button(controls, text="Next", command=self.next)
        self.next_button.grid(row=0, column=2, padx=(8, 0))
        self.predict_button = ttk.Button(
            controls, text="Predict selected image", command=self._predict_current
        )
        self.predict_button.grid(row=0, column=3, padx=(8, 0))

        self.info = ttk.Label(self, text="", anchor="w")
        self.info.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.reload()
        self.after_idle(self._refresh_preview)

    def reload(self) -> None:
        self.paths = sorted(
            path
            for path in self.folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self.index = 0 if self.paths else -1
        self._show_current()

    def previous(self) -> None:
        if self.paths:
            self.index = (self.index - 1) % len(self.paths)
            self._show_current()

    def next(self) -> None:
        if self.paths:
            self.index = (self.index + 1) % len(self.paths)
            self._show_current()

    def _show_current(self) -> None:
        if self.index < 0:
            self.preview.configure(image="", text=f"No images found in {self.folder}")
            self.position.configure(text="0 / 0")
            self.info.configure(text="")
            self._set_navigation(False)
            return

        path = self.paths[self.index]
        self.position.configure(text=f"{self.index + 1} / {len(self.paths)}")
        label = path.parent.name if path.parent != self.folder else "Unlabelled"
        self.info.configure(text=f"{path.name}    Class: {label}")
        self._set_navigation(True)
        self._refresh_preview()

    def _set_navigation(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.previous_button.configure(state=state)
        self.next_button.configure(state=state)
        self.predict_button.configure(state=state)

    def _predict_current(self) -> None:
        if self.index >= 0:
            self.predict_callback(self.paths[self.index])

    def _refresh_preview(self, _event: tk.Event | None = None) -> None:
        if self.index < 0:
            return
        path = self.paths[self.index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                available_width = max(self.preview.winfo_width() - 20, 100)
                available_height = max(self.preview.winfo_height() - 20, 100)
                image.thumbnail((available_width, available_height), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image)
        except (OSError, ValueError) as error:
            self.photo = None
            self.preview.configure(image="", text=f"Unable to read {path.name}: {error}")
            return
        self.preview.configure(image=self.photo, text="")


class CNNDesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Cats and Dogs CNN")
        self.geometry("1100x760")
        self.minsize(720, 520)
        self.protocol("WM_DELETE_WINDOW", self._close)

        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cnn")
        self.operation: str | None = None
        self.future: Future[None] | None = None
        self.selected_image: Path | None = None
        self.selected_photo: ImageTk.PhotoImage | None = None
        self.evaluation_accuracy: str | None = None
        self.epochs_var = tk.StringVar(value="5")

        self._build_operation_panel()
        self._build_prediction_panel()
        self._build_browsers()
        self.after(100, self._poll_messages)

    def _build_operation_panel(self) -> None:
        panel = ttk.LabelFrame(self, text="CNN operations", padding=10)
        panel.pack(fill="x", padx=10, pady=(10, 5))
        for column in range(3):
            panel.columnconfigure(column, weight=1)

        self.operation_buttons: dict[str, ttk.Button] = {}
        self.progress_bars: dict[str, ttk.Progressbar] = {}
        training_options = ttk.Frame(panel)
        training_options.grid(row=1, column=0, sticky="ew", padx=5, pady=(4, 0))
        training_options.columnconfigure(1, weight=1)
        ttk.Label(training_options, text="Epochs:").grid(
            row=0, column=0, sticky="w", padx=(0, 4)
        )
        self.epochs_spinbox = ttk.Spinbox(
            training_options,
            from_=1,
            to=1000,
            textvariable=self.epochs_var,
            width=8,
            validate="key",
            validatecommand=(self.register(self._validate_epoch_input), "%P"),
        )
        self.epochs_spinbox.grid(row=0, column=1, sticky="ew")
        labels = (("train", "Train model"), ("evaluate", "Evaluate model"), ("predict", "Predict one image"))
        for column, (key, label) in enumerate(labels):
            button = ttk.Button(panel, text=label, command=lambda key=key: self._start(key))
            button.grid(row=0, column=column, sticky="ew", padx=5)
            checkpoint_ready = key == "train" and MODEL_PATH.is_file()
            progress = ttk.Progressbar(
                panel,
                mode="determinate",
                maximum=1,
                value=1 if checkpoint_ready else 0,
            )
            progress.grid(row=2, column=column, sticky="ew", padx=5, pady=(8, 0))
            self.operation_buttons[key] = button
            self.progress_bars[key] = progress

        self.status = ttk.Label(panel, text="Ready", anchor="w")
        self.status.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=(8, 0))

    @staticmethod
    def _validate_epoch_input(value: str) -> bool:
        return value == "" or value.isdigit()

    def _build_prediction_panel(self) -> None:
        panel = ttk.LabelFrame(self, text="Selected image and result", padding=10)
        panel.pack(fill="x", padx=10, pady=5)
        panel.columnconfigure(1, weight=1)
        self.choose_button = ttk.Button(panel, text="Choose image", command=self._choose_image)
        self.choose_button.grid(
            row=0, column=0, sticky="w"
        )
        self.image_name = ttk.Label(panel, text="No image selected")
        self.image_name.grid(row=0, column=1, sticky="w", padx=10)
        self.image_preview = ttk.Label(panel)
        self.image_preview.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.prediction_result = ttk.Label(panel, text="", anchor="w")
        self.prediction_result.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _build_browsers(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        testing = ImageBrowser(
            notebook,
            "Testing dataset: testing_data/",
            ROOT / "testing_data",
            self._predict_browser_image,
        )
        training = ImageBrowser(
            notebook,
            "Training image browser: testowe/ (local image collection)",
            ROOT / "testowe",
            self._predict_browser_image,
        )
        notebook.add(testing, text="Testing images")
        notebook.add(training, text="Training images")

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._set_selected_image(Path(path))

    def _set_selected_image(self, path: Path) -> None:
        self.selected_image = path
        self.image_name.configure(text=str(self.selected_image))
        try:
            with Image.open(self.selected_image) as image:
                image.thumbnail((180, 120), Image.Resampling.LANCZOS)
                self.selected_photo = ImageTk.PhotoImage(image.convert("RGB"))
        except (OSError, ValueError) as error:
            self.selected_photo = None
            self.prediction_result.configure(text=f"Unable to display image: {error}")
            return
        self.image_preview.configure(image=self.selected_photo)
        self.prediction_result.configure(text="Ready to predict")

    def _predict_browser_image(self, path: Path) -> None:
        if self.operation is not None:
            return
        self._set_selected_image(path)
        self._start("predict")

    def _start(self, operation: str) -> None:
        if self.operation is not None:
            return
        if operation == "predict" and self.selected_image is None:
            self._choose_image()
            if self.selected_image is None:
                return

        self.operation = operation
        if operation == "predict":
            self.prediction_result.configure(text="Predicting...")
        if operation == "evaluate":
            self.evaluation_accuracy = None
        if operation == "train":
            try:
                epochs = int(self.epochs_var.get())
            except ValueError:
                self._reset_operation_state()
                messagebox.showerror("Invalid epochs", "Training epochs must be a positive whole number.")
                return
            if epochs < 1:
                self._reset_operation_state()
                messagebox.showerror("Invalid epochs", "Training epochs must be at least 1.")
                return
        else:
            epochs = None
        self.status.configure(text=f"{operation.title()} is running...")
        for button in self.operation_buttons.values():
            button.configure(state="disabled")
        self.epochs_spinbox.configure(state="disabled")
        self.choose_button.configure(state="disabled")
        progress = self.progress_bars[operation]
        progress.configure(mode="determinate", value=0)
        if operation == "train":
            progress.configure(maximum=epochs)
        elif operation == "predict":
            progress.configure(maximum=1)
        self.future = self.executor.submit(self._run_operation, operation, epochs)

    def _run_operation(self, operation: str, epochs: int | None) -> None:
        cli_stream = getattr(sys, "__stdout__", None)
        output = GuiOutput(self.messages, cli_stream)
        progress_callback = lambda current, total: self.messages.put(
            ("progress", (current, total))
        )
        try:
            with redirect_stdout(output):
                if operation == "train":
                    from train import train_model

                    train_model(epochs=epochs, progress_callback=progress_callback)
                elif operation == "evaluate":
                    from evaulate import evaluate_model

                    evaluate_model(progress_callback=progress_callback)
                else:
                    from predict import predict_image

                    predict_image(str(self.selected_image))
            output.flush()
            self.messages.put(("complete", ""))
        except Exception as error:
            output.flush()
            self.messages.put(("error", f"{type(error).__name__}: {error}"))

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "output":
                    self._handle_output(str(value))
                elif kind == "progress":
                    current, total = value
                    self._update_progress(current, total)
                elif kind == "complete":
                    self._finish(None)
                else:
                    self._finish(value)
        except queue.Empty:
            pass
        self.after(100, self._poll_messages)

    def _handle_output(self, line: str) -> None:
        self.status.configure(text=line)
        if self.operation == "predict" and line.startswith("[WYNIK]"):
            self.prediction_result.configure(text=line)
        if self.operation == "evaluate":
            match = re.search(r"^\s*accuracy\s+([0-9]+(?:\.[0-9]+)?)", line)
            if match:
                accuracy = float(match.group(1))
                if accuracy <= 1:
                    accuracy *= 100
                self.evaluation_accuracy = f"{accuracy:.2f}%"
                self.status.configure(text=f"Evaluation accuracy: {self.evaluation_accuracy}")
        if line.startswith("Błąd:"):
            self.status.configure(text=line)

    def _update_progress(self, current: int, total: int) -> None:
        if self.operation is None or total <= 0:
            return
        progress = self.progress_bars[self.operation]
        progress.configure(mode="determinate", maximum=total, value=current)
        if self.operation == "train":
            self.status.configure(text=f"Training epoch {current}/{total}")
        elif self.operation == "evaluate":
            self.status.configure(text=f"Evaluating batch {current}/{total}")

    def _finish(self, error: str | None) -> None:
        if self.operation is None:
            return
        operation = self.operation
        progress = self.progress_bars[operation]
        self.operation = None
        self.future = None
        for button in self.operation_buttons.values():
            button.configure(state="normal")
        self.epochs_spinbox.configure(state="normal")
        self.choose_button.configure(state="normal")

        if error:
            self.status.configure(text=f"{operation.title()} failed: {error}")
            if operation == "predict":
                self.prediction_result.configure(text=error)
            return
        if self.status.cget("text").startswith("Błąd:"):
            self.status.configure(text=f"{operation.title()} failed")
            return
        progress.configure(mode="determinate", value=progress["maximum"])
        if operation == "evaluate" and self.evaluation_accuracy is not None:
            self.status.configure(
                text=f"Evaluate completed - accuracy: {self.evaluation_accuracy}"
            )
            return
        self.status.configure(text=f"{operation.title()} completed")

    def _reset_operation_state(self) -> None:
        self.operation = None

    def _close(self) -> None:
        if self.operation is not None:
            messagebox.showinfo("Operation in progress", "Wait for the current operation to finish.")
            return
        self._destroy_window()

    def _handle_interrupt(self, _signal_number: int, _stack_frame: object) -> None:
        self._destroy_window()

    def _destroy_window(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()


if __name__ == "__main__":
    app = CNNDesktopApp()
    signal.signal(signal.SIGINT, app._handle_interrupt)
    app.mainloop()
