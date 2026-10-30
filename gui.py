import io
import json
import queue
import re
import signal
import sys
import tempfile
import webbrowser
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, TextIO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path("/tmp/cat_dog_cnn.pth")
MODEL_METADATA_PATH = Path("/tmp/cat_dog_cnn.metadata.json")
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


ImageLoader = Callable[[int], Image.Image]
BrowserSelection = Callable[[Path | None, Image.Image, str], None]


class ImageBrowser(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        predict_callback: BrowserSelection,
        folder: Path | None = None,
    ) -> None:
        super().__init__(parent, padding=10)
        self.folder = folder
        self.title = title
        self.predict_callback = predict_callback
        self.paths: list[Path] = []
        self.image_loader: ImageLoader | None = None
        self.labels: list[str] = []
        self.label_loader: Callable[[int], str] | None = None
        self.image_count = 0
        self.selected_index: int | None = None
        self.thumbnail_images: OrderedDict[int, ImageTk.PhotoImage] = OrderedDict()
        self.thumbnail_errors: dict[int, str] = {}
        self.columns = 1
        self.cell_width = 170
        self.cell_height = 160
        self._resize_redraw_id: str | None = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        ttk.Label(self, text=title, font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        browser = ttk.Frame(self)
        browser.grid(row=1, column=0, sticky="nsew")
        browser.columnconfigure(0, weight=1)
        browser.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(browser, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(
            browser, orient="vertical", command=self._scrollbar_yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind("<Configure>", self._schedule_resize_redraw)
        self.canvas.bind("<Button-1>", self._select_from_click)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda _event: self._scroll_by_units(-3))
        self.canvas.bind("<Button-5>", lambda _event: self._scroll_by_units(3))

        controls = ttk.Frame(self)
        controls.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        controls.columnconfigure(0, weight=1)
        self.predict_button = ttk.Button(
            controls, text="Predict selected image", command=self._predict_current,
            state="disabled",
        )
        self.predict_button.grid(row=0, column=1, padx=(8, 0))

        self.info = ttk.Label(self, text="", anchor="w")
        self.info.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        if folder is not None:
            self.set_local_source(folder)

    def set_local_source(self, folder: Path) -> None:
        self.folder = folder
        self.paths = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self.labels = [
            path.parent.name if path.parent != folder else "Unlabelled"
            for path in self.paths
        ]
        self.label_loader = None
        self.image_count = len(self.paths)
        self.image_loader = self._load_local_image
        self._reset_source()

    def set_dataset_source(
        self,
        dataset: object,
        image_count: int,
        label_loader: Callable[[int], str],
    ) -> None:
        self.folder = None
        self.paths = []
        self.labels = []
        self.label_loader = label_loader
        self.image_count = image_count
        self.image_loader = lambda index: dataset[index]["image"].convert("RGB")
        self._reset_source()

    def set_loading(self, message: str) -> None:
        self.image_count = 0
        self.image_loader = None
        self.thumbnail_images.clear()
        self.thumbnail_errors.clear()
        self.canvas.delete("all")
        self.info.configure(text=message)
        self.predict_button.configure(state="disabled")

    def _reset_source(self) -> None:
        self.selected_index = 0 if self.image_count else None
        self.thumbnail_images.clear()
        self.thumbnail_errors.clear()
        self._redraw()
        self._update_info()

    def _load_local_image(self, index: int) -> Image.Image:
        return Image.open(self.paths[index]).convert("RGB")

    def _redraw(self, _event: tk.Event | None = None) -> None:
        if self.image_count == 0 or self.image_loader is None:
            self.canvas.delete("all")
            return

        old_top = self.canvas.canvasy(0)
        self.columns = max(1, self.canvas.winfo_width() // self.cell_width)
        rows = (self.image_count + self.columns - 1) // self.columns
        content_height = rows * self.cell_height
        viewport_height = max(self.canvas.winfo_height(), 1)
        max_top = max(0, content_height - viewport_height)
        top = min(max(old_top, 0), max_top)
        self.canvas.configure(
            scrollregion=(0, 0, self.columns * self.cell_width, content_height)
        )
        self.canvas.yview_moveto(top / max(content_height, 1))
        first_row = max(0, int(self.canvas.canvasy(0) // self.cell_height))
        last_row = min(rows, first_row + int(self.canvas.winfo_height() // self.cell_height) + 2)
        self.canvas.delete("thumbnail")
        for index in range(first_row * self.columns, min(last_row * self.columns, self.image_count)):
            self._draw_thumbnail(index)
        visible_start = first_row * self.columns
        visible_end = min(last_row * self.columns, self.image_count)
        for index in list(self.thumbnail_images):
            if index < visible_start or index >= visible_end:
                del self.thumbnail_images[index]
        while len(self.thumbnail_images) > 300:
            self.thumbnail_images.popitem(last=False)

    def _schedule_resize_redraw(self, _event: tk.Event | None = None) -> None:
        if self._resize_redraw_id is None:
            self._resize_redraw_id = self.after_idle(self._redraw_after_resize)

    def _redraw_after_resize(self) -> None:
        self._resize_redraw_id = None
        self._redraw()

    def _draw_thumbnail(self, index: int) -> None:
        if index in self.thumbnail_errors:
            self.canvas.create_text(
                self._x_for(index) + self.cell_width / 2,
                self._y_for(index) + 60,
                text=f"Unreadable\n{self.thumbnail_errors[index]}",
                tags=("thumbnail", f"image-{index}"),
            )
            return
        try:
            photo = self.thumbnail_images.get(index)
            if photo is None:
                image = self.image_loader(index)
                image.thumbnail((145, 120), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                image.close()
                self.thumbnail_images[index] = photo
            else:
                self.thumbnail_images.move_to_end(index)
        except (OSError, ValueError, KeyError) as error:
            self.thumbnail_errors[index] = str(error)
            self.canvas.create_text(
                self._x_for(index) + self.cell_width / 2,
                self._y_for(index) + 60,
                text=f"Unreadable\n{error}",
                tags=("thumbnail", f"image-{index}"),
            )
            return
        self.canvas.create_image(
            self._x_for(index) + self.cell_width / 2,
            self._y_for(index) + 58,
            image=photo,
            tags=("thumbnail", f"image-{index}"),
        )
        self.canvas.create_text(
            self._x_for(index) + self.cell_width / 2,
            self._y_for(index) + 140,
            text=f"{index + 1}: {self._name_for(index)}",
            tags=("thumbnail", f"image-{index}"),
        )
        if index == self.selected_index:
            self.canvas.create_rectangle(
                self._x_for(index) + 3,
                self._y_for(index) + 3,
                self._x_for(index) + self.cell_width - 3,
                self._y_for(index) + self.cell_height - 3,
                outline="#2a7fff",
                width=3,
                tags=("thumbnail",),
            )

    def _x_for(self, index: int) -> int:
        return (index % self.columns) * self.cell_width

    def _y_for(self, index: int) -> int:
        return (index // self.columns) * self.cell_height

    def _name_for(self, index: int) -> str:
        return self.paths[index].name if self.paths else f"image {index + 1}"

    def _label_for(self, index: int) -> str:
        if self.labels:
            return self.labels[index]
        if self.label_loader is not None:
            return self.label_loader(index)
        return "Unlabelled"

    def _select_from_click(self, event: tk.Event) -> None:
        if self.image_count == 0:
            return
        index = int(self.canvas.canvasx(event.x) // self.cell_width)
        index += int(self.canvas.canvasy(event.y) // self.cell_height) * self.columns
        if 0 <= index < self.image_count:
            self.selected_index = index
            self._redraw()
            self._update_info()

    def _update_info(self) -> None:
        if self.selected_index is None:
            self.info.configure(text="No images available")
            self.predict_button.configure(state="disabled")
            return
        self.info.configure(
            text=f"{self.selected_index + 1} / {self.image_count}    {self._name_for(self.selected_index)}"
        )
        self.predict_button.configure(state="normal")

    def _predict_current(self) -> None:
        if self.selected_index is None or self.image_loader is None:
            return
        try:
            image = self.image_loader(self.selected_index)
        except (OSError, ValueError) as error:
            self.info.configure(text=f"Unable to read selected image: {error}")
            return
        path = self.paths[self.selected_index] if self.paths else None
        self.predict_callback(path, image, self._name_for(self.selected_index))

    def _on_mousewheel(self, event: tk.Event) -> None:
        self._scroll_by_units(int(-event.delta / 120) or -1)

    def _scroll_by_units(self, units: int) -> None:
        self.canvas.yview_scroll(units, "units")
        self._redraw()

    def _scrollbar_yview(self, *args: str) -> None:
        self.canvas.yview(*args)
        self._redraw()

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
        self.selected_temp_image: Path | None = None
        self.selected_photo: ImageTk.PhotoImage | None = None
        self.evaluation_accuracy: str | None = None
        self.epochs_var = tk.StringVar(value="5")
        self.training_browser: ImageBrowser | None = None
        self.browser_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-browser")
        self.training_dataset_future: Future[object] | None = None

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
        self.trained_epochs_label = ttk.Label(panel, text="")
        self.trained_epochs_label.grid(
            row=2, column=0, sticky="w", padx=5, pady=(4, 0)
        )
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
            if key == "evaluate":
                self.confusion_matrix_button = ttk.Button(
                    panel,
                    text="Open confusion matrix",
                    command=self._open_confusion_matrix,
                    state="normal" if Path("confusion_matrix.png").is_file() else "disabled",
                )
                self.confusion_matrix_button.grid(
                    row=1, column=column, sticky="ew", padx=5, pady=(4, 0)
                )
            checkpoint_ready = key == "train" and MODEL_PATH.is_file()
            progress = ttk.Progressbar(
                panel,
                mode="determinate",
                maximum=1,
                value=1 if checkpoint_ready else 0,
            )
            progress.grid(row=3, column=column, sticky="ew", padx=5, pady=(4, 0))
            self.operation_buttons[key] = button
            self.progress_bars[key] = progress

        self.status = ttk.Label(panel, text="Ready", anchor="w")
        self.status.grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=(8, 0))
        self._refresh_trained_epochs()

    @staticmethod
    def _validate_epoch_input(value: str) -> bool:
        return value == "" or value.isdigit()

    def _refresh_trained_epochs(self) -> None:
        self.trained_epochs_label.configure(text="")
        if not MODEL_PATH.is_file() or not MODEL_METADATA_PATH.is_file():
            return
        try:
            with MODEL_METADATA_PATH.open(encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            epochs = metadata.get("epochs")
            if isinstance(epochs, int) and epochs > 0:
                self.trained_epochs_label.configure(
                    text=f"Trained for {epochs} epochs"
                )
        except (OSError, ValueError, TypeError):
            return

    def _open_confusion_matrix(self) -> None:
        confusion_matrix_path = Path("confusion_matrix.png").resolve()
        if not confusion_matrix_path.is_file():
            self.confusion_matrix_button.configure(state="disabled")
            messagebox.showerror(
                "Confusion matrix unavailable",
                "Run model evaluation before opening the confusion matrix.",
            )
            return
        if not webbrowser.open(confusion_matrix_path.as_uri()):
            messagebox.showerror(
                "Unable to open confusion matrix",
                f"Could not open {confusion_matrix_path}.",
            )

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
            self._predict_browser_image,
            ROOT / "testing_data",
        )
        training = ImageBrowser(
            notebook,
            "Training dataset: Hugging Face microsoft/cats_vs_dogs",
            self._predict_browser_image,
        )
        training.set_loading("Loading the training dataset...")
        self.training_browser = training
        self.training_dataset_future = self.browser_executor.submit(
            self._load_training_dataset
        )
        notebook.add(testing, text="Testing images")
        notebook.add(training, text="Training images")

    @staticmethod
    def _load_training_dataset() -> object:
        import torch
        from datasets import load_dataset
        from torch.utils.data import random_split

        raw_dataset = load_dataset(
            "microsoft/cats_vs_dogs",
            split="train",
            cache_dir="/goinfre/krbartos",
        )
        train_size = int(0.8 * len(raw_dataset))
        test_size = len(raw_dataset) - train_size
        train_dataset, _ = random_split(
            raw_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        return train_dataset

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
        self._clear_selected_temp_image()
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

    def _predict_browser_image(
        self, path: Path | None, image: Image.Image, name: str
    ) -> None:
        if self.operation is not None:
            return
        if path is None:
            self._clear_selected_temp_image()
            temporary = tempfile.NamedTemporaryFile(
                suffix=Path(name).suffix or ".png", delete=False
            )
            temporary_path = Path(temporary.name)
            temporary.close()
            image.save(temporary_path)
            self.selected_temp_image = temporary_path
            self.selected_image = temporary_path
            display_image = image.copy()
            self.image_name.configure(text=name)
            display_image.thumbnail((180, 120), Image.Resampling.LANCZOS)
            self.selected_photo = ImageTk.PhotoImage(display_image)
            self.image_preview.configure(image=self.selected_photo)
            self.prediction_result.configure(text="Ready to predict")
        else:
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
        self.confusion_matrix_button.configure(state="disabled")
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
        if self.training_dataset_future is not None and self.training_dataset_future.done():
            future = self.training_dataset_future
            self.training_dataset_future = None
            try:
                dataset = future.result()
                if self.training_browser is not None:
                    self.training_browser.set_dataset_source(
                        dataset,
                        len(dataset),
                        lambda index: ("Cat" if dataset[index]["labels"] == 0 else "Dog"),
                    )
            except Exception as error:
                if self.training_browser is not None:
                    self.training_browser.set_loading(
                        f"Unable to load training data: {error}"
                    )
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
        self.confusion_matrix_button.configure(
            state="normal"
            if Path("confusion_matrix.png").is_file()
            else "disabled"
        )
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
        if operation == "train":
            self._refresh_trained_epochs()
        if operation == "evaluate" and self.evaluation_accuracy is not None:
            self.status.configure(
                text=f"Evaluate completed - accuracy: {self.evaluation_accuracy}"
            )
            return
        self.status.configure(text=f"{operation.title()} completed")

    def _clear_selected_temp_image(self) -> None:
        if self.selected_temp_image is not None:
            try:
                self.selected_temp_image.unlink(missing_ok=True)
            except OSError:
                pass
            self.selected_temp_image = None

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
        self._clear_selected_temp_image()
        self.browser_executor.shutdown(wait=False, cancel_futures=True)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()


if __name__ == "__main__":
    app = CNNDesktopApp()
    signal.signal(signal.SIGINT, app._handle_interrupt)
    app.mainloop()
