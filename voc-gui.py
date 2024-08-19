import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import voc

# Version Information
VER = '0.1.0'
API = voc.VER

# Helper Functions
def validate_dir(path: Path):
    """Create directory if it doesn't exist."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

# Main GUI Class
class VocGuiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Voc CLI GUI (Version {VER})")

        # Cache, Smoothness, FFT, Y-Offset Options
        self.cache_var = tk.BooleanVar(value=True)
        self.smoothness_var = tk.IntVar(value=10)
        self.fft_var = tk.BooleanVar(value=False)
        self.y_offset_var = tk.DoubleVar(value=0.0)

        self.create_widgets()

    def create_widgets(self):
        """Create the GUI layout similar to CLI options and commands."""

        # Options Frame
        options_frame = tk.Frame(self)
        options_frame.pack(padx=10, pady=10, fill=tk.X)

        tk.Checkbutton(options_frame, text="Cache", variable=self.cache_var).grid(row=0, column=0, sticky=tk.W)
        tk.Label(options_frame, text="Smoothness:").grid(row=0, column=1, sticky=tk.E)
        tk.Entry(options_frame, textvariable=self.smoothness_var).grid(row=0, column=2, sticky=tk.W)

        tk.Checkbutton(options_frame, text="FFT", variable=self.fft_var).grid(row=1, column=0, sticky=tk.W)
        tk.Label(options_frame, text="Y-Offset:").grid(row=1, column=1, sticky=tk.E)
        tk.Entry(options_frame, textvariable=self.y_offset_var).grid(row=1, column=2, sticky=tk.W)

        # Folder Selection Frame
        folder_frame = tk.Frame(self)
        folder_frame.pack(padx=10, pady=10, fill=tk.X)

        tk.Label(folder_frame, text="Folder:").grid(row=0, column=0, sticky=tk.E)
        self.folder_entry = tk.Entry(folder_frame, width=50)
        self.folder_entry.grid(row=0, column=1, sticky=tk.W)
        tk.Button(folder_frame, text="Browse...", command=self.select_folder).grid(row=0, column=2, sticky=tk.W)

        # Save Directory Frame
        save_dir_frame = tk.Frame(self)
        save_dir_frame.pack(padx=10, pady=10, fill=tk.X)

        tk.Label(save_dir_frame, text="Save Dir:").grid(row=0, column=0, sticky=tk.E)
        self.save_dir_entry = tk.Entry(save_dir_frame, width=50)
        self.save_dir_entry.grid(row=0, column=1, sticky=tk.W)
        tk.Button(save_dir_frame, text="Browse...", command=self.select_save_dir).grid(row=0, column=2, sticky=tk.W)

        # Buttons for Operations
        buttons_frame = tk.Frame(self)
        buttons_frame.pack(padx=10, pady=10, fill=tk.X)

        tk.Button(buttons_frame, text="Plot Signals", command=self.plot_signals).grid(row=0, column=0, padx=5)
        tk.Button(buttons_frame, text="Show Avg Signal", command=self.show_avg_signal).grid(row=0, column=1, padx=5)
        tk.Button(buttons_frame, text="Compare Runs", command=self.compare_runs).grid(row=0, column=2, padx=5)
        tk.Button(buttons_frame, text="Show Version Info", command=self.show_version_info).grid(row=0, column=3, padx=5)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Folder")
        if folder:
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder)

    def select_save_dir(self):
        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if save_dir:
            self.save_dir_entry.delete(0, tk.END)
            self.save_dir_entry.insert(0, save_dir)

    def plot_signals(self):
        folder = self.folder_entry.get()
        save_dir = self.save_dir_entry.get()
        if not folder or not save_dir:
            messagebox.showerror("Error", "Both folder and save directory must be selected.")
            return

        save_dir_path = Path(save_dir)
        validate_dir(save_dir_path)

        signals = voc.Run(folder, cache=self.cache_var.get(), smoothness=self.smoothness_var.get(), y_offset=self.y_offset_var.get())
        signals.plot(save_dir_path, fft=self.fft_var.get())
        messagebox.showinfo("Success", f"Plotted signals to folder: {save_dir_path}")

    def show_avg_signal(self):
        folder = self.folder_entry.get()
        if not folder:
            messagebox.showerror("Error", "Folder must be selected.")
            return

        signals = voc.Run(folder, cache=self.cache_var.get(), smoothness=self.smoothness_var.get(), y_offset=self.y_offset_var.get())
        signals.show_avg_signal(fft=self.fft_var.get())
        messagebox.showinfo("Success", f"Displayed average signal for run in folder: {folder}")

    def compare_runs(self):
        folder_a = filedialog.askdirectory(title="Select Folder A")
        folder_b = filedialog.askdirectory(title="Select Folder B")
        print(folder_a, folder_b)
        save_dir = self.save_dir_entry.get()

        if not folder_a or not folder_b:
            messagebox.showerror("Error", "Both Folder A and Folder B must be selected.")
            return

        save_path = Path(save_dir) if save_dir else None
        show_plot = True if not save_dir else False

        if save_path:
            validate_dir(save_path)

        run_a = voc.Run(folder_a, cache=self.cache_var.get(), smoothness=self.smoothness_var.get(), y_offset=self.y_offset_var.get())
        run_b = voc.Run(folder_b, cache=self.cache_var.get(), smoothness=self.smoothness_var.get(), y_offset=self.y_offset_var.get())
        voc.plot_average_signals(run_a, run_b, save_path, fft=self.fft_var.get(), show=show_plot)

        if save_path:
            messagebox.showinfo("Success", f"Compared average signals and saved plot to {save_path}")
        else:
            messagebox.showinfo("Success", "Compared average signals and displayed the plot.")

    def show_version_info(self):
        """Display version information."""
        version_info = f"VOC GUI: v{VER}\nUsing VOC API: v{API}"
        messagebox.showinfo("Version Info", version_info)

if __name__ == "__main__":
    app = VocGuiApp()
    app.mainloop()
