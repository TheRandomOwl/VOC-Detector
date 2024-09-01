import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess

class VOCGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VOC GUI")

        # CLI Path
        self.cli_path_var = tk.StringVar(value="voc-cli.py")

        # GUI Elements
        self.cli_path_frame()
        self.make_options()
        self.make_buttons()
        self.cli_output()

    def cli_path_frame(self):
        cli_frame = tk.Frame(self.root)
        cli_frame.pack(padx=10, pady=10, fill=tk.X)

        tk.Label(cli_frame, text="VOC-CLI Path:").grid(row=0, column=0, sticky=tk.E)
        tk.Entry(cli_frame, textvariable=self.cli_path_var, width=50).grid(row=0, column=1, sticky=tk.W)
        tk.Button(cli_frame, text="Browse...", command=self.select_cli_path).grid(row=0, column=2, sticky=tk.W)

    def make_options(self):
        options_frame = tk.Frame(self.root)
        options_frame.pack(padx=10, pady=10, fill=tk.X)

        self.cache_var = tk.BooleanVar(value=True)
        self.smoothness_var = tk.IntVar(value=10)
        self.fft_var = tk.BooleanVar(value=False)
        self.y_offset_var = tk.DoubleVar(value=0.0)
        self.threshold_var = tk.DoubleVar(value=0.0)

        tk.Checkbutton(options_frame, text="Cache", variable=self.cache_var).grid(row=0, column=0, sticky=tk.W)
        tk.Label(options_frame, text="Smoothness:").grid(row=0, column=1, sticky=tk.E)
        tk.Entry(options_frame, textvariable=self.smoothness_var).grid(row=0, column=2, sticky=tk.W)

        tk.Checkbutton(options_frame, text="FFT", variable=self.fft_var).grid(row=1, column=0, sticky=tk.W)
        tk.Label(options_frame, text="Y-Offset:").grid(row=1, column=1, sticky=tk.E)
        tk.Entry(options_frame, textvariable=self.y_offset_var).grid(row=1, column=2, sticky=tk.W)

        tk.Label(options_frame, text="Threshold:").grid(row=2, column=1, sticky=tk.E)
        tk.Entry(options_frame, textvariable=self.threshold_var).grid(row=2, column=2, sticky=tk.W)

    def make_buttons(self):
        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(padx=10, pady=10, fill=tk.X)

        tk.Button(buttons_frame, text="Plot Signals", command=self.run_plot).grid(row=0, column=0, padx=5)
        tk.Button(buttons_frame, text="Show Avg Signal", command=self.run_average).grid(row=0, column=1, padx=5)
        tk.Button(buttons_frame, text="Compare Runs", command=self.run_compare).grid(row=0, column=2, padx=5)
        tk.Button(buttons_frame, text="Export to One CSV", command=lambda: self.run_export(single=True)).grid(row=1, column=0, padx=5)
        tk.Button(buttons_frame, text="Export to Multiple CSVs", command=lambda: self.run_export(single=False)).grid(row=1, column=1, padx=5)
        tk.Button(buttons_frame, text="Show Version Info", command=self.show_version_info).grid(row=1, column=2, padx=5)

    def cli_output(self):
        self.output_text = tk.Text(self.root, height=15, width=60)
        self.output_text.pack(pady=10)

    def run_cli_command(self, command, *args):
        try:
            cli_path = self.cli_path_var.get()
            result = subprocess.run(
                [cli_path, command] + list(args),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, result.stdout)
            if result.stderr:
                messagebox.showerror("Error", result.stderr)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_average(self):
        folder = filedialog.askdirectory(title="Select Data Folder")
        if folder:
            save_dir = filedialog.askdirectory(title="Select Save Directory")
            if save_dir:
                self.run_cli_command("average", "--save-dir", save_dir, folder)

    def run_compare(self):
        folder_a = filedialog.askdirectory(title="Select Folder A")
        folder_b = filedialog.askdirectory(title="Select Folder B")
        if folder_a and folder_b:
            save_dir = filedialog.askdirectory(title="Select Save Directory")
            if save_dir:
                self.run_cli_command("compare", "--save-dir", save_dir, folder_a, folder_b)

    def run_export(self, single=True):
        folder = filedialog.askdirectory(title="Select Data Folder")
        if folder:
            save_path = filedialog.askdirectory(title="Select Save Directory")
            if save_path:
                save_as = "single" if single else "multi"
                self.run_cli_command("export", "--save-as", save_as, folder, save_path)

    def run_plot(self):
        folder = filedialog.askdirectory(title="Select Data Folder")
        if folder:
            save_dir = filedialog.askdirectory(title="Select Save Directory")
            if save_dir:
                self.run_cli_command("plot", folder, save_dir)

    def show_version_info(self):
        self.run_cli_command("--version")

    def select_cli_path(self):
        path = filedialog.askopenfilename(title="Select VOC-CLI Executable")
        if path:
            self.cli_path_var.set(path)

if __name__ == '__main__':
    root = tk.Tk()
    app = VOCGUI(root)
    root.mainloop()

