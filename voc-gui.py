import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import platform
from pathlib import Path
from shutil import which
import threading
import webbrowser

VER = '0.4.2'

class Gui:
    def __init__(self, root):
        self.root = root
        self.root.title(f"VOC GUI v{VER}")

        # Allow the window to resize
        self.root.geometry("600x500")  # Set an initial size for the window

        # Automatically set the path to the CLI executable based on the platform
        if platform.system() == "Windows":
            path = (Path(__file__).parent/'..'/'..').resolve() / 'voc-cli' / 'voc-cli.exe'
            if path.exists():
                self.cli_path = tk.StringVar(value=str(path))
            else:
                self.cli_path = tk.StringVar(value="dist\\voc-cli\\voc-cli.exe")
        else:
            if which("voc-cli"):
                self.cli_path = tk.StringVar(value=which("voc-cli"))
            else:
                self.cli_path = tk.StringVar(value="dist/voc-cli")

        # GUI Elements
        self.cli_path_frame()
        self.make_options()
        self.make_buttons()
        self.cli_output()

    def cli_path_frame(self):
        cli_frame = tk.Frame(self.root)
        cli_frame.pack(padx=10, pady=10, fill=tk.X)

        tk.Label(cli_frame, text="VOC-CLI Path:").grid(row=0, column=0, sticky=tk.E)
        tk.Entry(cli_frame, textvariable=self.cli_path, width=50).grid(row=0, column=1, sticky=tk.W)
        tk.Button(cli_frame, text="Browse...", command=self.select_cli_path).grid(row=0, column=2, sticky=tk.W)

    def make_options(self):
        options_frame = tk.Frame(self.root)
        options_frame.pack(padx=10, pady=10, fill=tk.X)

        self.cache_var = tk.BooleanVar(value=True)
        self.smoothness_var = tk.IntVar(value=10)
        self.fft_var = tk.BooleanVar(value=False)
        self.y_offset_var = tk.DoubleVar(value=0.0)
        self.threshold_var = tk.DoubleVar(value=float('-inf'))

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
        tk.Button(buttons_frame, text="Help", command=lambda: webbrowser.open("https://github.com/TheRandomOwl/VOC-Detector/blob/main/README.md#voc-gui")).grid(row=2, column=0, padx=5)

    def cli_output(self):
        # Make the Text widget resizable by setting 'fill' to 'both' and 'expand' to True
        self.output_text = tk.Text(self.root, state="disabled")
        self.output_text.pack(pady=10, fill='both', expand=True)

    def run_cli(self, command, *args):
        """Run CLI command in a separate thread and update the text box in real-time."""
        def run():
            try:
                cli_path = self.cli_path.get()
                flags = [
                         "--cache" if self.cache_var.get() else "--no-cache",
                         "--smoothness", str(self.smoothness_var.get()),
                         "--fft" if self.fft_var.get() else "--no-fft",
                         "--y-offset", str(self.y_offset_var.get()),
                         "--threshold", str(self.threshold_var.get())
                        ]
                process = subprocess.Popen(
                    [cli_path, *flags, command, *args],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                err = False
                # Continuously read the output
                for line in process.stdout:
                    self.update_output(line)

                # Continuously read the error
                for line in process.stderr:
                    err = True
                    self.update_output(line)
                if err:
                    messagebox.showerror("Error", "An error occurred. Please check the output for more information.")

            except Exception as e:
                messagebox.showerror("Internal Error", str(e))

        # Start the CLI command in a separate thread
        threading.Thread(target=run).start()

    def update_output(self, text):
        """Update the text box with the given text."""
        self.output_text.config(state="normal")
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)  # Auto-scroll to the end
        self.output_text.config(state="disabled")

    def run_average(self):
        folder = filedialog.askdirectory(title="Select Data Folder")
        if folder:
            self.run_cli("average", folder)
            return
        messagebox.showinfo("Info", "Canceled operation.")

    def run_compare(self):
        folder_a = filedialog.askdirectory(title="Select First Run")
        folder_b = filedialog.askdirectory(title="Select Second Run")
        if folder_a and folder_b:
            self.run_cli("compare", folder_a, folder_b)
            return
        messagebox.showinfo("Info", "Canceled operation.")

    def run_export(self, single=True):
        folder = filedialog.askdirectory(title="Select Data Folder")
        if folder:
            save_path = filedialog.askdirectory(title="Select Save Directory")
            if save_path:
                save_as = "single" if single else "multi"
                self.run_cli("export", "--save-as", save_as, folder, save_path)
                return
        messagebox.showinfo("Info", "Canceled operation.")

    def run_plot(self):
        folder = filedialog.askdirectory(title="Select Data Folder")
        if folder:
            save_dir = filedialog.askdirectory(title="Select Save Directory")
            if save_dir:
                self.run_cli("plot", folder, save_dir)
                return
        messagebox.showinfo("Info", "Canceled operation.")

    def show_version_info(self):
        self.run_cli("--version")

    def select_cli_path(self):
        path = filedialog.askopenfilename(title="Select VOC-CLI Executable")
        if path:
            self.cli_path.set(path)

if __name__ == '__main__':
    root = tk.Tk()
    app = Gui(root)
    root.mainloop()
