import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import platform
from pathlib import Path
from shutil import which
import threading
import webbrowser

VER = '0.6.3'

class Gui:
    def __init__(self, root):
        self.root = root
        self.root.title(f"VOC GUI v{VER}")
        self.subprocess = None
        self.shut_down = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        # Allow the window to resize
        self.root.geometry("600x600")  # Set an initial size for the window

        # Automatically set the path to the CLI executable based on the platform
        if platform.system() == "Windows":
            path = (Path(__file__).parent/'..'/'..').resolve() / 'voc-cli' / 'voc-cli.exe'
            if path.exists():
                self.cli_path = tk.StringVar(value=str(path))
            else:
                path = Path("dist\\voc-cli\\voc-cli.exe")
                self.cli_path = tk.StringVar(value=path)
                if not path.exists():
                    messagebox.showerror("Error", "Could not find the voc-cli executable. Please specify the path manually.")
        else:
            if which("voc-cli"):
                self.cli_path = tk.StringVar(value=which("voc-cli"))
            else:
                path = Path("dist/voc-cli")
                self.cli_path = tk.StringVar(value=path)
                if not path.exists():
                    messagebox.showerror("Error", "Could not find the voc-cli executable. Please specify the path manually.")

        # GUI Elements
        self.cli_path_frame()
        self.make_options()
        self.make_buttons()
        self.cli_output()

    def close(self):
        """Handle the window close event."""
        self.shut_down = True  # Set the flag to indicate that the window is closing
        if self.subprocess is not None and self.subprocess.poll() is None:  # If process is running
            self.subprocess.terminate()  # Terminate the subprocess
        self.root.destroy()  # Destroy the Tkinter root window

    def cancel_process(self):
        """Cancel the current process."""
        if self.subprocess is not None and self.subprocess.poll() is None:
            self.subprocess.terminate()
            messagebox.showinfo("Info", "Process has been canceled.")
        else:
            messagebox.showinfo("Info", "No process is currently running.")

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

        button_width = 17  # Set the width of the buttons

        button_height = 2  # Set the height of the buttons

        tk.Button(buttons_frame, text="Plot Signals", command=self.run_plot, width=button_width, height=button_height).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(buttons_frame, text="Show Avg Signal", command=self.run_average, width=button_width, height=button_height).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(buttons_frame, text="Compare Runs", command=self.run_compare, width=button_width, height=button_height).grid(row=0, column=2, padx=5, pady=5)
        tk.Button(buttons_frame, text="Export Single CSV", command=lambda: self.run_export(single=True), width=button_width, height=button_height).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(buttons_frame, text="Export Multiple CSVs", command=lambda: self.run_export(single=False), width=button_width, height=button_height).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(buttons_frame, text="Export Average to CSV", command=lambda: self.run_export(avg=True), width=button_width, height=button_height).grid(row=1, column=2, padx=5, pady=5)
        tk.Button(buttons_frame, text="Show Version Info", command=self.show_version_info, width=button_width, height=button_height).grid(row=2, column=0, padx=5, pady=5)
        tk.Button(buttons_frame, text="Cancel Process", command=self.cancel_process, width=button_width, height=button_height).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(buttons_frame, text="Help", command=lambda: webbrowser.open("https://github.com/TheRandomOwl/VOC-Detector/tree/main#voc-gui"), width=button_width, height=button_height).grid(row=2, column=2, padx=5, pady=5)

    def cli_output(self):
        # Make the Text widget resizable by setting 'fill' to 'both' and 'expand' to True
        self.output_text = tk.Text(self.root, state="disabled")
        self.output_text.pack(pady=10, fill='both', expand=True)

    def run_cli(self, command, *args, notify=True):
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
                    universal_newlines=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
                )

                self.subprocess = process

                # Continuously read the output
                threading.Thread(target=read_output, args=(process.stdout,)).start()

                # Continuously read the error
                threading.Thread(target=read_output, args=(process.stderr,)).start()

                # notify when the process is done
                if notify:
                    threading.Thread(target=notify_completion).start()

            except Exception as e:
                messagebox.showerror("Internal Error", str(e))

        def read_output(output):
            try:
                for line in output:
                    if self.shut_down:
                        return
                    self.update_output(line)
            except ValueError:
                # IO stream is closed
                return
            except Exception as e:
                messagebox.showerror("Internal Error", str(e))

        def notify_completion():
            try:
                # Wait for the process to finish
                self.subprocess.communicate()
            except OSError:
                # Process is terminated
                return

            if self.subprocess.returncode == 0:
                messagebox.showinfo("Info", "Command has finished running.")
            elif not self.shut_down and self.subprocess.returncode > 0:
                messagebox.showerror("Error", "Something went wrong. Check the output for more information.")

        # Start the CLI command in a separate thread
        if self.subprocess is None or self.subprocess.poll() is not None:
            threading.Thread(target=run).start()

        else:
            response = messagebox.askyesno("Process Running",
                "Another process is already running. Do you want to cancel the current process and run a new process?")
            if response:
                self.subprocess.terminate()
                # Wait for the process to terminate
                try:
                    self.subprocess.communicate()
                except OSError:
                    # Process is terminated
                    pass

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
            self.run_cli("average", "--method", "all", folder)
            return
        messagebox.showinfo("Info", "Canceled operation.")

    def run_compare(self):
        folder_a = filedialog.askdirectory(title="Select First Run")
        folder_b = filedialog.askdirectory(title="Select Second Run")
        if folder_a and folder_b:
            self.run_cli("compare", "--method", "all",folder_a, folder_b)
            return
        messagebox.showinfo("Info", "Canceled operation.")

    def run_export(self, single=True, avg=False):
        folder = filedialog.askdirectory(title="Select Data Folder")
        if folder:
            save_path = filedialog.askdirectory(title="Select Save Directory")
            if save_path and not avg:
                save_as = "single" if single else "multi"
                self.run_cli("export", "--save-as", save_as, folder, save_path)
                return
            elif save_path and avg:
                self.run_cli("export", "--method", "avg", folder, save_path)
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
        self.run_cli("--version", notify=False)

    def select_cli_path(self):
        path = filedialog.askopenfilename(title="Select VOC-CLI Executable")
        if path:
            self.cli_path.set(path)

if __name__ == '__main__':
    root = tk.Tk()
    app = Gui(root)
    root.mainloop()
