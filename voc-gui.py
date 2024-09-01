import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess

# Function to run the CLI command
def run_cli_command(command, *args):
    try:
        # Get the executable path from the entry field
        cli_path = cli_path_var.get()
        result = subprocess.run(
            [cli_path, command] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Display output
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, result.stdout)
        if result.stderr:
            messagebox.showerror("Error", result.stderr)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Command-specific functions
def run_average():
    folder = filedialog.askdirectory(title="Select Data Folder")
    if folder:
        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if save_dir:
            run_cli_command("average", "--save-dir", save_dir, folder)

def run_compare():
    folder_a = filedialog.askdirectory(title="Select Folder A")
    folder_b = filedialog.askdirectory(title="Select Folder B")
    if folder_a and folder_b:
        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if save_dir:
            run_cli_command("compare", "--save-dir", save_dir, folder_a, folder_b)

def run_export(single=True):
    folder = filedialog.askdirectory(title="Select Data Folder")
    if folder:
        save_path = filedialog.askdirectory(title="Select Save Directory")
        if save_path:
            save_as = "single" if single else "multi"
            run_cli_command("export", "--save-as", save_as, folder, save_path)

def run_plot():
    folder = filedialog.askdirectory(title="Select Data Folder")
    if folder:
        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if save_dir:
            run_cli_command("plot", folder, save_dir)

def show_version_info():
    run_cli_command("--version")

def select_cli_path():
    path = filedialog.askopenfilename(title="Select VOC-CLI Executable")
    if path:
        cli_path_var.set(path)

def create_gui():
    # Create the main window
    root = tk.Tk()
    root.title("VOC GUI")

    # CLI Path Selection
    global cli_path_var
    cli_path_var = tk.StringVar(value="voc-cli.py")

    cli_frame = tk.Frame(root)
    cli_frame.pack(padx=10, pady=10, fill=tk.X)

    tk.Label(cli_frame, text="VOC-CLI Path:").grid(row=0, column=0, sticky=tk.E)
    tk.Entry(cli_frame, textvariable=cli_path_var, width=50).grid(row=0, column=1, sticky=tk.W)
    tk.Button(cli_frame, text="Browse...", command=select_cli_path).grid(row=0, column=2, sticky=tk.W)

    # Cache, Smoothness, FFT, Y-Offset, Threshold Options
    cache_var = tk.BooleanVar(value=True)
    smoothness_var = tk.IntVar(value=10)
    fft_var = tk.BooleanVar(value=False)
    y_offset_var = tk.DoubleVar(value=0.0)
    threshold_var = tk.DoubleVar(value=0.0)

    # Options Frame
    options_frame = tk.Frame(root)
    options_frame.pack(padx=10, pady=10, fill=tk.X)

    tk.Checkbutton(options_frame, text="Cache", variable=cache_var).grid(row=0, column=0, sticky=tk.W)
    tk.Label(options_frame, text="Smoothness:").grid(row=0, column=1, sticky=tk.E)
    tk.Entry(options_frame, textvariable=smoothness_var).grid(row=0, column=2, sticky=tk.W)

    tk.Checkbutton(options_frame, text="FFT", variable=fft_var).grid(row=1, column=0, sticky=tk.W)
    tk.Label(options_frame, text="Y-Offset:").grid(row=1, column=1, sticky=tk.E)
    tk.Entry(options_frame, textvariable=y_offset_var).grid(row=1, column=2, sticky=tk.W)

    tk.Label(options_frame, text="Threshold:").grid(row=2, column=1, sticky=tk.E)
    tk.Entry(options_frame, textvariable=threshold_var).grid(row=2, column=2, sticky=tk.W)

    # Command Buttons Frame
    buttons_frame = tk.Frame(root)
    buttons_frame.pack(padx=10, pady=10, fill=tk.X)

    tk.Button(buttons_frame, text="Plot Signals", command=run_plot).grid(row=0, column=0, padx=5)
    tk.Button(buttons_frame, text="Show Avg Signal", command=run_average).grid(row=0, column=1, padx=5)
    tk.Button(buttons_frame, text="Compare Runs", command=run_compare).grid(row=0, column=2, padx=5)
    tk.Button(buttons_frame, text="Export to One CSV", command=lambda: run_export(single=True)).grid(row=1, column=0, padx=5)
    tk.Button(buttons_frame, text="Export to Multiple CSVs", command=lambda: run_export(single=False)).grid(row=1, column=1, padx=5)
    tk.Button(buttons_frame, text="Show Version Info", command=show_version_info).grid(row=1, column=2, padx=5)

    # Output Text Area
    global output_text
    output_text = tk.Text(root, height=15, width=60)
    output_text.pack(pady=10)

    root.mainloop()

if __name__ == '__main__':
    create_gui()
