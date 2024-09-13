'''
For: LLU Volatile Organic Compound Detector Signal Analysis

This program provides classes and functions for analyzing volatile organic compound (VOC) detector signals. It allows for loading, processing, plotting, and exporting of signal data.

Classes:
- Signal: Represents a single signal with attributes for units, x-values, y-values, and name. Provides methods for plotting, smoothing, calculating FFT, and exporting the signal.
- Run: Represents a collection of signals in a run. Provides methods for averaging signals, plotting signals, smoothing signals, and exporting signals.

Functions:
- mvavg: Calculates the moving average of an input array with a given window size.
- load: Loads a pickled object from a file.
- save: Saves a run object to a pickle file.
- export: Exports data to a file.
- plot_average_signals: Plots the average signal or FFT for two runs.
- corr_coef: Calculates the similarity between two runs.

Based on code by: Eli Haynal
Supervisor: Dr. Reinhard Schulte
For: LLU Volatile Organic Compound Detector Siganl Analysis
Version: 10:50 am 6/23/2023

Modified by: Nathan Perry and Nathan Fisher

'''

__version__ = '4.6.0'

# These statements import the libraries needed for the code to run
import csv  # A library for reading and writing csv files
import multiprocessing  # A library for parallel processing
from pathlib import Path  # A library for handling file paths
import pickle  # A library for saving data in a python-readable format
import sys  # A library for interacting with the system
import warnings  # A library for handling warnings

import matplotlib.pyplot as plt  # A library for generating plots
import numpy as np  # A library with useful data storage structures and mathematical operations
from scipy.integrate import trapezoid  # A library for numerical integration
from tqdm import tqdm  # A library for progress bars

VER = __version__

METRIC = {
    '(us)': 1e-6,
    '(ms)': 1e-3,
    '(s)': 1
}

WINDOW_SIZE = 10

def _mvavg(x, y, window_size):
    """
    Calculate the moving average of an input array 'y' with a given window size.
    Parameters:
        x (array-like): The input array of x-values.
        y (array-like): The input array of y-values.
        window_size (int): The size of the moving average window. use 'default' for default window size.
    Returns:
        tuple: A tuple containing the aligned x-values and the moving average.
    Raises:
        ValueError: If the window size is less than 1 or greater than the length of the input array.
        TypeError: If the window size is not an integer.
    """

    # Convert input arrays to numpy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if window_size == 'default':
        window_size = WINDOW_SIZE

    if type(window_size) is not int:
        raise TypeError("Window size must be an integer.")

    if window_size < 1 or window_size > len(y):
        raise ValueError("Window size must be between 1 and the length of the input array.")

    # Calculate moving average
    averages = np.convolve(y, np.ones(window_size)/window_size, mode='valid')

    # Align x array with the moving average
    start_index = window_size - 1
    aligned_x = np.asarray(x[start_index : start_index+len(averages)])
    aligned_x -= (aligned_x[0] - x[0]) / 2

    return aligned_x, averages

class Signal():
    """
    Class representing a signal (one file).
    Attributes:
        units (list): List of units of the signal. Index 0 is the x-axis unit, index 1 is the y-axis unit.
        x (ndarray): Array of x values for the signal.
        y (ndarray): Array of y values for the signal.
        name (str): Name of the signal.
        xf (ndarray): Array of x values for the FFT of the signal.
        yf (ndarray): Array of y values for the FFT of the signal.
    """

    def __init__(self, infile, name = None, baseline_shift = 0, smooth_window = 0, fft = False, normalize = False):
        """
        Initializes an instance of the Signal class.
        Parameters:
            infile (str): The path to the input file.
            name (str, optional): The name of the object. If not provided, the name will be extracted from the input file path.
            baseline_shift (float, optional): The amount to add to the y values of the signal. Default is 0.
            smooth_window (int, optional): The size of the window for smoothing the signal. Default is 0 (no smoothing).
            fft (bool, optional): Perform FFT on the signal if True. Default is False.
            normalize (bool, optional): Normalize the x-axis values of the signal to start at zero if True. Default is False.
        Returns:
        None
        """

        infile = Path(infile)

        # Open the specified input file and read all of its lines into a list
        with open(infile) as f:
            reader = csv.reader(f,delimiter='\t')
            data = list(reader)

            # Extract the units of the signal from the header
            self.units = tuple(data[1])

            # Eliminate the three header lines
            data = data[3:]

            # Create a numpy array of the x values for the signal
            self.x = np.asarray([float(elm[0]) for elm in data])

            # Create a numpy array of y values for the signal and add baseline shift
            self.y = np.asarray([float(elm[1])+baseline_shift for elm in data])

        # Set a name for the object so that it can be identified
        self.name = str(infile.stem) if name is None else name

        # Smooth the signal if specified
        self.smooth(smooth_window)

        self.y_offset = baseline_shift

        # Create placeholders for the FFT results
        self.xf = np.array([])
        self.yf = np.array([])

        if fft:
            self.fft()

        if normalize:
            self.normalize_x()

    def __repr__(self):
        return self.name

    def normalize_x(self):
        """
        Normalize the x-axis values of the signal.
        Returns:
            None
        """
        self.x = self.x - self.x[0]

    def plot(self,folder,fft = False, ymin = None, ymax = None):
        """
        Generate and save a plot of the signal or its FFT.
        Parameters:
            folder (str): Directory to save the plot image. Created if it doesn't exist.
            fft (bool, optional): Plot FFT if True, time-domain signal if False. Default is False.
            ymin (float, optional): Minimum value for y-axis range. Default is -400.
            ymax (float, optional): Maximum value for y-axis range. Default is -150.
        """

        if ymin is None:
            ymin = -400
        if ymax is None:
            ymax = -150

        if fft:
            plt.plot(self.xf, np.abs(self.yf))
            plt.title('FFT: ' + self.name)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
        else:
            plt.ylim(ymin + self.y_offset, ymax + self.y_offset)
            plt.title(self.name)
            plt.xlabel('Time ' + self.units[0])
            plt.ylabel('Amplitude ' + self.units[1])
            plt.plot(self.x,self.y)

        # Create the output folder if it does not already exist
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        # Generate the filename and convert it into a filepath
        filename = self.name + '.png'
        path = folder_path / filename

        # Save the figure and clear the plotting tool's buffer
        plt.savefig(path)
        plt.clf()

    # Checks if there exists no peak and returns true if there isn't a peak else return false
    def is_empty(self, threshold):
        """
        Check if there exists no peak in the signal.
        Parameters:
            threshold (float): Threshold to determine peak existence.
        Returns:
            bool: True if there is no peak above the threshold, False otherwise.
        """

        return self.y.max() <= threshold

    def smooth(self, window_size = None):
        """
        Smooth the signal using a moving average
        Parameters:
            window_size (int, optional): Window size for smoothing. Default is the value of WINDOW_SIZE.
        Returns:
            None
        """
        if window_size == 0:
            return

        if window_size is None:
            window_size = 'default'

        self.x, self.y = _mvavg(self.x, self.y, window_size)

    def fft(self, units = None):
        """
        Calculate the Fast-Fourier transform of the signal. Use np.abs() on yf to get magnitude.
        Parameters:
            units (optional): The units of the time axis of the signal,
            such as "(us)", "(ms)" or "(s)". Default is the value of self.units[0].
        Returns:
            None
        """
        # remove dc component
        y = self.y - np.mean(self.y)

        # Convert signal time axis to seconds
        if units is None:
            x = np.asarray(self.x) * METRIC[self.units[0]]
        else:
            x = np.asarray(self.x) * units

        # Sample spacing
        dt = np.mean(np.diff(x))
        n = len(x)

        # Perform the FFT. yf is a complex number array
        self.yf = np.fft.rfft(y)
        self.xf = np.fft.rfftfreq(n, dt)

    def show_signal(self, fft = False):
        """
        Show a graph of the signal.
        Parameters:
            fft (bool, optional): Plot FFT if True, time-domain signal if False. Default is False.
        Returns:
            None
        """
        # Plot the frequency vs. magnitude or time vs. amplitude
        plt.plot(self.xf if fft else self.x, np.abs(self.yf) if fft else self.y)

        # Label the axes
        plt.xlabel('Frequency (Hz)' if fft else 'Time ' + self.units[0])
        plt.ylabel('Magnitude' if fft else 'Amplitude ' + self.units[1])

        # Add a title
        plt.title('FFT of the Signal' if fft else 'Signal')

        # Display the plot
        plt.show()

        # Clear the plot
        plt.clf()

    def export(self, filepath, fft = False):
        """
        Export the signal to a file.
        Parameters:
            filepath (str): The path to the output file.
        Returns:
            None
        """
        if fft:
            _export(filepath, self.xf, np.abs(self.yf), header=['(Hz)', 'Units'])
        else:
            _export(filepath, self.x, self.y, header=[self.units[0], self.units[1]])

class Run():
    """
    Class representing a run of signals (all files).
    Attributes:
        name (str): Name of the run folder.
        signals (list): List of signal objects in the run.
        smoothed (bool): True if the signals are smoothed, False otherwise.
        smoothness (int): The size of the window for smoothing the signals.
        units (list): List of units of the signals in the run.
    """

    def __init__(self, foldername, y_offset = 0, cache = True, smoothness = 'default', fft = True, normalize = False):
        """
        Initialize a run instance from a specified folder of .txt files.
        Parameters:
            foldername (str): Path to the folder containing .txt files.
            y_offset (float, optional): The amount to shift the y values of the signals. Default is 0.
            cache (bool, optional): If True, save the run object to cache. Default is True.
            smoothness (int, optional): The size of the window for smoothing the signals. Default is 'default'.
            fft (bool, optional): Perform FFT on the signals. Default is True.
            normalize (bool, optional): Normalize the x-axis values of the signals to start a zero. Default is False.
        Returns:
            None

        """

        # Set the version of the program
        self.version = VER

        # True if signals are smoothed
        self.smoothed = smoothness == 'default' or smoothness > 0
        self.smoothness = smoothness

        foldername = Path(foldername)
        self.name = str(foldername.name)
        self.path = foldername
        self.y_offset = y_offset

        if fft:
            self.fft_check = True
        else:
            self.fft_check = False

        if normalize:
            self.norm = True
        else:
            self.norm = False

        try:
            if cache:
                # Try to load the cache
                print(f"Trying to load cache from {self.path.with_suffix('.pickle')}")
                run_cache = _load(self.path.with_suffix('.pickle'))

            # Check if the cache is valid before loading
            if cache and self.__validate(run_cache):
                self.signals = run_cache.signals
                self.units = run_cache.units
                print("Loaded run from cache")
                return
            if self.version != run_cache.version:
                print(f"Cache version mismatch. Expected {self.version}, got {run_cache.version}")
        except FileNotFoundError:
            print("Cache not found")
        except UnboundLocalError:
            # Ignore if run_cache is not defined due to cache being False
            pass
        except Exception as e:
            warnings.warn(f"Unable to load cache: {e}")


        # Get the list of files to be processed and keep files that end with '.txt' filter out hidden files
        files = [file for file in foldername.iterdir() if file.suffix == '.txt' and not file.name.startswith('.')]

        # Create a pool of worker processes
        with multiprocessing.Pool() as pool:
            # Use pool.map to parallelize the loading of signals
            results = list(tqdm(pool.imap(self.load_signal, [(f, y_offset, fft, smoothness, normalize) for f in files]), total=len(files), desc="Loading files", file=sys.stdout))

        # Filter out any None results (in case of errors)
        self.signals = [res for res in results if res is not None]

        if len(self.signals) == 0:
            raise RuntimeError("No signals could be found. Make sure the files exist and are in the correct format.")

        # Get units from signals
        self.units = self.get(0).units


        # Try to save signals to cache
        try:
            if cache:
                _save(self)
                print(f"Saved cache to {self.path.with_suffix('.pickle')}")
        except Exception as e:
            warnings.warn(f"Unable to save cache: {e}")

    @staticmethod
    def load_signal(args):
        """
        Load a signal from a file.

        Parameters:
        - args (tuple): A tuple containing the following elements:
            - f (Path): The path to the file.
            - offset (float): The amount to shift the y values of the signals.
            - fft (bool): Perform FFT on the signals.
            - smoothness (int): The size of the window for smoothing the signals.
            - norm (bool): Normalize the x-axis values of the signals.

        Returns:
        - Signal or None: The loaded signal if successful, or None if an error occurred.
        """
        f, offset, fft, smoothness, norm = args
        try:
            return Signal(f, baseline_shift=offset, fft=fft, smooth_window=smoothness, normalize=norm)
        except ValueError:
            return None

    def normalize_x(self):
        """
        Normalize the x-axis values of the signals.
        Returns:
            None
        """
        for s in self.signals:
            s.normalize_x()

    def __validate(self, cache):
        """
        Validate the cache object.

        Parameters:
            run_cache (Run): The cached run object.

        Returns:
            bool: True if the cache is valid, False otherwise.
        """
        return (self.version == cache.version
                and self.path == cache.path
                and self.smoothness == cache.smoothness
                and self.y_offset == cache.y_offset
                and self.fft_check == cache.fft_check
                and self.norm == cache.norm)

    def get(self, index):
        """
        Returns the signal at the specified index.

        Parameters:
            index (int): The index of the signal to retrieve.

        Returns:
            object: The signal at the specified index.
        """
        return self.signals[index]

    # A function defining how a run object is represented when printed.
    def __repr__(self):
        return self.name

    def plot(self,folder,fft = False, ymin = None, ymax = None):
        """
        Plot every signal in the run to a specified folder.
        Parameters:
            folder (str): Directory to save the plot images.
            fft (bool, optional): Plot FFT if True, time-domain signal if False. Default is False.
        Returns:
            None
        """
        with multiprocessing.Pool() as pool:
            # Use pool.map to parallelize the plotting of signals and us tqdm to show progress
            list(tqdm(pool.imap(self.plot_signals, [(s, folder, fft, ymin, ymax) for s in self.signals]), total=len(self.signals), desc="Plotting signals", file=sys.stdout))

    @staticmethod
    def plot_signals(args):
        """
        Plot signals.

        Args:
            args (tuple): A tuple containing the following elements:
            - s: The signal object.
            - folder: The folder path where the plot will be saved.
            - plot_fft (bool): Whether to plot the FFT of the signal or not.
        """
        s, folder, plot_fft, ymin, ymax = args
        s.plot(folder, fft=plot_fft, ymin=ymin, ymax=ymax)

    def smooth(self, smoothness = None):
        """
        Smooth each signal in the run with a moving average.
        Parameters:
            smoothness (int, optional): The size of the window for smoothing the signals. Default is None.
        Returns:
            None
        """

        with multiprocessing.Pool() as pool:
            # Use pool.map to parallelize the smoothing of signals and us tqdm to show progress
            self.signals = list(tqdm(pool.imap(self.smooth_signals, [(s, smoothness) for s in self.signals]),
                            total=len(self.signals), desc="Smoothing signals", file=sys.stdout))

    @staticmethod
    def smooth_signals(args):
        """
        Smooths the given signals.

        Args:
            args (tuple): A tuple containing the signal object and the smoothness value.

        Returns:
            Signal: The smoothed signal object.
        """
        s, smoothness = args
        s.smooth(smoothness)
        return s

    def export(self, directory, fft = False):
        """
        Export every signal in the run to a specified folder.
        Parameters:
            directory (str): Directory to save the output files.
            fft (bool, optional): Export FFT if True, time-domain signal if False. Default is False.
        Returns:
            None
        """
        with multiprocessing.Pool() as pool:
            # Use pool.map to parallelize the exporting of signals
            list(tqdm(pool.imap(self.export_signals, [(s, directory, fft) for s in self.signals]), total=len(self.signals), desc="Exporting signals", file=sys.stdout))

    @staticmethod
    def export_signals(args):
        """
        Export signals to a CSV file.

        Args:
            args (tuple): A tuple containing the following parameters:
                - s (Signal): The signal to export.
                - folder (str): The folder path where the CSV file will be saved.
                - fft (bool): Flag indicating whether to apply FFT before exporting.

        Returns:
            Signal: The exported signal.

        """
        s, folder, fft = args
        filepath = Path(folder) / (s.name + '.csv')
        s.export(filepath, fft)
        return s

    def export_all(self, filepath, fft = False, unique = True):
        """
        Export all signals in the run to a single file.
        Parameters:
            filepath (str): The path to the output file. If a directory is provided, the file will be named after the run.
            fft (bool, optional): Export FFT if True, time-domain signal if False. Default is False.
            label (bool, optional): If True, include a header with the units of the signals. Default is False.
            unique (bool, optional): If True, include numerical identifiers for each signal in the header. Default is True.
        Returns:
            None
        """
        filepath = Path(filepath)
        if filepath.is_dir():
            filepath = filepath / (self.name + '.csv')

        data = []
        header = []

        for i, s in enumerate(self.signals):
            if fft:
                header.append(f"(Hz) #{i}" if unique else "(Hz)")
                header.append(f"Units #{i}" if unique else "Units")

                data.append(s.xf)
                data.append(np.abs(s.yf))
            else:
                data.append(s.x)
                data.append(s.y)

                header.append(f"{s.units[0]} #{i}" if unique else s.units[0])
                header.append(f"{s.units[1]} #{i}" if unique else s.units[1])


        _export(filepath, *data, header=header)

    def export_avg(self, filepath, fft = False):
        """
        Export the average signal or FFT for the run to a file.
        Parameters:
            filepath (str): The path to the output file. If a directory is provided, the file will be named after the run.
            fft (bool, optional): Export FFT if True, time-domain signal if False. Default is False.
        Returns:
            None
        """
        filepath = Path(filepath)
        if filepath.is_dir():
            filepath = filepath / (self.name + '_avg.csv')

        x, avg_y = self.avg_signal(fft)
        if fft:
            _export(filepath, x, np.abs(avg_y), header=['(Hz)', 'Units'])
        else:
            _export(filepath, x, avg_y, header=[self.units[0], self.units[1]])

    def fft(self, units = None):
        """
        Calculate the FFT for every signal in the run.
        Parameters:
            units (optional): The units of the time axis of the signal,
            such as "(um)", "(ms)" or "(s)" . Default is the value of self.units[0].
        Returns:
            None
        """
        for s in self.signals:
            s.fft(units)

    def clean_empty(self, threshold):
        """
        Remove signals without peaks from the run.
        Parameters:
            threshold (float): The minimum peak height for a signal to be included.
        Returns:
            None
        """
        new = []
        for signal in self.signals:
            if not signal.is_empty(threshold):
                new.append(signal)
        if len(new) == 0:
            warnings.warn("No signals remain after cleaning, try a lower threshold.")
        self.signals = new

    def avg_signal(self, fft = False):
        """
        Calculate the average signal or FFT for the run.
        Parameters:
            fft (bool): If True, calculate the average FFT. If False, calculate the average time-domain signal.
        Returns:
            tuple: Arrays of x-values and average y-values.
        """
        # Extract the y or yf arrays
        y_arrays = [s.yf if fft else s.y for s in self.signals]

        # Stack the arrays along a new axis and compute the mean
        avg_y = np.mean(np.stack(y_arrays), axis=0)

        # Extract the corresponding x or xf array (assuming they are the same for all signals)
        x = self.signals[0].xf if fft else self.signals[0].x

        return x, avg_y

    def avg_area(self):
        """
        Calculate the average area under the curve for the run.
        Returns:
            float: The average area under the curve.
        """
        x, avg_y = self.avg_signal()
        area = trapezoid(avg_y, x)
        return area

    def avg_max(self):
        """
        Calculate the average maximum value for the run.
        Returns:
            float: The average maximum value.
        """
        _, avg_y = self.avg_signal()
        return avg_y.max()

    def avg_voltage(self):
        """
        Calculate the average voltage for the run.
        Returns:
            float: The average voltage.
        """
        _, voltage = self.avg_signal()
        return np.mean(voltage)

    def plot_average_signal(
            self,
            filepath = None,
            fft = False,
            ybottom = None,
            ytop = None,
            xleft = None,
            xright = None
        ):
        """
        Plot and show the average signal or FFT for the run.
        Parameters:
            filepath (str, optional): Directory to save the plot image. Default is None.
            fft (bool, optional): Plot FFT if True, time-domain signal if False. Default is False.
            ybottom (optional): Bottom limit for y-axis.
            ytop (optional): Top limit for y-axis.
            xleft (optional): Left limit for x-axis.
            xright (optional): Right limit for x-axis.
        Returns:
            None
        """
        x, avg_y = self.avg_signal(fft)
        plt.plot(x, np.abs(avg_y) if fft else avg_y)
        plt.title('Average FFT: ' + self.name if fft else 'Average Signal: ' + self.name)
        plt.xlabel('Frequency (Hz)' if fft else 'Time ' + self.units[0])
        plt.ylabel('Magnitude' if fft else 'Amplitude ' + self.units[1])
        plt.ylim(bottom=ybottom, top=ytop)
        plt.xlim(left=xleft, right=xright)

        if filepath is None:
            plt.show()
        else:
            plt.savefig(Path(filepath) / (self.name + '_avg_signals.png'))

        # Clear the plot
        plt.clf()

def plot_average_signals(run1, run2, filepath = None, fft = False):
    """
    Plot the average signal or FFT for two runs. Useful for subjectively identifying
    typical signal differences between two treatments. If filepath is provided,
    the plot will be saved to a file else, the plot will be displayed.
    Parameters:
        run1 (run): The first run.
        run2 (run): The second run.
        filepath (str, optional): Directory to save the plot image. Default is None.
        fft (bool, optional): Plot FFT if True, time-domain signal if False. Default is False.
        show (bool, optional): If True, display the plot. If False, save the plot. Default is False.
    Returns:
        None
    """
    x1, y1 = run1.avg_signal(fft)

    x2, y2 = run2.avg_signal(fft)

    #Clear the plotting tool
    plt.clf()

    # Plot both average signals
    if fft:
        plt.plot(x1, np.abs(y1),label=run1.name)
        plt.plot(x2, np.abs(y2),label=run2.name)
    else:
        plt.plot(x1,y1,'o',markersize=3,label=run1.name)
        plt.plot(x2,y2,'o',markersize=3,label=run2.name)

    # Add a title and axis labels to the plot
    plt.title('Average signals')
    plt.xlabel('Frequency (Hz)' if fft else 'Time ' + run1.units[0])
    plt.ylabel('Magnitude' if fft else 'Amplitude ' + run1.units[1])

    # Add a legend to the plot
    plt.legend()

    # Save the plot at the specified location with an auto-generated name
    if filepath is None:
        plt.show()
    else:
        plt.savefig(Path(filepath) / (run1.name + '-' + run2.name + '_avg.png'))

    # Clear the plot
    plt.clf()

def corr_coef(run1, run2):
    """
    Calculate the similarity between two runs.
    Parameters:
        run1 (run): The first run.
        run2 (run): The second run.
    Returns:
        float: The correlation coefficent between the two runs.
    """
    _, y1 = run1.avg_signal()
    _, y2 = run2.avg_signal()
    return np.corrcoef(y1, y2)[0, 1]

def _export(filepath, *lists, header = None):
    """
    Export data to a file.

    Parameters:
        filepath (str): The path to the output file.
        *lists (array-like): The lists of values to export. Each list will be a column in the CSV.
        header (list, optional): The header to write to the output file.

    Returns:
        None
    """
    filepath = Path(filepath)
    # Create the output directory if it does not already exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Find the maximum length of the lists to handle cases where lists have different lengths
    max_length = max(len(lst) for lst in lists)

    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header if provided
        if header is not None:
            writer.writerow(header)

        # Write the lists row by row
        for i in range(max_length):
            row = [lst[i] if i < len(lst) else '' for lst in lists]
            writer.writerow(row)

def _save(run):
    """
    Save the run object to a pickle file.

    Parameters:
    run (object): The run object to be saved.

    Returns:
    None
    """
    with open(run.path.with_suffix('.pickle'),'wb') as f:
        pickle.dump(run,f)

def _load(file):
    """
    Load a pickled object from a file.

    Parameters:
    file (str): The path to the file containing the pickled object.

    Returns:
    object: The unpickled object.

    Raises:
    FileNotFoundError: If the file does not exist.
    """
    with open(file,'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    print(VER)
