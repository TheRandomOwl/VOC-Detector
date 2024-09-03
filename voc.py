'''
Code by: Eli Haynal
Supervisor: Dr. Reinhard Schulte
For: LLU Volatile Organic Compound Detector Siganl Analysis
Version: 10:50 am 6/23/2023

Modified by: Nathan Perry and Nathan Fisher
'''

#These statements import the libraries needed for the code to run
import csv  # A library for reading and writing csv files
import matplotlib.pyplot as plt  # A library for generating plots
import multiprocessing  # A library for parallel processing
import numpy as np  # A library with useful data storage structures and mathematical operations
import os  # A library for loading and writing to the filesystem more easily
from pathlib import Path # A library for handling file paths
import pickle  # A library for saving data in a python-readable format
from scipy.integrate import trapezoid  # A library for numerical integration
import sys  # A library for interacting with the system
from tqdm import tqdm  # A library for progress bars

VER = '4.2.3'

METRIC = {
    '(us)': 1e-6,
    '(ms)': 1e-3,
    '(s)': 1
}

def mvavg(x, y, window_size):
    """
    Calculate the moving average of an input array 'y' with a given window size.
    Parameters:
        x (array-like): The input array of x-values.
        y (array-like): The input array of y-values.
        window_size (int): The size of the moving average window.
    Returns:
        aligned_x (ndarray): The x-values aligned with the moving average.
        averages (ndarray): The calculated moving averages.
    Raises:
        ValueError: If the window size is less than 1 or greater than the length of the input array.
    """
    if window_size < 1 or window_size > len(y):
        raise ValueError("Window size must be between 1 and the length of the input array.")

    # Calculate moving average
    averages = np.convolve(y, np.ones(window_size)/window_size, mode='valid')

    # Align x with the moving average
    start_index = window_size - 1
    aligned_x = np.asarray(x[start_index: start_index + len(averages)])
    aligned_x -= (aligned_x[0] - x[0]) / 2

    return aligned_x, averages

class Signal():
    """
    Class representing a signal.
    Attributes:
        units (list): List of units of the signal.
        x (ndarray): Array of x values for the signal.
        y (ndarray): Array of y values for the signal.
        name (str): Name of the signal.
    """

    #The function initiating each class instance from a specified .txt file
    def __init__(self, infile, name = None, baseline_shift = 0, smooth_window=0):
        """
        Initializes an instance of the Signal class.
        Parameters:
            infile (str): The path to the input file.
            name (str, optional): The name of the object. If not provided, the name will be extracted from the input file path.
            baseline_shift (float, optional): The amount to shift the y values of the signal. Default is 0.
            smooth_window (int, optional): The size of the window for smoothing the signal. Default is 0 (no smoothing).
        Returns:
        None
        """

        #Open the specified input file and read all of its lines into a list
        with open(infile) as f:
            reader = csv.reader(f,delimiter='\t')
            data = [row for row in reader]

            #Extract the units of the signal from the header
            self.units = tuple(data[1])

            #Eliminate the three header lines
            data = data[3:]

            #Create a list of the x values for the signal
            self.x = np.asarray([float(elm[0]) for elm in data])

            #Create a list of y values for the signal, moving them to zero the signal
            self.y = np.asarray([float(elm[1])+baseline_shift for elm in data])

        #Set a name for the object so that it can be identified
        self.name = str(Path(infile).stem) if name is None else name

        # Smooth the signal if specified
        self.smooth(smooth_window)

        self.y_offset = baseline_shift

    def plot(self,folder,fft = False):
        """
        Generate and save a plot of the signal or its FFT.
        Parameters:
            folder (str): Directory to save the plot image. Created if it doesn't exist.
            fft (bool, optional): Plot FFT if True, time-domain signal if False. Default is False.
        """
        if fft:
            plt.plot(self.xf, np.abs(self.yf))
            plt.title('FFT: ' + self.name)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude ' + self.units[1])
        else:
            plt.ylim(-400 + self.y_offset,-150 + self.y_offset)
            plt.title(self.name)
            plt.xlabel('Time ' + self.units[0])
            plt.ylabel('Amplitude ' + self.units[1])
            plt.plot(self.x,self.y)

        #Create the output folder if it does not already exist
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        #Generate the filename
        #and properly converting it into a filepath the computer will understand
        filename = self.name + '.png'
        path = folder_path / filename

        #Save the figure and clear the plotting tool's buffer
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
        Smooth the signal using a moving average and recalculate FFT and signal statistics.
        Parameters:
            window_size (int, optional): Window size for smoothing. Default is 10.
        Returns:
            None
        """
        if window_size == 0:
            return
        elif window_size == None:
            window_size = 10

        self.x, self.y = mvavg(self.x, self.y, window_size)
        self.fft()

    #Calculate the Fast-Fourier transform of the signal
    def fft(self, metric_prefix = None):
        """
        Calculate the Fast-Fourier transform of the signal.
        Parameters:
            metric_prefix (optional): The units of the time axis of the signal,
            such as "(um)", "(ms)" or "(s)". Default is the value of self.units[0].
        Returns:
            None
        """
        # remove dc component
        y = self.y - np.mean(self.y)

        # Convert signal time axis to seconds
        if metric_prefix == None:
            x = np.asarray(self.x) * METRIC[self.units[0]]
        else:
            x = np.asarray(self.x) * metric_prefix

        # Sample spacing
        dt = np.mean(np.diff(x))
        n = len(x)

        self.yf = np.fft.rfft(y)
        self.xf = np.fft.rfftfreq(n, dt)

    # Plot the magnitude of the FFT results
    def show_signal(self, fft=False):
        """
        Show a graph of the signal.
        Parameters:
            fft (bool, optional): Plot FFT if True, time-domain signal if False. Default is False.
        Returns:
            None
        """
        # Plot the frequency vs. magnitude
        plt.plot(self.xf if fft else self.x, np.abs(self.yf) if fft else self.y)

        # Label the axes
        plt.xlabel('Frequency (Hz)' if fft else 'Time ' + self.units[0])
        plt.ylabel('Magnitude' if fft else 'Amplitude ' + self.units[1])

        # Add a title
        plt.title('FFT of the Signal' if fft else 'Signal')

        # Display the plot
        plt.show()

    def export(self, filepath, fft = False):
        """
        Export the signal to a file.
        Parameters:
            filepath (str): The path to the output file.
        Returns:
            None
        """
        if fft:
            export(filepath, self.xf, np.abs(self.yf), header=['(Hz)', 'Units'])
        else:
            export(filepath, self.x, self.y, header=[self.units[0], self.units[1]])

class Run():
    """
    Class representing a run of signals.
    Attributes:
        name (str): Name of the run.
        signals (list): List of signal objects in the run.
        smoothed (bool): True if the signals are smoothed, False otherwise.
        smoothness (int): The size of the window for smoothing the signals.
        units (list): List of units of the signals in the run.
    """

    def __init__(self, foldername, y_offset = 0, cache = True, smoothness = 'default'):
        """
        Initialize a run instance from a specified folder of .txt files.
        Parameters:
            foldername (str): Path to the folder containing .txt files.
            y_offset (float, optional): The amount to shift the y values of the signals. Default is 0.
            cache (bool, optional): If True, save the run object to cache. Default is True.
            smoothness (int, optional): The size of the window for smoothing the signals. Default is 'default'.
        Returns:
            None

        """

        self.version = VER

        # True if signals are smoothed
        self.smoothed = smoothness == 'default' or smoothness > 0
        self.smoothness = smoothness

        self.name = os.path.split(foldername)[1]
        self.path = str(Path(foldername))
        self.y_offset = y_offset

        try:
            if cache:
                print(f"Trying to load cache from {self.path + '.pickle'}")
                run_cache = load(self.path + '.pickle')
            if cache and self.version == run_cache.version and self.path == run_cache.path and self.smoothness == run_cache.smoothness and self.y_offset == run_cache.y_offset:
                self.signals = run_cache.signals
                self.units = run_cache.units
                if self.smoothed and not run_cache.smoothed:
                    self.smooth(smoothness)
                    save(self)
                    print("Saved run to cache")
                print("Loaded run from cache")
                return
            elif self.version != run_cache.version:
                print(f"Cache version mismatch. Expected {self.version}, got {run_cache.version}")
        except FileNotFoundError:
            print("Cache not found")
        except UnboundLocalError:
            # Ignore if run_cache is not defined due to cache being False
            pass
        except:
            print("Unable to load cache")


        # Get the list of files to be processed and keep files that end with '.txt' filter out hidden files
        files = [os.path.join(foldername, filename) for filename in os.listdir(foldername) if filename[0] != '.' and filename[-4:] == '.txt']

        # Create a pool of worker processes
        with multiprocessing.Pool() as pool:
            # Use pool.map to parallelize the loading of signals
            results = list(tqdm(pool.imap(self.load_signal, [(f, y_offset) for f in files]), total=len(files), desc="Loading files", file=sys.stdout))

        # Filter out any None results (in case of errors)
        self.signals = [res for res in results if res is not None]

        if len(self.signals) == 0:
            raise ValueError("No signals could be loaded")

        # Get units from signals
        self.units = self.signals[0].units

        # Smooth the signals
        if self.smoothed:
            self.smooth(smoothness)

        # Try to save signals to cache
        try:
            if cache:
                save(self)
                print(f"Saved cache to {self.path + '.pickle'}")
        except:
            print("Unable to save cache")

    @staticmethod
    def load_signal(args):
        f, offset = args
        try:
            return Signal(f, baseline_shift=offset)
        except ValueError:
            return None

    def get(self, index):
        """
        Returns the signal at the specified index.

        Parameters:
            index (int): The index of the signal to retrieve.

        Returns:
            object: The signal at the specified index.
        """
        return self.signals[index]

    #A function defining how a run object is represented when printed
    #to the command line, etc. Increases readability.
    def __repr__(self):
        return(self.name)

    def plot(self,folder,fft = False):
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
            list(tqdm(pool.imap(self.plot_signals, [(s, folder, fft) for s in self.signals]), total=len(self.signals), desc="Plotting signals", file=sys.stdout))

    @staticmethod
    def plot_signals(args):
        s, folder, plot_fft = args
        s.plot(folder, fft=plot_fft)

    def smooth(self, smoothness = None):
        """
        Smooth each signal in the run with a 10-point moving average.
        Parameters:
            smoothness (int, optional): The size of the window for smoothing the signals. Default is None.
        Returns:
            None
        """
        if smoothness == 'default':
            smoothness = None
        with multiprocessing.Pool() as pool:
            # Use pool.map to parallelize the smoothing of signals and us tqdm to show progress
            self.signals = list(tqdm(pool.imap(self.smooth_signals, [(s, smoothness) for s in self.signals]),
                            total=len(self.signals), desc="Smoothing signals", file=sys.stdout))

    @staticmethod
    def smooth_signals(args):
        s, smoothness = args
        s.smooth(smoothness)
        return s

    def export(self, dir, fft = False):
        """
        Export every signal in the run to a specified folder.
        Parameters:
            dir (str): Directory to save the output files.
            fft (bool, optional): Export FFT if True, time-domain signal if False. Default is False.
        Returns:
            None
        """
        with multiprocessing.Pool() as pool:
            # Use pool.map to parallelize the exporting of signals
            list(tqdm(pool.imap(self.export_signals, [(s, dir, fft) for s in self.signals]), total=len(self.signals), desc="Exporting signals", file=sys.stdout))

    @staticmethod
    def export_signals(args):
        s, folder, fft = args
        filepath = Path(folder) / (s.name + '.csv')
        s.export(filepath, fft)
        return s

    # export signals to a single file
    def export_all(self, filepath, fft = False, show_name = False):
        """
        Export all signals in the run to a single file.
        Parameters:
            filepath (str): The path to the output file. If a directory is provided, the file will be named after the run.
            fft (bool, optional): Export FFT if True, time-domain signal if False. Default is False.
            label (bool, optional): If True, include a header with the units of the signals. Default is False.
            show_name (bool, optional): If True, include the name of the signal in the header. Default is False.
        Returns:
            None
        """
        filepath = Path(filepath)
        if filepath.is_dir():
            filepath = filepath / (self.name + '.csv')

        data = []
        header = []

        for s in self.signals:
            if fft:
                header.append(f"(Hz) {s.name}" if show_name else "(Hz)")
                header.append('Units')

                data.append(s.xf)
                data.append(np.abs(s.yf))
            else:
                data.append(s.x)
                data.append(s.y)

                header.append(f"{s.units[0]} {s.name}" if show_name else s.units[0])
                header.append(s.units[1])


        export(filepath, *data, header=header)

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
            export(filepath, x, np.abs(avg_y), header=['(Hz)', 'Units'])
        else:
            export(filepath, x, avg_y, header=[self.units[0], self.units[1]])

    def fft(self, metric_prefix = None):
        """
        Calculate the FFT for every signal in the run.
        Parameters:
            metric_prefix (optional): The units of the time axis of the signal,
            such as "(um)", "(ms)" or "(s)" . Default is the value of self.units[0].
        Returns:
            None
        """
        for s in self.signals:
            s.fft(metric_prefix)

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
        self.signals = new

    def avg_signal(self, fft = False):
        """
        Calculate the average signal or FFT for the run.
        Parameters:
            fft (bool): If True, calculate the average FFT. If False, calculate the average time-domain signal.
        Returns:
            tuple: Arrays of x-values and average y-values.
        """
        if len(self.signals) == 0:
            raise ValueError("No signals to average.")
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
        areas = [trapezoid(s.y, s.x) for s in self.signals]
        return np.mean(areas)

    def avg_max(self):
        """
        Calculate the average maximum value for the run.
        Returns:
            float: The average maximum value.
        """
        maxes = [s.y.max() for s in self.signals]
        return np.mean(maxes)

    def avg_voltage(self):
        """
        Calculate the average voltage for the run.
        Returns:
            float: The average voltage.
        """
        _, voltage = self.avg_signal()
        return np.mean(voltage)

    def plot_average_signal(self, filepath = None, fft=False, ybottom=None, ytop=None, xleft=None, xright=None):
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

        if filepath == None:
            plt.show()
        else:
            plt.savefig(os.path.join(filepath, self.name + '_avg_signals.png'))

        plt.clf()

def plot_average_signals(A, B, filepath = None, fft=False):
    """
    Plot the average signal or FFT for two runs. Useful for subjectively identifying
    typical signal differences between two treatments.
    Parameters:
        A (run): The first run.
        B (run): The second run.
        filepath (str): The directory to save the plot image.
        fft (bool, optional): Plot FFT if True, time-domain signal if False. Default is False.
        show (bool, optional): If True, display the plot. If False, save the plot. Default is False.
    Returns:
        None
    """
    A_x, A_y = A.avg_signal(fft)

    B_x, B_y = B.avg_signal(fft)

    #Clear the plotting tool
    plt.clf()

    #Plot both average signals over time
    if fft:
        plt.plot(A_x, np.abs(A_y),label=A.name)
        plt.plot(B_x, np.abs(B_y),label=B.name)
    else:
        plt.plot(A_x,A_y,'o',markersize=3,label=A.name)
        plt.plot(B_x,B_y,'o',markersize=3,label=B.name)

    #Add a title and axis labels to the plot
    plt.title('Average signals')
    plt.xlabel('Frequency (Hz)' if fft else 'Time ' + A.units[0])
    plt.ylabel('Magnitude' if fft else 'Amplitude ' + A.units[1])

    #Add a legend to the plot
    plt.legend()

    #Save the plot at the specified location with an auto-generated name
    #and clear the plotting tool
    if filepath == None:
        plt.show()
    else:
        plt.savefig(os.path.join(filepath, A.name + '-' + B.name + '_avg_signals.png'))

    plt.clf()

def corr_coef(A, B):
    """
    Calculate the similarity between two runs.
    Parameters:
        A (run): The first run.
        B (run): The second run.
    Returns:
        float: The similarity between the two runs.
    """
    _, A_y = A.avg_signal()
    _, B_y = B.avg_signal()
    return np.corrcoef(A_y, B_y)[0, 1]

def export(filepath, *lists, header=None):
    """
    Export data to a file.

    Parameters:
        filepath (str): The path to the output file.
        *lists (array-like): The lists of values to export. Each list will be a column in the CSV.
        header (list, optional): The header to write to the output file.

    Returns:
        None
    """
    filepath = Path(filepath, dir_okay=False, file_okay=True)
    # Create the output directory if it does not already exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Find the maximum length of the lists to handle cases where lists have different lengths
    max_length = max(len(lst) for lst in lists)

    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header if provided
        if header != None:
            writer.writerow(header)

        # Write the lists row by row
        for i in range(max_length):
            row = [lst[i] if i < len(lst) else '' for lst in lists]
            writer.writerow(row)

"""
The following functions are used to save and load run objects to and from the filesystem.
"""
def save(run):
    with open(run.path + '.pickle','wb') as f:
        pickle.dump(run,f)
    return

def load(file):
    with open(file,'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    print(VER)