## Build `voc-gui` and `voc-cli` for Windows

1. **Install Python**:  
   Download and install Python from [python.org](https://www.python.org/).

2. **Add Python to PATH**:  
   During installation, ensure that you select the option to "Add python.exe to PATH."

3. **Run the Build Script**:  
   Execute the `build.bat` script to create the `voc-cli.exe` and `voc-gui.exe` files.

4. **Locate the Executables**:  
   The `voc-cli.exe` and `voc-gui.exe` programs should be located in the `dist` folder.

## How to Use

### VOC-CLI

`voc-cli.py` is a CLI tool to analyze data from Picoscope 7.

**Usage**: `voc-cli.py [OPTIONS] COMMAND [ARGS]...`

**Options**:
- `--version`                   Show the version and exit.
- `--cache / --no-cache`        Cache the data. Default is to cache.
- `--smoothness INTEGER RANGE`  Smoothness of the data. Default is 10.  [x>=0]
- `--fft / --no-fft`            Use FFT instead of time-domain signal. Default is no-fft.
- `--y-offset FLOAT`            Y-axis offset for the signal. Default is 0.
- `--threshold FLOAT`           Minimum peak height for a signal to be included.
- `--help`                      Show this message and exit.

**Example**:
   ```bash
   voc-cli.py --y-offset 400 --threshold 75 COMMAND [ARGS]...
   ```

**Commands**:
- `average`  Analyze the average signal for a run. Only the plot method works with fft.
  - **Usage**: `voc-cli.py average [OPTIONS] FOLDER`
  - **Options**:
    - `--save-dir DIRECTORY`      Directory to save the average plot. Optional.
    - `--method [plot|area|max]`  Method to analyze signals. Default is plot.
    - `--help`                    Show this message and exit.
  - **Example**:
    ```bash
    voc-cli.py average --save-dir results/average_plot --method area data/run1
    ```

- `compare`  Compare the signals of two runs. Only the method avg-plot supports fft.
  - **Usage**: `voc-cli.py compare [OPTIONS] FOLDER_A FOLDER_B`
  - **Options**:
    - `--save-dir DIRECTORY`            Directory to save the comparison plot. Optional.
    - `--method [avg-plot|avg-area|avg-max|average|correlation]`  Method to compare signals. Default is avg-plot.
    - `--help`                          Show this message and exit.
  - **Example**:
    ```bash
    voc-cli.py compare --save-dir results/comparison_plot --method avg-area data/run1 data/run2
    ```

- `export`   Export the signals of a run to CSV files.
  - **Usage**: `voc-cli.py export [OPTIONS] DATA SAVE_PATH`
  - **Options**:
    - `--save-as [single|multi]`  Export as multiple CSV files or as a single CSV file. Default is single.
    - `--help`                    Show this message and exit.
  - **Example**:
    ```bash
    voc-cli.py export --save-as multi data/run1 results/data_export
    ```

- `plot`     Plot all signals from a run and save them to a specified folder.
  - **Usage**: `voc-cli.py plot [OPTIONS] FOLDER SAVE_DIR`
  - **Options**:
    - `--help`  Show this message and exit.
  - **Example**:
    ```bash
    voc-cli.py plot data/run1 results/plots
    ```
    
### VOC-GUI

`voc-gui` provides a graphical interface for analyzing data from Picoscope 7.

**Features**:
- **Cache**: Option to enable or disable caching of data.
- **Smoothness**: Adjust the smoothness of the data (default is 10).
- **FFT**: Toggle between FFT and time-domain signal analysis.
- **Y-Offset**: Set the Y-axis offset for the signal (default is 0).
- **Threshold**: Set the minimum peak height for a signal to be included.
- **Folder Selection**: Choose the folder containing the data.
- **Save Directory**: Choose where to save the output files.

**Operations**:
- **Plot Signals**: Plot all signals from the selected folder and save them to the specified directory.
- **Show Avg Signal**: Display the average signal for the selected folder.
- **Compare Runs**: Compare signals from two different folders.
- **Export to CSV**: Export signals to CSV files (single or multiple).
- **Show Version Info**: Display the version information of the GUI and VOC API.

**Example Usage**:
1. **Plot Signals**:
   - Select the folder containing the data.
   - Choose a directory to save the plotted signals.
   - Click "Plot Signals."

2. **Show Average Signal**:
   - Select the folder containing the data.
   - Click "Show Avg Signal."

3. **Compare Runs**:
   - Select folders for Folder A and Folder B.
   - Choose a directory to save the comparison plot.
   - Click "Compare Runs."

4. **Export to CSV**:
   - Select the folder containing the data.
   - Choose a directory to save the CSV files.
   - Click "Export to one CSV" or "Export to multiple CSVs."

5. **Show Version Info**:
   - Click "Show Version Info" to display the current version of the GUI and VOC API.
