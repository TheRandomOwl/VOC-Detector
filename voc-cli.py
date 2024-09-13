"""
A CLI tool to analyze data from Picoscope 7.

This program provides a command-line interface (CLI) for analyzing data from Picoscope 7. It offers several commands to perform different analysis tasks on the data.

Commands:
- `plot`: Plot all signals from a run and save them to a specified folder.
- `average`: Analyze the average signal for a run. Supports different analysis methods.
- `compare`: Compare the signals of two runs. Supports different comparison methods.
- `export`: Export the signals of a run to CSV files.

Options:
- `--cache/--no-cache`: Cache the data. Default is to cache.
- `--smoothness`: Smoothness of the data. Default is 10.
- `--fft/--no-fft`: Use FFT instead of time-domain signal. Default is no-fft.
- `--y-offset`: Y-axis offset for the signal. Default is 0.
- `--threshold`: Minimum peak height for a signal to be included.

Usage:
python voc-cli.py [command] [options] [arguments]

For more information on each command and its options, use the `--help` flag after the command.

Note: This program requires the `voc` version 4.5.0+ module to be installed.

Author: Nathan Perry
Supervisor: Dr. Reinhard Schulte
For: LLU Volatile Organic Compound Detector Siganl Analysis
Date: September 2024

"""

import multiprocessing
from pathlib import Path
import click
import voc

VER = '1.0.0'
API = voc.VER

# Main CLI
@click.group()
@click.version_option(version=VER, message=f"%(prog)s, version %(version)s, VOC API version {API}")
@click.option('--cache/--no-cache', default=True, help="Cache the data. Default is to cache.")
@click.option('--smoothness', type=click.IntRange(min=0), default=voc.WINDOW_SIZE, help=f"Smoothness of the data. Default is {voc.WINDOW_SIZE}.")
@click.option('--fft/--no-fft', default=False, help="Use FFT instead of time-domain signal. Default is no-fft.")
@click.option('--y-offset', type=float, default=0, help="Y-axis offset for the signal. Default is 0.")
@click.option('--threshold', type=float, help="Minimum peak height for a signal to be included.")
@click.option('--normalize/--no-normalize', default=False, help="Normalize the signals to start at zero seconds. Default is no-normalize.")
@click.pass_context
def cli(ctx, cache, smoothness, fft, y_offset, threshold, normalize):
    """A CLI tool to analyze data from Picoscope 7."""
    # Store options in context
    ctx.ensure_object(dict)
    ctx.obj.update(cache=cache, smoothness=smoothness, fft=fft, y_offset=y_offset, min=threshold, norm=normalize)

# CLI Commands

# Plot Command
@cli.command()
@click.argument('folder', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('save-dir', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--top', type=float, help="Top limit for the plot. Optional.")
@click.option('--bottom', type=float, help="Bottom limit for the plot. Optional.")
@click.pass_context
def plot(ctx, folder, save_dir, top, bottom):
    """Plot all signals from a run and save them to a specified folder."""

    signals = voc.Run(folder, cache=ctx.obj['cache'], smoothness=ctx.obj['smoothness'], y_offset=ctx.obj['y_offset'], fft=ctx.obj['fft'], normalize=ctx.obj['norm'])
    if ctx.obj['min'] is not None:
        signals.clean_empty(ctx.obj['min'])

    signals.plot(save_dir, fft=ctx.obj['fft'], ymin=bottom, ymax=top)
    click.echo(f"Plotted signals to folder: {save_dir}")

# Average Command
@cli.command()
@click.argument('folder', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--save-dir', type=click.Path(file_okay=False, dir_okay=True), required=False, help="Directory to save the average plot. Optional.")
@click.option('--method', type=click.Choice(['plot', 'area', 'max', 'average', 'all']), default='plot', help="Method to analyze signals. Default is plot.")
@click.pass_context
def average(ctx, folder, save_dir, method):
    """Analyze the average signal for a run. Only the plot method works with fft."""

    signals = voc.Run(folder, cache=ctx.obj['cache'], smoothness=ctx.obj['smoothness'], y_offset=ctx.obj['y_offset'], fft=ctx.obj['fft'], normalize=ctx.obj['norm'])
    if ctx.obj['min'] is not None:
        signals.clean_empty(ctx.obj['min'])

    if method in ['area', 'all']:
        click.echo(f"Area of {signals.name}: {signals.avg_area()} {signals.units[0]}*{signals.units[1]}")
    if method in ['max', 'all']:
        click.echo(f"Max of {signals.name}: {signals.avg_max()} {signals.units[1]}")
    if method in ['average', 'all']:
        click.echo(f"Average voltage of {signals.name}: {signals.avg_voltage()} {signals.units[1]}")
    if method in ['plot', 'all']:
        signals.plot_average_signal(save_dir, fft=ctx.obj['fft'])
        if save_dir is not None:
            click.echo(f"Saved average plot to folder: {save_dir}")

# Compare Command
@cli.command()
@click.argument('folder_a', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('folder_b', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--save-dir', type=click.Path(file_okay=False, dir_okay=True), help="Directory to save the comparison plot. Optional.")
@click.option('--method', type=click.Choice(['avg-plot', 'avg-area', 'avg-max', 'average', 'correlation', 'all'])
              , default='avg-plot', help="Method to compare signals. Default is avg-plot.")
@click.pass_context
def compare(ctx, folder_a, folder_b, save_dir, method):
    """Compare the signals of two runs. Only the method avg-plot supports fft."""

    run1 = voc.Run(folder_a, cache=ctx.obj['cache'], smoothness=ctx.obj['smoothness'], y_offset=ctx.obj['y_offset'], fft=ctx.obj['fft'], normalize=ctx.obj['norm'])
    run2 = voc.Run(folder_b, cache=ctx.obj['cache'], smoothness=ctx.obj['smoothness'], y_offset=ctx.obj['y_offset'], fft=ctx.obj['fft'], normalize=ctx.obj['norm'])
    if ctx.obj['min'] is not None:
        run1.clean_empty(ctx.obj['min'])
        run2.clean_empty(ctx.obj['min'])

    if method in ['avg-area', 'all']:
        click.echo(f"Area of {run1.name}: {run1.avg_area()} {run1.units[0]}*{run1.units[1]}")
        click.echo(f"Area of {run2.name}: {run2.avg_area()} {run2.units[0]}*{run2.units[1]}")
    if method in ['avg-max', 'all']:
        click.echo(f"Max of {run1.name}: {run1.avg_max()} {run1.units[1]}")
        click.echo(f"Max of {run2.name}: {run2.avg_max()} {run2.units[1]}")
    if method in ['average', 'all']:
        click.echo(f"Average voltage of {run1.name}: {run1.avg_voltage()} {run1.units[1]}")
        click.echo(f"Average voltage of {run2.name}: {run2.avg_voltage()} {run2.units[1]}")
    if method in ['correlation', 'all']:
        click.echo(f"Correlation coefficient: {voc.corr_coef(run1,run2)}")
    if method in ['avg-plot', 'all']:
        voc.plot_average_signals(run1, run2, save_dir, fft=ctx.obj['fft'])
        if save_dir is not None:
            click.echo(f"Saved comparison plot to folder: {save_dir}")

# Export Command
@cli.command()
@click.argument('data', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('save-path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--method', type=click.Choice(['raw', 'avg']), default='raw', help="Export raw signals or average signals. Default is raw.")
@click.option('--save-as', type=click.Choice(['single', 'multi']), default='single', help="Export as multiple CSV files or as a single CSV file. Default is single.")
@click.pass_context
def export(ctx, data, save_path, method, save_as):
    """Export the signals of a run to CSV files."""

    signals = voc.Run(data, cache=ctx.obj['cache'], smoothness=ctx.obj['smoothness'], y_offset=ctx.obj['y_offset'], fft=ctx.obj['fft'], normalize=ctx.obj['norm'])
    if ctx.obj['min'] is not None:
        signals.clean_empty(ctx.obj['min'])

    if save_as == 'multi':
        if Path(save_path).is_file():
            raise click.BadParameter("Save path must be a directory for multi export.")
        if method == 'avg':
            raise click.BadParameter("Average signal cannot be exported as multiple files.")

        signals.export(save_path, fft=ctx.obj['fft'])
        click.echo(f"Exported signals to folder: {save_path}")
    elif save_as == 'single':
        if method == 'avg':
            signals.export_avg(save_path, fft=ctx.obj['fft'])
            click.echo(f"Exported average signals to: {save_path}")
        elif method == 'raw':
            signals.export_all(save_path, fft=ctx.obj['fft'])
            click.echo(f"Exported signals to: {save_path}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    cli()
