import click
import voc
import multiprocessing
from pathlib import Path

"""
A CLI tool to analyze data from Picoscope 7.

Uses the voc module to analyze data from Picoscope 7 and plot signals.
"""

VER = '0.7.6'
API = voc.VER

# Main CLI
@click.group()
@click.version_option(version=VER, message=f"%(prog)s, version %(version)s, VOC API version {API}")
@click.option('--cache/--no-cache', default=True, help="Cache the data. Default is to cache.")
@click.option('--smoothness', type=click.IntRange(min=0), default=10, help="Smoothness of the data. Default is 10.")
@click.option('--fft/--no-fft', default=False, help="Use FFT instead of time-domain signal. Default is no-fft.")
@click.option('--y-offset', type=float, default=0, help="Y-axis offset for the signal. Default is 0.")
@click.option('--threshold', type=float, help="Minimum peak height for a signal to be included.")
@click.pass_context
def cli(ctx, cache, smoothness, fft, y_offset, threshold):
    """A CLI tool to analyze data from Picoscope 7."""
    # Store options in context
    ctx.ensure_object(dict)
    ctx.obj.update(cache=cache, smoothness=smoothness, fft=fft, y_offset=y_offset, min=threshold)

# CLI Commands

# Plot Command
@cli.command()
@click.argument('folder', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('save-dir', type=click.Path(file_okay=False, dir_okay=True))
@click.pass_context
def plot(ctx, folder, save_dir):
    """Plot all signals from a run and save them to a specified folder."""

    signals = voc.Run(folder, cache=ctx.obj['cache'], smoothness=ctx.obj['smoothness'], y_offset=ctx.obj['y_offset'])
    if ctx.obj['min'] != None:
        signals.clean_empty(ctx.obj['min'])

    signals.plot(save_dir, fft=ctx.obj['fft'])
    click.echo(f"Plotted signals to folder: {save_dir}")

# Average Command
@cli.command()
@click.argument('folder', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--save-dir', type=click.Path(file_okay=False, dir_okay=True), required=False, help="Directory to save the average plot. Optional.")
@click.option('--method', type=click.Choice(['plot', 'area', 'max']), default='plot', help="Method to analyze signals. Default is plot.")
@click.pass_context
def average(ctx, folder, save_dir, method):
    """Analyze the average signal for a run. Only the plot method works with fft."""

    signals = voc.Run(folder, cache=ctx.obj['cache'], smoothness=ctx.obj['smoothness'], y_offset=ctx.obj['y_offset'])
    if ctx.obj['min'] != None:
        signals.clean_empty(ctx.obj['min'])

    if method == 'plot':
        signals.plot_average_signal(save_dir, fft=ctx.obj['fft'])
        if save_dir != None:
            click.echo(f"Saved average plot to folder: {save_dir}")
    elif method == 'area':
        click.echo(f"Average area: {signals.avg_area()}")
    elif method == 'max':
        click.echo(f"Average max: {signals.avg_max()}")

# Compare Command
@cli.command()
@click.argument('folder_a', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('folder_b', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--save-dir', type=click.Path(file_okay=False, dir_okay=True), help="Directory to save the comparison plot. Optional.")
@click.option('--method', type=click.Choice(['avg-plot', 'avg-area', 'avg-max', 'average', 'correlation'])
              , default='avg-plot', help="Method to compare signals. Default is avg-plot.")
@click.pass_context
def compare(ctx, folder_a, folder_b, save_dir, method):
    """Compare the signals of two runs. Only the method avg-plot supports fft."""

    A = voc.Run(folder_a, cache=ctx.obj['cache'], smoothness=ctx.obj['smoothness'], y_offset=ctx.obj['y_offset'])
    B = voc.Run(folder_b, cache=ctx.obj['cache'], smoothness=ctx.obj['smoothness'], y_offset=ctx.obj['y_offset'])
    if ctx.obj['min'] != None:
        A.clean_empty(ctx.obj['min'])
        B.clean_empty(ctx.obj['min'])

    if method == 'avg-plot':
        voc.plot_average_signals(A, B, save_dir, fft=ctx.obj['fft'])
        if save_dir != None:
            click.echo(f"Saved comparison plot to folder: {save_dir}")
    elif method == 'avg-area':
        click.echo(f"Area of {A.name}: {A.avg_area()}")
        click.echo(f"Area of {B.name}: {B.avg_area()}")
    elif method == 'avg-max':
        click.echo(f"Max of {A.name}: {A.avg_max()}")
        click.echo(f"Max of {B.name}: {B.avg_max()}")
    elif method == 'average':
        click.echo(f"Average voltage of {A.name}: {A.avg_voltage()}")
        click.echo(f"Average voltage of {B.name}: {B.avg_voltage()}")
    elif method == 'correlation':
        click.echo(f"Correlation coefficient: {voc.corr_coef(A,B)}")

# Export Command
@cli.command()
@click.argument('data', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('save-path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--method', type=click.Choice(['raw', 'avg']), default='raw', help="Export raw signals or average signals. Default is raw.")
@click.option('--save-as', type=click.Choice(['single', 'multi']), default='single', help="Export as multiple CSV files or as a single CSV file. Default is single.")
@click.pass_context
def export(ctx, data, save_path, method, save_as):
    """Export the signals of a run to CSV files."""

    signals = voc.Run(data, cache=ctx.obj['cache'], smoothness=ctx.obj['smoothness'], y_offset=ctx.obj['y_offset'])
    if ctx.obj['min'] != None:
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
