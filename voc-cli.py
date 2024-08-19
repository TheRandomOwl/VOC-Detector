import click
import voc
from pathlib import Path

"""
A CLI tool to analyze data from Picoscope 7.

Uses the voc module to analyze data from Picoscope 7 and plot signals.
"""

VER = '0.4.0'
API = voc.VER

# Helper Functions
def validate_dir(path: Path):
    """Create directory if it doesn't exist."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

# Main CLI
@click.group()
@click.version_option(version=VER, message=f"Version {VER} of the voc-cli using voc API {API}")
@click.option('--cache/--no-cache', default=True, help="Cache the data. Default is to cache.")
@click.option('--smoothness', type=click.IntRange(min=0), default=10, help="Smoothness of the data. Default is 10.")
@click.option('--fft/--no-fft', default=False, help="Plot FFT instead of time-domain signal. Default is no-fft.")
@click.option('--y-offset', type=float, default=0, help="Y-axis offset for the signal. Default is 0.")
@click.option('--min', type=float, help="Minimum peak height for a signal to be included.")
@click.pass_context
def cli(ctx, cache, smoothness, fft, y_offset, min):
    """A CLI tool to analyze data from Picoscope 7."""
    # Store options in context
    ctx.ensure_object(dict)
    ctx.obj.update(cache=cache, smoothness=smoothness, fft=fft, y_offset=y_offset, min=min)

# CLI Commands
@cli.command()
@click.argument('folder', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--save-dir', type=click.Path(file_okay=False, dir_okay=True), required=True, help="Directory to save the plots.")
@click.pass_context
def plot(ctx, folder, save_dir):
    """Plot signals and save to the specified folder."""
    # cache, smoothness, fft, y_offset = get_common_options(ctx)
    save_path = Path(save_dir)
    validate_dir(save_path)

    signals = voc.Run(folder, cache=ctx.obj['cache'], smoothness=ctx.obj['smoothness'], y_offset=ctx.obj['y_offset'])
    if ctx.obj['min'] != None:
        signals.clean_empty(threshold=ctx.obj['min'])

    signals.plot(save_path, fft=ctx.obj['fft'])
    click.echo(f"Plotted signals to folder: {save_path}")

@cli.command()
@click.argument('folder', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--save-dir', type=click.Path(file_okay=False, dir_okay=True), required=False, help="Directory to save the average plot. Optional.")
@click.pass_context
def average(ctx, folder, save_dir):
    """Plot the average signal or FFT for a run."""

    if save_dir != None:
        save_dir = Path(save_dir)
        validate_dir(save_dir)
    
    signals = voc.Run(folder, cache=ctx.obj['cache'], smoothness=ctx.obj['smoothness'], y_offset=ctx.obj['y_offset'])
    if ctx.obj['min'] != None:
        signals.clean_empty(threshold=ctx.obj['min'])
    
    signals.plot_average_signal(save_dir, fft=ctx.obj['fft'])
    if save_dir != None:
        click.echo(f"Saved average plot to folder: {save_dir}")

@cli.command()
@click.argument('folder_a', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('folder_b', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--save-dir', type=click.Path(file_okay=False, dir_okay=True), required=False, help="Directory to save the comparison plot. Optional.")
@click.option('--method', type=click.Choice(['avg-plot', 'placeholder']), default='avg-plot', help="Method to compare signals. Default is avg-plot.")
@click.pass_context
def compare(ctx, folder_a, folder_b, save_dir, method):
    """Compare the signals of two runs."""
    if save_dir != None:
        save_dir = Path(save_dir)
        validate_dir(save_dir)

    A = voc.Run(folder_a, cache=ctx.obj['cache'], smoothness=ctx.obj['smoothness'], y_offset=ctx.obj['y_offset'])
    B = voc.Run(folder_b, cache=ctx.obj['cache'], smoothness=ctx.obj['smoothness'], y_offset=ctx.obj['y_offset'])
    if ctx.obj['min'] != None:
        A.clean_empty(threshold=ctx.obj['min'])
        B.clean_empty(threshold=ctx.obj['min'])
    
    if method == 'avg-plot':
        voc.plot_average_signals(A, B, save_dir, fft=ctx.obj['fft'])
        if save_dir != None:
            click.echo(f"Saved comparison plot to folder: {save_dir}")

if __name__ == '__main__':
    cli()
