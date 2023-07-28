"""
https://www.perplexity.ai/search/df594c3b-5104-4c76-857d-ce706244ca14?s=c
"""
import click
import cv2

from detector import detect_aruco_cv2, detect_coins_cv2, detect_legos_cv2, detect_objects_yolo, detect_legos_yolo_custom
from paper import detect_paper_cv2
from util import capture_image


@click.group()
def cli():
    pass

@cli.command()
@click.argument('input_file', type=click.Path(), required=False)
def detect_aruco(input_file):
    """Detect ArUco markers in an image."""
    click.echo(f"Detecting ArUco markers in {input_file}...")
    input_file = input_file if input_file else capture_image()
    img = cv2.imread(input_file)

    detect_aruco_cv2(img)

@cli.command()
@click.argument('input_file', type=click.Path(), required=False)
def detect_coins(input_file):
    """Detect coins in an image."""
    click.echo(f"Detecting coins in {input_file}...")
    input_file = input_file if input_file else capture_image()
    img = cv2.imread(input_file)

    detect_coins_cv2(img)


@cli.command()
@click.argument('input_file', type=click.Path(), required=False)
def detect_paper(input_file):
    """Detect paper in an image and applies perspective transformation."""
    click.echo(f"Detecting paper in {input_file}...")
    input_file = input_file if input_file else capture_image()
    img = cv2.imread(input_file)

    detect_paper_cv2(img)


@cli.command()
@click.argument('input_file', type=click.Path(), required=False)
def detect_legos(input_file):
    """Detect legos in an image."""
    click.echo(f"Detecting legos in {input_file}...")
    input_file = input_file if input_file else capture_image()
    img = cv2.imread(input_file)

    detect_legos_cv2(img)

@cli.command()
@click.argument('input_file', type=click.Path(), required=False)
def detect_legos_nn(input_file):
    """Detect legos in an image using a custom trained YOLOv7 neural-net."""
    click.echo(f"Detecting LEGO bricks in {input_file}...")
    input_file = input_file if input_file else capture_image()
    img = cv2.imread(input_file)
    detect_legos_yolo_custom(img)

@cli.command()
@click.argument('input_file', type=click.Path(), required=False)
def detect_objects_nn(input_file):
    """Detect objects in an image using the standard YOLOv3 neural-net."""
    click.echo(f"Detecting objects using YOLO in {input_file}...")
    input_file = input_file if input_file else capture_image()
    img = cv2.imread(input_file)
    detect_objects_yolo(img)

if __name__ == '__main__':
    cli()
