import tempfile

import click
import cv2


def capture_image():
    """
    https://www.perplexity.ai/search/fc1411c8-6630-4b18-bf42-7d34f14c8457
    """
    # Capture image from camera
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    camera.release()

    if not ret:
        click.echo("Failed to capture image from camera.")
        return

    # Save the frame to a new temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, frame)
        click.echo(f"Frame saved to {tmp_file.name}")
        return tmp_file.name
