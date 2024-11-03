# Image Colorization GUI

This application colorizes grayscale images, using a deep learning model with a simple and intuitive GUI built with `wxPython` and `OpenCV`. Users can load grayscale images, colorize them, and save the results. This project demonstrates deep learning integration in a desktop GUI.

## Features

- **Image Loading**: Select an image from your computer to colorize.
- **Drag-and-Drop Support**: Drag and drop an image file directly onto the app to load it.
- **Colorization**: Colorizes grayscale images using a pre-trained deep learning model.
- **Save Colorized Image**: Save the colorized output in your desired location.

## Requirements

- Python 3.6 or above
- OpenCV
- wxPython
- NumPy

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/ImageColorizationGUI.git
    cd ImageColorizationGUI
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:
    ```bash
    python Final.py
    ```

2. **Load Image**: Use the "Open File" button to load a grayscale image, or drag and drop an image onto the app.

3. **Colorize**: Click the "Colorize" button to generate a colorized version of the loaded image.

4. **Save**: Save the colorized image by clicking "Save Image."

## File Structure

. ├── app.py                  # Main application script ├── model/ │   ├── colorization_deploy_v2.prototxt │   ├── colorization_release_v2.caffemodel │   └── pts_in_hull.npy └── README.md               # This file

## Troubleshooting

- **Model Loading Issues**: If the model files are not found, ensure they are correctly placed in the `model` directory.
- **wxPython GUI Errors**: wxPython can sometimes have installation issues. Make sure you are using a compatible Python version, as wxPython does not work with all versions of Python.

## Acknowledgements

- The colorization model is from Richard Zhang et al., "[Colorful Image Colorization](https://arxiv.org/abs/1603.08511)."

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"""
