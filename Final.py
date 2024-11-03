import wx
import cv2
import numpy as np
import os

prototxt = os.path.join(os.path.dirname(__file__), 'model/colorization_deploy_v2.prototxt')
model = os.path.join(os.path.dirname(__file__), 'model/colorization_release_v2.caffemodel')
points = os.path.join(os.path.dirname(__file__), 'model/pts_in_hull.npy')

if not os.path.isfile(model):
    print("Model file missing. Please download colorization_release_v2.caffemodel and place it in the 'model' folder.")
    exit()

net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image_filename=None, cv2_frame=None):
    image = cv2.imread(image_filename) if image_filename else cv2_frame
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    return image, colorized

class ColorizeApp(wx.Frame):
    def __init__(self, *args, **kw):
        super(ColorizeApp, self).__init__(*args, **kw)
        self.SetTitle("Colorize Old Photos")
        self.SetBackgroundColour(wx.Colour("#f4f4f9"))
        self.SetSize((900, 700))
        
        # Main Panel
        panel = wx.Panel(self)
        
        # Custom Font for Labels
        font = wx.Font(12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        
        # Layouts
        main_vbox = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        # Labels
        label_input = wx.StaticText(panel, label="Original Image")
        label_input.SetFont(font)
        label_output = wx.StaticText(panel, label="Colorized Image")
        label_output.SetFont(font)

        # Image placeholders with borders
        self.input_image = wx.StaticBitmap(panel, size=(350, 350))
        self.input_image.SetBackgroundColour(wx.Colour("#e0e0e0"))
        self.output_image = wx.StaticBitmap(panel, size=(350, 350))
        self.output_image.SetBackgroundColour(wx.Colour("#e0e0e0"))

        # Arrange images and labels
        input_vbox = wx.BoxSizer(wx.VERTICAL)
        output_vbox = wx.BoxSizer(wx.VERTICAL)
        input_vbox.Add(label_input, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        input_vbox.Add(self.input_image, 1, wx.EXPAND | wx.ALL, 10)
        output_vbox.Add(label_output, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        output_vbox.Add(self.output_image, 1, wx.EXPAND | wx.ALL, 10)
        
        hbox1.Add(input_vbox, 1, wx.EXPAND | wx.ALL, 10)
        hbox1.Add(output_vbox, 1, wx.EXPAND | wx.ALL, 10)

        # Controls for File Picker, Gray Option, and Buttons
        self.file_picker = wx.FilePickerCtrl(panel, message="Select an image to colorize", wildcard="Image files (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png")
        self.gray_checkbox = wx.CheckBox(panel, label="Convert to grayscale before colorizing")
        colorize_btn = wx.Button(panel, label="Colorize Image", size=(140, 40))
        save_btn = wx.Button(panel, label="Save Colorized Image", size=(160, 40))

        colorize_btn.SetBackgroundColour(wx.Colour("#4CAF50"))
        colorize_btn.SetForegroundColour(wx.Colour("white"))
        save_btn.SetBackgroundColour(wx.Colour("#2196F3"))
        save_btn.SetForegroundColour(wx.Colour("white"))

        # Arrange controls
        hbox2.Add(self.file_picker, 1, wx.EXPAND | wx.ALL, 10)
        hbox2.Add(self.gray_checkbox, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 10)
        hbox2.Add(colorize_btn, 0, wx.ALL, 5)
        hbox2.Add(save_btn, 0, wx.ALL, 5)

        # Combine layouts
        main_vbox.Add(hbox1, 1, wx.EXPAND)
        main_vbox.Add(hbox2, 0, wx.EXPAND)
        panel.SetSizer(main_vbox)

        # Event Bindings
        self.Bind(wx.EVT_BUTTON, self.on_colorize, colorize_btn)
        self.Bind(wx.EVT_BUTTON, self.on_save, save_btn)
        self.Bind(wx.EVT_FILEPICKER_CHANGED, self.on_file_selected, self.file_picker)

        # Images for Processing
        self.original_image = None
        self.colorized_image = None

    def on_file_selected(self, event):
        filepath = self.file_picker.GetPath()
        self.original_image = cv2.imread(filepath)
        self.display_image(self.input_image, self.original_image)

    def on_colorize(self, event):
        if self.original_image is None:
            wx.MessageBox("Please select an image first.", "Error", wx.ICON_ERROR)
            return

        if self.gray_checkbox.IsChecked():
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.merge([gray_image, gray_image, gray_image])
            _, self.colorized_image = colorize_image(cv2_frame=gray_image)
        else:
            _, self.colorized_image = colorize_image(cv2_frame=self.original_image)

        self.display_image(self.output_image, self.colorized_image)

    def on_save(self, event):
        if self.colorized_image is None:
            wx.MessageBox("No colorized image to save. Please colorize an image first.", "Error", wx.ICON_ERROR)
            return

        with wx.FileDialog(self, "Save Colorized Image", wildcard="JPEG files (*.jpg)|*.jpg|PNG files (*.png)|*.png",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return

            save_path = file_dialog.GetPath()
            cv2.imwrite(save_path, self.colorized_image)
            wx.MessageBox("Image saved successfully.", "Info", wx.ICON_INFORMATION)

    def display_image(self, wx_image_ctrl, cv_image):
        h, w = cv_image.shape[:2]
        wx_image = wx.Bitmap.FromBuffer(w, h, cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        wx_image_ctrl.SetBitmap(wx_image)
        self.Layout()

if __name__ == "__main__":
    app = wx.App(False)
    frame = ColorizeApp(None)
    frame.Show()
    app.MainLoop()
