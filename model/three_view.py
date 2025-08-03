from PIL import Image
from OCC.Display.OCCViewer import OffscreenRenderer
from OCC.Core.STEPControl import STEPControl_Reader
# from OCC.Display.OCCViewer import Viewer3d
from OCC.Core.Graphic3d import Graphic3d_BufferType
import numpy as np
import matplotlib.pyplot as plt

# def read_step_file(filename):
#     step_reader = STEPControl_Reader()
#     step_reader.ReadFile(filename)
#     step_reader.TransferRoots()
#     shape = step_reader.OneShape()
#     return shape

def plot_views(views):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (view, title) in zip(axes, views.items()):
        ax.imshow(view)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def get_three_view(shape):

    offscreen_renderer = OffscreenRenderer()
    # by default, the offscreenrenderer size is 640*480
    offscreen_renderer.Create()
    offscreen_renderer.SetModeShaded()
    if shape is not None:
        offscreen_renderer.DisplayShape(shape, update=True)

    print("render ready")
    
    # Capture views
    views = {}
    for view_name in ['top', 'front', 'side']:
        # image = capture_view(offscreen_renderer, view_name)
        display = offscreen_renderer
        display.FitAll()
        display.View_Iso()
        if view_name == 'top':
            display.View_Top()
        elif view_name == 'front':
            display.View_Front()
        elif view_name == 'side':
            display.View_Right()
        display.FitAll()
        display.Repaint()
        
        # Capture image from the display
        # export to a 640*480 image data
        data_640_480 = display.GetImageData(640, 480, Graphic3d_BufferType.Graphic3d_BT_RGB)
        print("got data")
        
        pil_image = Image.frombytes('RGB', (640, 480), data_640_480)
        # pil_image.show()
        views[view_name] = pil_image
        
    
    # Close display
    offscreen_renderer.Context.RemoveAll(True)
    
    # Plot and return views as NumPy arrays
    # plot_views(views)
    return views

def get_three_view_shapes(shapes, height=256, width=256):

    offscreen_renderer = OffscreenRenderer(screen_size=(width, height))
    # by default, the offscreenrenderer size is 640*480
    # offscreen_renderer.Create()
    # offscreen_renderer.SetModeShaded()

    shapes_view = []
    for shape in shapes:
        if shape is not None:
            ais_shape = offscreen_renderer.DisplayShape(shape, update=True)  # don't set update=True or would zoom in

        # print("render ready")
        
        # Capture views
        views = []
        display = offscreen_renderer
        # display.View_Iso()
        for view_name in ['top', 'front', 'side']:
            if view_name == 'top':
                display.View_Top()
            elif view_name == 'front':
                display.View_Front()
            elif view_name == 'side':
                display.View_Right()
            
            # if shape == shapes[0]:
            display.FitAll()
            # display.Repaint()
            display.ZoomFactor(0.9)
            
            # Capture image from the display
            # export to a 640*480 image data
            data_640_480 = display.GetImageData(width, height, Graphic3d_BufferType.Graphic3d_BT_RGB)
            # print("got data")
            
            # pil_image = Image.frombytes('RGB', (640, 480), data_640_480)
            # np array image
            np_image = np.frombuffer(data_640_480, dtype=np.uint8).reshape(height, width, 3)
            views.append(np_image)
        # Close display
        # display.Context.RemoveAll(True)
        if shape is not None:
            display.Context.Remove(ais_shape[0], True)
        shapes_view.append(views)

    return shapes_view


