import tkinter as tk
from PIL import Image, ImageTk
import torch
import os
import numpy as np

root = tk.Tk()
root.title("Caine Viewer")
canvas = tk.Canvas(root, width=512, height=512)
canvas.pack()

def update_viewport():
    if os.path.exists('caine_vision.pt'):
        try:
            reconstruction = torch.load('caine_vision.pt', map_location='cpu')
            if reconstruction.dim() == 4:
                img_tensor = reconstruction[0]
            else:
                img_tensor = reconstruction
            img = (img_tensor.permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)

            pil_img = Image.fromarray(img).resize((512, 512), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(pil_img)
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            canvas.image = photo

        except Exception as e:
            print(f"Error loading image: {e}")

    root.after(100, update_viewport)

update_viewport()
root.mainloop()
