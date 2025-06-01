import torch
import cv2

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

import torchvision.transforms as transforms

import app_config



# Load your TorchScript model (replace with your model path)
model_path = app_config.model_path
model = torch.jit.load(model_path)
model.eval()

def open_image():
    filepath = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if not filepath:
        return
    
    # Загружаем изображение с помощью PIL
    img = Image.open(filepath).convert("RGB")
    
    # Преобразуем в numpy массив для передачи в callback
    input_tensor = torch.from_numpy(np.array(img).astype(np.float32)).to(app_config.device)
    
    # Создаем фиктивные боксы и классы для примера (или получайте их из модели)
    with torch.no_grad():
        output = model(input_tensor)

    boxes, class_ids, confidences, _ = output
    boxes = boxes.cpu().numpy()
    class_ids = class_ids.cpu().numpy()
    confidences = confidences.cpu().numpy()
    
    # Вызов вашей callback функции
    result_img_bgr = app_config.callback_func(input_tensor, boxes, class_ids, confidences)
    
    # Конвертируем обратно в PIL для отображения
    result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
    result_pil_img = Image.fromarray(result_img_rgb)
    
    # Отображаем результат в интерфейсе
    result_display_img = result_pil_img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(result_display_img)
    
    label_image.config(image=img_tk)
    label_image.image = img_tk


# Create GUI window
root = tk.Tk()
root.title(app_config.app_name)

# Center the window on the screen
window_width = 400  # Set your desired width
window_height = 400  # Set your desired height

# Get the screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the x and y coordinates to center the window
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)

# Set the dimensions of the window
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Button to open image file
btn_open = tk.Button(root, text="Open Image", command=open_image)
btn_open.pack(pady=10)

# Label to display image
label_image = tk.Label(root)
label_image.pack()

# Run the app
root.mainloop()