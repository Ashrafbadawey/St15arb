from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import io
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from rembg import remove
import tensorflow as tf
import tensorflow_hub as hub
import cv2


# Load the neural style transfer model from TensorFlow Hub
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

app = FastAPI()



# Function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# Function to resize image while maintaining aspect ratio
def resize_image(image, target_size):
    image.thumbnail(target_size, Image.LANCZOS)
    return image

# Function to separate foreground and background
def separate_foreground_background(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    output_image = remove(image)
    input_rgb = np.array(image.convert('RGB'))
    output_rgba = np.array(output_image)

    alpha = output_rgba[:, :, 3]
    alpha3 = np.dstack((alpha, alpha, alpha))
    background_rgb = input_rgb.astype(np.float32) * (1 - alpha3.astype(np.float32) / 255)
    background_rgb = background_rgb.astype(np.uint8)

    foreground = Image.fromarray(output_rgba)
    background = Image.fromarray(background_rgb)
    return foreground, background

# Style transfer function
def apply_style_transfer(content_image, style_image, intensity=1.0):
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.0
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.0

    style_image = style_image * intensity
    outputs = model(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    return tensor_to_image(stylized_image)

# Function to apply glossy effect
def apply_glossy_effect(image):
    image = Image.fromarray(image)

    color_enhancer = ImageEnhance.Color(image)
    vibrant_image = color_enhancer.enhance(1)

    brightness_enhancer = ImageEnhance.Brightness(vibrant_image)
    bright_image = brightness_enhancer.enhance(1)

    glow_image = bright_image.filter(ImageFilter.GaussianBlur(radius=2))
    glossy_image = Image.blend(bright_image, glow_image, alpha=0.9)

    contrast_enhancer = ImageEnhance.Contrast(glossy_image)
    glossy_image = contrast_enhancer.enhance(1.5)

    sharpness_enhancer = ImageEnhance.Sharpness(glossy_image)
    glossy_image = sharpness_enhancer.enhance(2.8)

    edges = glossy_image.filter(ImageFilter.FIND_EDGES)
    glossy_image = Image.blend(glossy_image, edges, alpha=0.1)

    np_img = np.array(glossy_image)
    rows, cols, _ = np_img.shape
    kernel_x = cv2.getGaussianKernel(cols, cols / 2)
    kernel_y = cv2.getGaussianKernel(rows, rows / 2)
    kernel = kernel_y @ kernel_x.T
    mask = 255 * kernel / np.max(kernel)
    vignette = np.zeros_like(np_img, dtype=np.uint8)
    for i in range(3):
        vignette[..., i] = np_img[..., i] * mask.astype(np.float32) / 255
    glossy_image = Image.fromarray(np.clip(vignette, 0, 255).astype(np.uint8))

    return glossy_image  # Return as PIL Image

# Function to add a watermark
def add_watermark(image, text="AIarabAI.com"):
    width, height = image.size
    draw = ImageDraw.Draw(image)

    # Calculate font size based on image width
    font_size = int(width / 35)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text size and position
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = width - text_width - 10
    y = height - text_height - 10

    # Add text to image
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 128))  # White text with transparency

    return image

# Function to process image
def process_image(content_image, style_image):
    if isinstance(content_image, np.ndarray):
        content_image = Image.fromarray(content_image)
    if isinstance(style_image, np.ndarray):
        style_image = Image.fromarray(style_image)

    foreground, background = separate_foreground_background(content_image)

    target_size = content_image.size
    foreground = resize_image(foreground, target_size)
    background = resize_image(background, target_size)

    foreground_rgb = np.array(foreground.convert('RGB'))
    background_rgb = np.array(background)

    styled_foreground = apply_style_transfer(foreground_rgb, np.array(style_image), intensity=1.0)
    styled_background = apply_style_transfer(background_rgb, np.array(style_image), intensity=0.3)

    styled_foreground = styled_foreground.resize(target_size, Image.LANCZOS)
    styled_background = styled_background.resize(target_size, Image.LANCZOS)

    styled_foreground_np = np.array(styled_foreground)
    styled_background_np = np.array(styled_background)

    alpha = np.array(foreground)[:, :, 3] / 255.0
    alpha_resized = np.array(foreground.resize(target_size))[:, :, 3] / 255.0

    combined_image_np = (styled_foreground_np * alpha_resized[..., np.newaxis] +
                         styled_background_np * (1 - alpha_resized[..., np.newaxis]))

    combined_image = Image.fromarray(np.clip(combined_image_np, 0, 255).astype(np.uint8))
    final_image = apply_glossy_effect(np.array(combined_image))

    # Add watermark
    final_image = add_watermark(final_image)

    return final_image

@app.post("/process")
async def process(content_file: UploadFile = File(...), style_file: UploadFile = File(None), style_select: str = Form(None)):
    content_image = Image.open(content_file.file)

    # Determine which style image to use
    if style_file:
        style_image = Image.open(style_file.file)
    elif style_select:
        style_image_path = os.path.join("static/styles", style_select)
        style_image = Image.open(style_image_path)
    else:
        raise HTTPException(status_code=400, detail="No style image provided")

    # Process the images
    final_image = process_image(content_image, style_image)
    
    # Convert the final image to a byte stream
    byte_arr = io.BytesIO()
    final_image.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    
    return StreamingResponse(byte_arr, media_type="image/png")

