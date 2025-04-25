import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import io
import tensorflow as tf

# --- Set up the canvas ---
drawing = False
points = []

def on_press(event):
    global drawing
    drawing = True
    points.append((event.xdata, event.ydata))

def on_release(event):
    global drawing
    drawing = False

def on_move(event):
    if drawing and event.xdata is not None and event.ydata is not None:
        points.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, 'ko', markersize=12)  # Increased marker size
        fig.canvas.draw()

def clear_canvas(event):
    global points
    points = []
    ax.clear()
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 28)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.draw()

def save_and_predict(event):
    if not points:  # Don't predict if canvas is empty
        print("Please draw a digit first")
        return
    
    # Save canvas as image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=28, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # Load and process image
    img = Image.open(buf).convert('L')
    img_array = np.array(img)
    
    # Invert colors (MNIST has white digits on black background)
    img_array = 255 - img_array
    
    # Threshold to make pure black and white
    img_array[img_array < 128] = 0
    img_array[img_array >= 128] = 255
    
    # Find bounding box of the digit
    non_zero = np.nonzero(img_array)
    if len(non_zero[0]) == 0:
        print("No digit detected")
        return
        
    min_y, max_y = np.min(non_zero[0]), np.max(non_zero[0])
    min_x, max_x = np.min(non_zero[1]), np.max(non_zero[1])
    
    # Add padding
    pad = 2
    min_x, max_x = max(0, min_x-pad), min(img_array.shape[1], max_x+pad)
    min_y, max_y = max(0, min_y-pad), min(img_array.shape[0], max_y+pad)
    
    # Crop to digit
    img_array = img_array[min_y:max_y, min_x:max_x]
    img = Image.fromarray(img_array)
    
    # Resize to 20px while maintaining aspect ratio
    img.thumbnail((20, 20))
    
    # Center the digit in a 28x28 image
    new_img = Image.new('L', (28, 28), 0)
    new_img.paste(img, ((28-img.size[0])//2, (28-img.size[1])//2))
    
    # Convert to numpy array and normalize
    img_array = np.array(new_img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    try:
        # Load model and predict
        model = tf.keras.models.load_model('model.h5')
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Show the processed image
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(new_img, cmap='gray')
        plt.title("Processed Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.bar(range(10), prediction[0])
        plt.title(f"Prediction: {digit}\nConfidence: {confidence:.2f}")
        plt.xticks(range(10))
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
        
        print(f"Predicted Digit: {digit} with confidence {confidence:.2f}")
        
    except Exception as e:
        print(f"Error loading/predicting with model: {e}")

# --- Create the canvas ---
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.2)
ax.set_xlim(0, 28)
ax.set_ylim(0, 28)
ax.set_facecolor('white')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Draw a digit (0-9)")

# --- Add buttons ---
axclear = plt.axes([0.7, 0.05, 0.1, 0.075])
axsave = plt.axes([0.81, 0.05, 0.1, 0.075])
btn_clear = Button(axclear, 'Clear')
btn_clear.on_clicked(clear_canvas)

btn_save = Button(axsave, 'Predict')
btn_save.on_clicked(save_and_predict)

# --- Bind mouse events ---
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()