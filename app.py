import streamlit as st
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from streamlit_drawable_canvas import st_canvas


# --- Configuration ---
DATA_DIR = 'handdata' # Directory containing your digit folders (0, 1, ..., 9)
IMAGE_HEIGHT = 28     # Desired height for resized images
IMAGE_WIDTH = 28      # Desired width for resized images
NUM_CLASSES = 10      # For digits 0-9
MODEL_SAVE_PATH = 'handwritten_cnn_model.keras' # Changed to .keras for TensorFlow 2.x native format


# --- Functions ---
@st.cache_data # Cache the data loading so it runs only once
def load_and_preprocess_data(data_directory, img_height, img_width, num_classes):
    """
    Loads images and labels from a directory structure (e.g., class_id/image.png),
    resizes, preprocesses, and returns them for CNN training.
    Includes robust error handling and informative Streamlit messages.
    """
    images = []
    labels = []
    
    st.info(f"Attempting to load data from: '{os.path.abspath(data_directory)}'")

    # Validate data directory and find class folders
    try:
        class_dirs = sorted([d for d in os.listdir(data_directory)
                             if os.path.isdir(os.path.join(data_directory, d)) and d.isdigit()])
    except FileNotFoundError:
        st.error(f"❌ Error: Data directory '{os.path.abspath(data_directory)}' not found. Check `DATA_DIR` path.")
        return np.array([]), np.array([])
    except Exception as e:
        st.error(f"❌ Error accessing directory '{os.path.abspath(data_directory)}': {e}")
        return np.array([]), np.array([])

    if not class_dirs:
        st.error(f"❌ Error: No digit class directories (0-9) found in '{os.path.abspath(data_directory)}'.")
        return np.array([]), np.array([])

    st.info(f"✅ Found {len(class_dirs)} class directories: {', '.join(class_dirs)}")
    
    progress_bar = st.progress(0)
    total_images_loaded = 0
    
    # Iterate through class directories to load images
    for i, class_name in enumerate(class_dirs):
        class_path = os.path.join(data_directory, class_name)
        label = int(class_name)
        
        # Update progress bar and status
        progress_bar.progress((i + 1) / len(class_dirs), text=f"Loading images from class: {label}...")
        
        class_images_count = 0
        found_image_in_class = False

        for image_filename in os.listdir(class_path):
            # Only process common image file extensions
            if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(class_path, image_filename)
                
                try:
                    # Read image as grayscale
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        # Resize image to consistent dimensions
                        img = cv2.resize(img, (img_width, img_height))
                        images.append(img)
                        labels.append(label)
                        class_images_count += 1
                        total_images_loaded += 1
                        found_image_in_class = True
                    else:
                        st.warning(f"⚠️ Skipped: `cv2.imread` returned None for {image_path} (corrupted/unreadable).")
                except Exception as e:
                    st.warning(f"⚠️ Error reading image {image_path}: {e}")
        
        if not found_image_in_class:
            st.warning(f"⚠️ No readable image files found in class directory: '{class_path}'.")
        
        st.info(f"   Loaded {class_images_count} images for class {label}.")

    progress_bar.empty() # Clear the progress bar when done
    
    if total_images_loaded == 0:
        st.error("❌ Final Check: No images were loaded at all! Verify your dataset directory, its subfolders (0-9), and ensure actual readable image files are present.")
        return np.array([]), np.array([])

    st.success(f"✅ Successfully loaded {total_images_loaded} images.")
    images_np = np.array(images)
    labels_np = np.array(labels)

    st.write(f"**Raw image data shape after loading:** `{images_np.shape}` (Expected: `(num_images, height, width)`)")
    st.write(f"**Raw label data shape after loading:** `{labels_np.shape}` (Expected: `(num_images,)`)")

    
    # --- Debugging Raw Image Data (crucial for checking if images are black at source) ---
    st.subheader("Debugging Raw Image Data (before normalization)")
    if images_np.shape[0] > 0:
        st.write(f"Raw images data type: {images_np.dtype}")
        st.write(f"Raw images min pixel value: {np.min(images_np)}")
        st.write(f"Raw images max pixel value: {np.max(images_np)}")
        st.image(images_np[0], caption="Raw First Image Loaded", width=80)
    else:
        st.warning("No images loaded to display raw data.")
    st.markdown("---") # Separator

    # Preprocessing: Normalize and Reshape
    
    X = images_np.astype(np.float32) / 255.0 # Normalize to [0, 1]
    X = X.reshape(-1, img_height, img_width, 1) # Add channel dimension for grayscale CNN input
    y = tf.keras.utils.to_categorical(labels_np, num_classes=num_classes) # One-hot encode labels
    
    st.write(f"**Normalized and reshaped image data for CNN input:** `{X.shape}`")
    st.write(f"**One-hot encoded labels shape:** `{y.shape}`")
    
    # Check for single class problem
    if y.size > 0:
        y_labels_int = np.argmax(y, axis=1)
        unique_classes = np.unique(y_labels_int)
        if len(unique_classes) < num_classes:
            st.warning(f"⚠️ Warning: Loaded data contains only {len(unique_classes)} unique classes (expected {num_classes}). Detected classes: {unique_classes}.")
            if len(unique_classes) == 1:
                st.warning("⚠️ Only one class detected! `stratify` will be removed for data splitting.")

    return X, y # X_data represents your entire collection of preprocessed images


# Build Convolution Neural Network (CNN) - Define once globally
def build_cnn_model(img_height, img_width, num_classes):
    """
    Builds and returns the CNN model.
    """
    model = Sequential([
        # Change to sigmoid  / increase matrix shapes like 5 x 5  or 10 x 10
        Conv2D(32, (5, 5), activation='relu', input_shape=(img_height, img_width, 1)), # Detect simple patterns (edges, corners) # Change to sigmoid  
        BatchNormalization(),
        MaxPooling2D((2, 2)), 

        Conv2D(64, (3, 3), activation='relu'), # Detect combinations of patterns (shapes, curves)
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'), # Detect high-level features (digits, strokes, complex patterns)
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),  # Convert 3D feaures map into 1D vector

        Dense(128, activation='relu'),  # Fully connected layer with 128 neurons.
        Dropout(0.5),  # Prevents overfitting during training.
        Dense(num_classes, activation='softmax') # softmax = turns output into probabilities, so the model predicts the most likely digit.
    ])
    return model


# --- StreamlitCallback Class (as provided) ---
class StreamlitCallback(tf.keras.callbacks.Callback):
    """
    Keras Callback to update Streamlit components during training.
    """
    def __init__(self, metrics_placeholder, progress_bar_placeholder, total_epochs, plot_placeholder):
        self.metrics_placeholder = metrics_placeholder
        self.progress_bar_placeholder = progress_bar_placeholder
        self.total_epochs = total_epochs
        self.plot_placeholder = plot_placeholder # New placeholder for the plot
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            if key in self.history:
                self.history[key].append(value)

        # Update progress bar
        self.progress_bar_placeholder.progress((epoch + 1) / self.total_epochs)

        # Display current metrics
        self.metrics_placeholder.markdown(
            f"**Epoch {epoch + 1}/{self.total_epochs}:** "
            f"Loss: {logs.get('loss'):.4f}, Acc: {logs.get('accuracy'):.4f} | "
            f"Val Loss: {logs.get('val_loss'):.4f}, Val Acc: {logs.get('val_accuracy'):.4f}"
        )

        # Plot training history
        if len(self.history['accuracy']) > 0: # Plot after first epoch
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(self.history['accuracy'], label='Training Accuracy')
            ax1.plot(self.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Accuracy over Epochs')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()

            ax2.plot(self.history['loss'], label='Training Loss')
            ax2.plot(self.history['val_loss'], label='Validation Loss')
            ax2.set_title('Loss over Epochs')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()

            # Display the plot in its dedicated placeholder
            self.plot_placeholder.pyplot(fig)
            plt.close(fig) # Close the figure to prevent memory leaks


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Handwritten Digit CNN Trainer")

st.title("🔢 Handwritten Digit Classifier with CNN")

# --- SESSION STATE INITIALIZATION ---
# Initialize all session state variables at the beginning of the script
# This prevents 'AttributeError: 'NoneType' object has no attribute 'pop''
# because all keys are guaranteed to exist.
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'X' not in st.session_state:
    st.session_state['X'] = np.array([])
if 'y' not in st.session_state:
    st.session_state['y'] = np.array([])
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = np.array([])
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = np.array([])
if 'X_val' not in st.session_state:
    st.session_state['X_val'] = np.array([])
if 'y_val' not in st.session_state:
    st.session_state['y_val'] = np.array([])
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = np.array([])
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = np.array([])
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'training_history' not in st.session_state:
    st.session_state['training_history'] = None
if 'training_plot_figure' not in st.session_state: # Initialize this one too
    st.session_state['training_plot_figure'] = None
if 'epochs' not in st.session_state: # Initialize sidebar controls
    st.session_state['epochs'] = 10
if 'batch_size' not in st.session_state: # Initialize sidebar controls
    st.session_state['batch_size'] = 64
# --- END SESSION STATE INITIALIZATION ---

# Sidebar for controls
st.sidebar.header("Configuration")
st.session_state['epochs'] = st.sidebar.slider("Number of Epochs", min_value=1, max_value=50, value=st.session_state['epochs'])
st.session_state['batch_size'] = st.sidebar.slider("Batch Size", min_value=16, max_value=256, value=st.session_state['batch_size'], step=16)

# --- Main tabs for application sections ---
# CORRECTED LINE: Using four top-level tabs
load_data_tab, train_tab, predict_upload_tab, predict_draw_tab = st.tabs(["📂 Load Data", "📊 Train & Evaluate", "✍️ Predict (Upload)", "✍️ Predict (Draw)"])


# --- Load Data Tab ---
with load_data_tab:
    st.header("1. Load & Preprocess Data")
    st.write(f"Loading images from: `{DATA_DIR}`. Images will be resized to `{IMAGE_HEIGHT}x{IMAGE_WIDTH}`.")

    if st.button("Load & Preprocess Custom Data", key="load_preprocess_data_button"):
        with st.spinner("Loading and preprocessing data... This might take a while."):
            X_data, y_data = load_and_preprocess_data(DATA_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES)

            if X_data.size == 0:
                st.error("Data loading failed. Please resolve the issues reported above.")
                st.session_state['data_loaded'] = False
            else:
                st.session_state['data_loaded'] = True
                st.session_state['X'] = X_data
                st.session_state['y'] = y_data

                y_labels_int = np.argmax(y_data, axis=1)
                unique_classes = np.unique(y_labels_int)

                use_stratify = len(unique_classes) > 1 and np.all(np.bincount(y_labels_int) > 1)

                if not use_stratify:
                    st.warning("⚠️ Warning: Not enough unique classes or samples per class for stratified splitting. Using non-stratified split.")
                    st.session_state['X_train'], X_temp, st.session_state['y_train'], y_temp = \
                        train_test_split(X_data, y_data, test_size=0.2, random_state=42)
                    st.session_state['X_val'], st.session_state['X_test'], st.session_state['y_val'], st.session_state['y_test'] = \
                        train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
                else:
                    st.session_state['X_train'], X_temp, st.session_state['y_train'], y_temp = \
                        train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_labels_int)
                    st.session_state['X_val'], st.session_state['X_test'], st.session_state['y_val'], st.session_state['y_test'] = \
                        train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1))

                st.success(f"Data loaded and split into training, validation, and test sets.")

                # Display sample images
                st.subheader("Sample Images (First 5 from training set)")
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    with col:
                        if st.session_state['X_train'].shape[0] > i:
                            display_image = st.session_state['X_train'][i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
                            st.image(display_image,
                                    caption=f"Label: {np.argmax(st.session_state['y_train'][i])}",
                                    width=80)
                        else:
                            st.warning(f"Not enough training samples to display image {i+1}.")
    else:
        if not st.session_state['data_loaded']:
            st.info("Click 'Load & Preprocess Custom Data' to load your dataset.")

# --- Train & Evaluate Tab ---
with train_tab:
    st.header("2. Model Training & Evaluation")
    # ... (the rest of the training code from before) ...
    # This section remains unchanged
    if not st.session_state['data_loaded'] or st.session_state['X_train'].size == 0:
        st.warning("Please load the data first in the 'Load Data' tab.")
    else:
        epochs = st.session_state['epochs']
        batch_size = st.session_state['batch_size']

        if st.button("Train CNN Model", key="train_cnn_model_button"):
            st.session_state['model'] = build_cnn_model(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES)
            st.session_state['model'].compile(optimizer=Adam(learning_rate=0.0001), 
                                             loss='categorical_crossentropy',
                                             metrics=['accuracy'])
            st.subheader("Model Summary")
            model_summary_str = []
            st.session_state['model'].summary(print_fn=lambda x: model_summary_str.append(x))
            st.code("\n".join(model_summary_str))

            st.subheader("Training Progress")
            progress_bar_placeholder = st.empty()
            metrics_placeholder = st.empty()
            training_plot_placeholder = st.empty() 

            streamlit_callback = StreamlitCallback(metrics_placeholder, progress_bar_placeholder, epochs, training_plot_placeholder)

            with st.spinner("Training in progress..."):
                history = st.session_state['model'].fit(st.session_state['X_train'], st.session_state['y_train'],
                                                        epochs=epochs,
                                                        batch_size=batch_size,
                                                        validation_data=(st.session_state['X_val'], st.session_state['y_val']),
                                                        callbacks=[streamlit_callback],
                                                        verbose=0)
                st.session_state['training_history'] = history.history
                st.success("Training complete!")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(st.session_state['training_history']['accuracy'], label='Training Accuracy')
                ax1.plot(st.session_state['training_history']['val_accuracy'], label='Validation Accuracy')
                ax1.set_title('Accuracy over Epochs')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.legend()
                ax2.plot(st.session_state['training_history']['loss'], label='Training Loss')
                ax2.plot(st.session_state['training_history']['val_loss'], label='Validation Loss')
                ax2.set_title('Loss over Epochs')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.legend()
                st.session_state['training_plot_figure'] = fig
                plt.close(fig)

                st.session_state['model'].save(MODEL_SAVE_PATH)
                st.info(f"Model saved to: `{MODEL_SAVE_PATH}`")
        
        if st.session_state['training_plot_figure'] is not None:
            st.subheader("Final Training History")
            st.pyplot(st.session_state['training_plot_figure'])

        if st.session_state['model'] is not None:
            st.subheader("Model Evaluation")
            if st.button("Evaluate Model on Test Set", key="evaluate_model_button"):
                with st.spinner("Evaluating on test set..."):
                    if st.session_state['X_test'].size > 0 and st.session_state['y_test'].size > 0:
                        loss, accuracy = st.session_state['model'].evaluate(st.session_state['X_test'], st.session_state['y_test'], verbose=0)
                        st.metric("Test Loss", f"{loss:.4f}")
                        st.metric("Test Accuracy", f"{accuracy:.4f}")
                        st.success("Evaluation complete!")
                    else:
                        st.warning("Cannot evaluate: Test set is empty. Ensure data loaded correctly and contains multiple classes.")


# --- Predict (Upload) Tab ---
with predict_upload_tab:
    st.header("1. Predict from Uploaded Image")
    st.write("Upload a handwritten digit image to get a prediction.")

    # Only one place to load the model for all prediction tabs
    if os.path.exists(MODEL_SAVE_PATH):
        if st.button("Load Saved Model for Prediction", key="load_saved_model_for_prediction_button"):
            try:
                st.session_state['model'] = tf.keras.models.load_model(MODEL_SAVE_PATH)
                st.success(f"Model loaded successfully from `{MODEL_SAVE_PATH}`!")
                st.subheader("Loaded Model Summary")
                model_summary_str = []
                st.session_state['model'].summary(print_fn=lambda x: model_summary_str.append(x))
                st.code("\n".join(model_summary_str))
            except Exception as e:
                st.error(f"Error loading model: {e}")
    else:
        st.info(f"No saved model found at `{MODEL_SAVE_PATH}`. Please train a model in the 'Train & Evaluate' tab first.")

    st.subheader("Upload Handwritten Digit Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"], key="uploaded_image_file_uploader_predict")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            st.error("Could not load image. Please ensure it's a valid image file.")
        else:
            original_img_display = img.copy()
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            img_normalized = img / 255.0
            img_input = img_normalized.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)

            st.image(original_img_display, caption='Uploaded Image', use_container_width=True)
            st.write("Image preprocessed for model input.")

            if st.session_state['model'] is not None:
                prediction = st.session_state['model'].predict(img_input)
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0]) * 100

                st.success(f"**Predicted Digit: {predicted_class}** (Confidence: {confidence:.2f}%)")

                st.subheader("Prediction Probabilities")
                probabilities_df = tf.convert_to_tensor(prediction[0]).numpy()
                probabilities_df = np.round(probabilities_df, 4)

                fig, ax = plt.subplots(figsize=(8, 4))
                digits = [str(i) for i in range(NUM_CLASSES)]
                ax.bar(digits, probabilities_df)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Digit")
                ax.set_ylabel("Probability")
                ax.set_title("Predicted Probabilities for Each Digit")
                st.pyplot(fig)
                plt.close(fig)

                st.dataframe(pd.DataFrame({'Digit': digits, 'Probability': probabilities_df}))
            else:
                st.warning("Please train or load a model to make predictions!")


# --- Predict (Draw) Tab ---
with predict_draw_tab:
    st.header("2. Predict from Drawn Digit")
    

    # Always display the drawing canvas controls and the canvas itself
    st.subheader("Draw Your Digit Here")
    st.write("Draw a single digit (0-9) with a **white stroke on the black background**, similar to how MNIST digits are represented.")

    # Canvas controls
    col1, col2 = st.columns(2)
    with col1:
        stroke_width = st.slider("Stroke width:", 5, 25, 15, key="drawing_stroke_width")
    with col2:
        if st.button("Clear Canvas", key="clear_canvas_btn"):
            st.rerun()

    # Drawing canvas - simplified and more reliable configuration
    try:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",  # Transparent fill
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",  # White stroke
            background_color="#000000",  # Black background
            background_image=None,  # Explicitly set to None
            update_streamlit=True,  # Enable real-time updates
            height=280,  # Slightly larger for better drawing
            width=280,
            drawing_mode="freedraw",
            point_display_radius=0,  # Hide point indicators
            display_toolbar=True,  # Show drawing toolbar
            key="prediction_drawing_canvas",
        )
    except Exception as e:
        st.error(f"Error initializing canvas: {e}")
        st.info("Please make sure streamlit-drawable-canvas is properly installed: `pip install streamlit-drawable-canvas`")
        canvas_result = None

    # The prediction logic below will only run IF something is drawn and a model exists
    if canvas_result is not None and canvas_result.image_data is not None:
        # Convert RGBA to grayscale
        drawn_image_rgba = canvas_result.image_data.astype(np.uint8)
        
        # Check if anything was actually drawn
        if drawn_image_rgba.shape[2] == 4:  # RGBA
            # Convert to grayscale using the alpha channel or RGB
            drawn_image_gray = cv2.cvtColor(drawn_image_rgba, cv2.COLOR_RGBA2GRAY)
        else:  # RGB
            drawn_image_gray = cv2.cvtColor(drawn_image_rgba, cv2.COLOR_RGB2GRAY)

        # Check if there's actual drawing content (non-zero pixels)
        if np.max(drawn_image_gray) > 0:
            st.markdown("---")
            st.subheader("Processing Drawing for Prediction")
            
            # Show the current drawing
            st.image(drawn_image_gray, caption="Your Drawing", width=200)
            
            # Check if the model is loaded BEFORE attempting a prediction
            if 'model' not in st.session_state or st.session_state['model'] is None:
                st.warning("Please load a model first using the button above!")
            else:
                # Preprocess the image
                processed_drawn_image = cv2.resize(drawn_image_gray, (IMAGE_WIDTH, IMAGE_HEIGHT),
                                                    interpolation=cv2.INTER_AREA)
                processed_drawn_image = processed_drawn_image.astype(np.float32) / 255.0
                img_input_for_model = processed_drawn_image.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)

                # Show preprocessed image
                st.info("Your drawing after preprocessing (28x28, normalized):")
                fig_processed, ax_processed = plt.subplots(figsize=(3, 3))
                ax_processed.imshow(processed_drawn_image, cmap='gray')

                

                # Make prediction
                try:
                    prediction = st.session_state['model'].predict(img_input_for_model, verbose=0)
                    predicted_class = np.argmax(prediction[0])
                    confidence = np.max(prediction[0]) * 100

                    st.success(f"## Predicted Digit: **{predicted_class}** (Confidence: {confidence:.2f}%)")

                    # Show prediction probabilities
                    st.subheader("Prediction Probabilities")
                    probabilities_df = pd.DataFrame({
                        'Digit': [str(i) for i in range(NUM_CLASSES)],
                        'Probability': np.round(prediction[0], 4)
                    })

                    # Create probability bar chart
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = ax.bar(probabilities_df['Digit'], probabilities_df['Probability'], 
                                 color=['red' if i == predicted_class else 'blue' for i in range(NUM_CLASSES)])
                    ax.set_ylim(0, 1)
                    ax.set_xlabel("Digit")
                    ax.set_ylabel("Probability")
                    ax.set_title("Model's Confidence for Each Digit")
                    
                    # Highlight the predicted digit
                    bars[predicted_class].set_color('red')
                    
                    st.pyplot(fig)
                    plt.close(fig)

                    # Show probability table
                    st.dataframe(probabilities_df, hide_index=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
        else:
            st.info("👆 Draw something on the canvas above to see a prediction!")
    elif canvas_result is None:
        st.error("Canvas failed to initialize. Please refresh the page.")
    else:
        st.info("👆 Draw a digit on the canvas above!")

