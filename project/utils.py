import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# --------------------------------------------------------------------------------------------------------------

def clean_nested_columns(df):
    # Rename the column with the original typo
    if 'trianTestLabel' in df.columns:
        df.rename(columns={'trianTestLabel': 'trainTestLabel'}, inplace=True)
    
    # Inner function to extract and format text
    def extract_text(value):
        while isinstance(value, (np.ndarray, list)):
            if len(value) == 0:
                return "unknown"
            value = value[0]
        
        # Return the clean string in lowercase (e.g., 'none', 'training', 'center')
        return str(value).lower()
    
    # Apply cleaning to label columns
    columns_to_fix = ['trainTestLabel', 'failureType']
    
    for col in columns_to_fix:
        if col in df.columns:
            df[col] = df[col].apply(extract_text)
            
    return df

# --------------------------------------------------------------------------------------------------------------

def resize_wafer_maps(df, target_shape=(56, 56)):
    print(f"Ridimensionamento delle mappe dei wafer a {target_shape}...")
    
    resized_maps = []
    
    # Usiamo tqdm per vedere la barra di avanzamento
    for i in tqdm(range(len(df))):
        img = df['waferMap'].iloc[i]
        
        # Ridimensioniamo la matrice. 
        # order=0 indica un'interpolazione "nearest-neighbor", fondamentale per dati discreti 
        # (0, 1, 2) perché evita di creare sfumature con numeri decimali.
        img_resized = resize(img, target_shape, order=0, preserve_range=True, anti_aliasing=False)
        
        resized_maps.append(img_resized.astype(np.uint8))
        
    # Creiamo un unico grande array NumPy di forma (N_immagini, Altezza, Larghezza)
    X = np.array(resized_maps)
    
    # Aggiungiamo la dimensione del canale (1 per scala di grigi) richiesta dalle CNN
    X = np.expand_dims(X, axis=-1)
    
    return X

# --------------------------------------------------------------------------------------------------------------

def build_optimized_model(use_he=False, use_l2=False, optimizer='adam'):
    """
    Builds a flexible CNN model where specific optimizations can be toggled on or off.
    (Note: Data Augmentation is handled EXTERNALLY via ImageDataGenerator for DirectML compatibility).
    """
    INPUT_SHAPE = (56, 56, 1)
    NUM_CLASSES = 9
    
    model = Sequential(name="Flexible_Optimized_CNN")
        
    # --- TO-DO 2: He Initialization ---
    if use_he:
        print("-> Applying He Initialization")
        initializer = 'he_normal'
    else:
        initializer = 'glorot_uniform' # Default Keras initializer (Xavier)
        
    # --- TO-DO 4: L2 Regularization (Weight Decay) ---
    if use_l2:
        print("-> Applying L2 Regularization")
        regularizer = l2(0.001)
    else:
        regularizer = None

    # --- Block 1: Feature Extraction ---
    # Since augmentation is now handled externally, we ALWAYS need to define input_shape here
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                     kernel_initializer=initializer, kernel_regularizer=regularizer, 
                     input_shape=INPUT_SHAPE, name="conv_layer_1"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                     kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv_layer_2"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                     kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv_layer_3"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # --- Block 2: Classification ---
    model.add(Flatten())

    model.add(Dense(128, activation='relu', 
                    kernel_initializer=initializer, kernel_regularizer=regularizer))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # --- TO-DO 1: Compare Optimizers ---
    if isinstance(optimizer, str):
        opt = SGD(learning_rate=0.01, momentum=0.9) if optimizer == 'sgd_momentum' \
          else Adam(learning_rate=0.001)
    else:
        opt = optimizer  # oggetto passato direttamente dal training loop

    # Compile the Model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --------------------------------------------------------------------------------------------------------------

def plot_training_history(history_obj, experiment_name=""):
    print(f"Plotting learning curves for {experiment_name}...")
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_obj.history['accuracy'], label='Train Accuracy', color='teal', linewidth=2)
    plt.plot(history_obj.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
    plt.title(f'Model Accuracy - {experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot Loss (Error)
    plt.subplot(1, 2, 2)
    plt.plot(history_obj.history['loss'], label='Train Loss', color='teal', linewidth=2)
    plt.plot(history_obj.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title(f'Model Loss - {experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------------------------------------------

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
# --------------------------------------------------------------------------------------------------------------

def add_bar_labels(ax, bars):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f'{bar.get_height():.1f}%',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')

# --------------------------------------------------------------------------------------------------------------

def visualize_original_and_maps(model, input_image, label_index, class_names):
    # 1. Configurazione del modello per estrarre gli output di ogni layer Conv2D
    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(input_image[np.newaxis, ...], verbose=0)
    layer_names = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    
    # 2. Visualizza immagine originale con il nome della classe
    plt.figure(figsize=(2, 2))
    plt.imshow(input_image.reshape(56, 56), cmap='gray')
    
    # Recuperiamo il nome dalla lista usando l'indice
    label_name = class_names[label_index] 
    plt.title(f"Original\nClass: {label_name}")
    plt.axis('off')
    plt.show()
    
# 3. Visualizza feature maps per ogni strato
    for layer_name, layer_activation in zip(layer_names, activations):
        n_filters = layer_activation.shape[-1]
        cols = 16
        rows = math.ceil(n_filters / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), squeeze=False)
        axes_flat = axes.flatten()
        fig.suptitle(f'Layer: {layer_name}  ({n_filters} filters)', fontsize=12)

        for i in range(n_filters):
            axes_flat[i].imshow(layer_activation[0, :, :, i], cmap='viridis')
            axes_flat[i].axis('off')

        for i in range(n_filters, len(axes_flat)):
            axes_flat[i].axis('off')

        plt.tight_layout()
        plt.show()

# --------------------------------------------------------------------------------------------------------------

def build_light_model(use_he=False, use_l2=False, optimizer='adam'):
    """
    Lightweight version of build_optimized_model with half the filters/units per layer.
    Data Augmentation is handled EXTERNALLY via ImageDataGenerator.
    """
    INPUT_SHAPE = (56, 56, 1)
    NUM_CLASSES = 9

    model = Sequential(name="Light_CNN")

    # --- He Initialization ---
    if use_he:
        print("-> Applying He Initialization")
        initializer = 'he_normal'
    else:
        initializer = 'glorot_uniform'

    # --- L2 Regularization ---
    if use_l2:
        print("-> Applying L2 Regularization")
        regularizer = l2(0.001)
    else:
        regularizer = None

    # --- Block 1: Feature Extraction ---
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same',   # era 32
                     kernel_initializer=initializer, kernel_regularizer=regularizer,
                     input_shape=INPUT_SHAPE, name="conv_layer_1"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',   # era 64
                     kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv_layer_2"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',   # era 128
                     kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv_layer_3"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # --- Block 2: Classification ---
    model.add(Flatten())

    model.add(Dense(64, activation='relu',                                         # era 128
                    kernel_initializer=initializer, kernel_regularizer=regularizer))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # --- Optimizer ---
    if isinstance(optimizer, str):
        opt = SGD(learning_rate=0.01, momentum=0.9) if optimizer == 'sgd_momentum' \
          else Adam(learning_rate=0.001)
    else:
        opt = optimizer

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --------------------------------------------------------------------------------------------------------------

def build_ultra_light_model(use_he=False, use_l2=False, optimizer='adam'):
    """
    Versione Ultra-Light: Filtri ridotti (8-16-32) e Global Average Pooling.
    Risparmio di oltre il 90% dei parametri rispetto al modello con Flatten().
    """
    INPUT_SHAPE = (56, 56, 1)
    NUM_CLASSES = 9

    model = Sequential(name="Ultra_Light_CNN")

    # --- Inizializzazione ---
    if use_he:
        print("-> Applying He Initialization")
        initializer = 'he_normal'
    else:
        initializer = 'glorot_uniform'

    # --- Regolarizzazione ---
    if use_l2:
        print("-> Applying L2 Regularization")
        regularizer = l2(0.001)
    else:
        regularizer = None

    # --- Block 1: Feature Extraction ---
    # Ridotto da 16 a 8 filtri
    model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same',
                     kernel_initializer=initializer, kernel_regularizer=regularizer,
                     input_shape=INPUT_SHAPE, name="conv_layer_1"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Ridotto da 32 a 16 filtri
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same',
                     kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv_layer_2"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Ridotto da 64 a 32 filtri
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                     kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv_layer_3"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # --- Block 2: Classification (LA VERA RIVOLUZIONE) ---
    # Invece di appiattire (Flatten) e creare centinaia di migliaia di connessioni,
    # facciamo semplicemente la media spaziale di ogni singola feature map.
    model.add(GlobalAveragePooling2D(name="gap_layer"))
    
    # Opzionale: un po' di dropout per sicurezza, anche se il GAP è già un forte regolarizzatore
    model.add(Dropout(0.3)) 

    # Connettiamo direttamente i 32 valori medi alle 9 classi di output
    model.add(Dense(NUM_CLASSES, activation='softmax', name="output_layer"))

    # --- Optimizer ---
    if isinstance(optimizer, str):
        opt = SGD(learning_rate=0.01, momentum=0.9) if optimizer == 'sgd_momentum' \
          else Adam(learning_rate=0.001)
    else:
        opt = optimizer

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# --------------------------------------------------------------------------------------------------------------

def statistical_analysis(df):
    print("--- General Information ---")
    print(f"Dataset shape: {df.shape}")
    print(f"Null values per column:\n{df.isnull().sum()}\n")
    
    print("--- Class Distribution (failureType) ---")
    distribution = df['failureType'].value_counts()
    print(distribution)

    # Bar chart for defect distribution with percentages
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        y=distribution.index, 
        x=distribution.values, 
        color='steelblue'
    )
    
    # Calculate and add percentage labels
    total = distribution.sum()
    percentages = [f" {(value / total) * 100:.1f}%" for value in distribution.values]
    ax.bar_label(ax.containers[0], labels=percentages, label_type='edge', padding=3)

    plt.title("Failure Type Distribution")
    plt.xlabel("Count")
    plt.ylabel("Defect Type")
    
    # Expand x-axis limit slightly to make room for the percentage text
    plt.xlim(0, max(distribution.values) * 1.15) 
    
    plt.tight_layout()
    plt.show()

    # Histogram for die size distribution (if available)
    if 'dieSize' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df['dieSize'], bins=50, kde=True, color='teal')
        plt.title("Die Size Distribution")
        plt.xlabel("Size")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
        
# --------------------------------------------------------------------------------------------------------------

def visualize_wafer_samples(df):
    classes = df['failureType'].unique()
    n_classes = len(classes)
    
    cols = 3
    rows = int(np.ceil(n_classes / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.ravel()
    
    for i, cls in enumerate(classes):
        # Extract the first available wafer map for the current class
        sample = df[df['failureType'] == cls].iloc[0]
        
        axes[i].imshow(sample['waferMap'], cmap='inferno')
        axes[i].set_title(f"{cls}")
        axes[i].axis('off')
        
    # Hide excess empty axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------------------------------------------

def is_valid_label(label):
    # Handle nested lists or numpy arrays
    while isinstance(label, (np.ndarray, list)):
        if len(label) == 0:
            return False  # Discard empty arrays []
        label = label[0]
    
    # Convert to lowercase string to be safe
    label_str = str(label).strip().lower()
    
    # Return True only if it's explicitly training or test
    return label_str in ['training', 'test']
