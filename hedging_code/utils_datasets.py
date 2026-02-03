import yfinance as yf
import pandas as pd
import os
from typing import List
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import matplotlib.pyplot as plt 
from scipy.stats import norm

def download_historical_prices(ticker, start =  "2000-01-01"):
    df = yf.download(ticker, start = start, interval='1d', auto_adjust= True)
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index(drop=True)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    return df

def generate_ohlc_image(df, include_volume=True, include_ma=True, n_days=20, target_size=None):
    
    assert(df.shape[0] == n_days)

    #we allocate 20% of the image space to the volume bar
    price_height = {5: 32, 20: 64, 60: 96}[n_days]  # height for price section
    volume_height = {5: 6, 20: 12, 60: 19}[n_days] if include_volume else 0  # height for volume section
    
    
    width = 3 * n_days
    height_total = price_height + volume_height
    #normalization for price
    p_ref = df.iloc[0]['Close']


    #normalization of the prices
    norm_df = df.copy()
    for col in ['Open', 'High', 'Low', 'Close', 'MA20']:
        norm_df[col] = norm_df[col] / p_ref 
    
    
    volume_max = norm_df['Volume'].max()
    #normalization of the volume

    norm_df['Volume'] = norm_df['Volume']/volume_max

    price_max = norm_df['High'].max()
    price_min = norm_df['Low'].min()

    price_range = price_max - price_min

    #convert price to coordinates
    def price_to_y(p):
        return int((price_max- p) / price_range * (price_height - 1))

    img = Image.new("RGB", (width, height_total), "black")
    draw = ImageDraw.Draw(img)

    #drowing of plots

    for i in range(n_days):
        abscisse = i * 3

        row = norm_df.iloc[i]

        y_high = price_to_y(row['High'])
        y_low = price_to_y(row['Low'])
        y_open = price_to_y(row['Open'])
        y_close = price_to_y(row['Close'])

        draw.line([(abscisse + 1, y_high), (abscisse + 1, y_low)], fill="white")

        # Draw open (left tick) and close (right tick)
        img.putpixel((abscisse, y_open), (255, 255, 255))
        img.putpixel((abscisse + 2, y_close), (255, 255, 255))

        #draw volume
        vol_height = int(row['Volume'] * (volume_height - 1))  # already normalized
        vol_height = max(vol_height, 1)   # scale volume
        vol_y0 = price_height + (volume_height - vol_height)  # top of volume bar

        for dx in [0]:
            x = abscisse + dx
            if 0 <= x < width:
                for y in range(vol_y0, price_height + volume_height):
                    img.putpixel((x, y), (255, 255, 255)) 

        if include_ma:
# Ensure x is inside image width
            for dx in range(3):
                x = abscisse - dx
                y = price_to_y(row['MA20'])

                if 0 <= x < width and 0 <= y < price_height:
                    img.putpixel((x, y), (255, 255, 255))
    if target_size is not None:
        img = img.resize(target_size, Image.BICUBIC) 
    return img


# def generate_ohlc_image_bis(df, include_volume=True, include_ma=True, n_days=20, target_size=None):
#     from PIL import Image, ImageDraw

#     assert(df.shape[0] == n_days)

#     pixels_per_day = 6  # Thickness
#     price_height = {5: 64, 20: 96, 60: 128}[n_days]
#     volume_height = {5: 12, 20: 18, 60: 24}[n_days] if include_volume else 0

#     width = pixels_per_day * n_days
#     height_total = price_height + volume_height

#     p_ref = df.iloc[0]['Close']
#     norm_df = df.copy()
#     for col in ['Open', 'High', 'Low', 'Close', 'MA20']:
#         norm_df[col] = norm_df[col] / p_ref

#     volume_max = norm_df['Volume'].max()
#     norm_df['Volume'] = norm_df['Volume'] / volume_max

#     price_max = norm_df['High'].max()
#     price_min = norm_df['Low'].min()
#     price_range = price_max - price_min

#     def price_to_y(p):
#         return int((price_max - p) / price_range * (price_height - 1))

#     img = Image.new("RGB", (width, height_total), "black")
#     draw = ImageDraw.Draw(img)

#     for i in range(n_days):
#         abscisse = i * pixels_per_day
#         row = norm_df.iloc[i]

#         y_high = price_to_y(row['High'])
#         y_low = price_to_y(row['Low'])
#         y_open = price_to_y(row['Open'])
#         y_close = price_to_y(row['Close'])

#         is_last_day = (i == n_days - 1)

#         if not is_last_day:
#             # Draw thicker high-low vertical line
#             for dx in range(pixels_per_day // 2 - 1, pixels_per_day // 2 + 1):
#                 if 0 <= abscisse + dx < width:
#                     draw.line([(abscisse + dx, y_high), (abscisse + dx, y_low)], fill="white", width=1)

#             # Draw thicker Close tick (right side)
#             for dx in range(2):
#                 for dy in range(3):
#                     x = abscisse + pixels_per_day - 2 + dx
#                     y = y_close + dy
#                     if 0 <= x < width and 0 <= y < price_height:
#                         img.putpixel((x, y), (255, 255, 255))

#             # Draw volume
#             if include_volume:
#                 vol_height = int(row['Volume'] * (volume_height - 1))
#                 vol_height = max(vol_height, 1)
#                 vol_y0 = price_height + (volume_height - vol_height)
#                 for dx in range(3):
#                     x = abscisse + dx
#                     if 0 <= x < width:
#                         for y in range(vol_y0, price_height + volume_height):
#                             img.putpixel((x, y), (255, 255, 255))

#             # Draw moving average
#             if include_ma:
#                 for dx in range(pixels_per_day):
#                     x = abscisse + dx
#                     y = price_to_y(row['MA20'])
#                     if 0 <= x < width and 0 <= y < price_height:
#                         img.putpixel((x, y), (255, 255, 255))

#         # Always draw Open tick (even for last day)
#         for dx in range(2):  # 2 pixels wide
#             for dy in range(3):  # 3 pixels tall
#                 x = abscisse + dx
#                 y = y_open + dy
#                 if 0 <= x < width and 0 <= y < price_height:
#                     img.putpixel((x, y), (255, 255, 255))

#     # Resize final image
#     if target_size is not None:
#         img = img.resize(target_size, Image.BICUBIC)

#     return img

def generate_ohlc_image_bis(df, include_volume=True, include_ma=True, n_days=20, target_size=None):
    

    assert(df.shape[0] == n_days)

    pixels_per_day = 6
    price_height = {5: 64, 20: 96, 60: 128}[n_days]
    volume_height = {5: 12, 20: 18, 60: 24}[n_days] if include_volume else 0
    width = pixels_per_day * n_days
    height_total = price_height + volume_height

    p_ref = df.iloc[0]['Close']
    norm_df = df.copy()
    for col in ['Open', 'High', 'Low', 'Close', 'MA20']:
        norm_df[col] = norm_df[col] / p_ref

    volume_max = norm_df['Volume'].max()
    norm_df['Volume'] = norm_df['Volume'] / volume_max

    price_max = norm_df['High'].max()
    price_min = norm_df['Low'].min()
    price_range = price_max - price_min

    def price_to_y(p):
        return int((price_max - p) / price_range * (price_height - 1))

    img = Image.new("RGB", (width, height_total), "black")
    draw = ImageDraw.Draw(img)

    for i in range(n_days):
        abscisse = i * pixels_per_day
        row = norm_df.iloc[i]

        y_high = price_to_y(row['High'])
        y_low = price_to_y(row['Low'])
        y_open = price_to_y(row['Open'])
        y_close = price_to_y(row['Close'])

        is_last_day = (i == n_days - 1)


        daily_return = (row['Close'] - row['Open']) / (row['Open'] + 1e-6)  
        daily_volatility = (row['High'] - row['Low']) / (row['Open'] + 1e-6)
        daily_volume = row['Volume']

        r = int(min(max((daily_return + 0.1) / 0.2, 0.0), 1.0) * 255)  
        g = int(min(max(daily_volatility / 0.1, 0.0), 1.0) * 255)       
        b = int(min(max(daily_volume, 0.0), 1.0) * 255)

        color = (r, g, b)

        if not is_last_day:
            
            for dx in range(pixels_per_day // 2 - 1, pixels_per_day // 2 + 1):
                if 0 <= abscisse + dx < width:
                    draw.line([(abscisse + dx, y_high), (abscisse + dx, y_low)], fill=color, width=1)

            
            for dx in range(2):
                for dy in range(3):
                    x = abscisse + pixels_per_day - 2 + dx
                    y = y_close + dy
                    if 0 <= x < width and 0 <= y < price_height:
                        img.putpixel((x, y), color)

           
            if include_volume:
                vol_height = int(row['Volume'] * (volume_height - 1))
                vol_height = max(vol_height, 1)
                vol_y0 = price_height + (volume_height - vol_height)
                for dx in range(3):
                    x = abscisse + dx
                    if 0 <= x < width:
                        for y in range(vol_y0, price_height + volume_height):
                            img.putpixel((x, y), color)

            if include_ma and i < n_days - 1:
                
                next_row = norm_df.iloc[i + 1]

                y_current = price_to_y(row['MA20'])
                y_next = price_to_y(next_row['MA20'])

                x_current = abscisse + pixels_per_day // 2
                x_next = (i + 1) * pixels_per_day + pixels_per_day // 2

                
                draw.line([(x_current, y_current), (x_next, y_next)], fill=color, width=1)

        
        for dx in range(2):
            for dy in range(3):
                x = abscisse + dx
                y = y_open + dy
                if 0 <= x < width and 0 <= y < price_height:
                    img.putpixel((x, y), color)

    if target_size is not None:
        img = img.resize(target_size, Image.BICUBIC)

    return img



def create_ticker_dataset(df, ticker, ma, tresh,  output_dir = "dataset", n_days = 20, target_size=(96, 96)):
    os.makedirs(output_dir, exist_ok=True)                     # Create main folder if it doesn't exist
    images_dir = os.path.join(output_dir, ticker, "images")            # Subfolder for images
    os.makedirs(images_dir, exist_ok=True)   

    y_labels: List[int] = []

    start_index = df['MA20'].first_valid_index() 

    for first_day in range(start_index, df.shape[0] - n_days -1):
        df_extract = df.iloc[first_day:first_day + n_days]

        image = generate_ohlc_image(df_extract, include_volume=True, include_ma=ma, n_days=n_days, target_size=target_size)
        y_labels.append(int(df_extract.iloc[-1]['Open'] * (1 + tresh) <  df_extract.iloc[-1]['Open']))

        image_path = os.path.join(images_dir, f"{first_day - n_days + 1}.png")
        image.save(image_path)


    labels_df = pd.DataFrame({"y": y_labels})
    label_output_dir = os.path.join(output_dir, ticker, "labels")
    os.makedirs(label_output_dir, exist_ok=True)

    # Save label file with row index as image ID
    labels_df.to_csv(os.path.join(label_output_dir, "labels.csv"), index_label="id")

    return f"✅ Saved {len(y_labels)} images and labels to '{output_dir}/'"


def create_ticker_dataset_bis(df, ticker, ma, tresh,  output_dir = "dataset_bis", n_days = 20, target_size=(96, 96)):
    os.makedirs(output_dir, exist_ok=True)                     # Create main folder if it doesn't exist
    images_dir = os.path.join(output_dir, ticker, "images")            # Subfolder for images
    os.makedirs(images_dir, exist_ok=True)   

    y_labels: List[int] = []

    start_index = df['MA20'].first_valid_index() 

    for first_day in range(start_index, df.shape[0] - n_days -1):
        df_extract = df.iloc[first_day:first_day + n_days]

        image = generate_ohlc_image_bis(df_extract, include_volume=True, include_ma=ma, n_days=n_days, target_size=target_size)
        y_labels.append(int(df_extract.iloc[-1]['Open'] * (1 + tresh) <  df.iloc[first_day + n_days]['Close']))

        image_path = os.path.join(images_dir, f"{first_day - n_days + 1}.png")
        image.save(image_path)


    labels_df = pd.DataFrame({"y": y_labels})
    label_output_dir = os.path.join(output_dir, ticker, "labels")
    os.makedirs(label_output_dir, exist_ok=True)

    # Save label file with row index as image ID
    labels_df.to_csv(os.path.join(label_output_dir, "labels.csv"), index_label="id")

    return f"✅ Saved {len(y_labels)} images and labels to '{output_dir}/'"



def create_data_set(ticker, window = 20, target_size=(96, 96), tresh = 0.005, ma = False):
    df_ticker = download_historical_prices(ticker)
    create_ticker_dataset(df_ticker, ticker, ma, n_days=window, target_size=target_size, tresh = tresh)

def create_data_set_bis(ticker, window = 20, target_size=(96, 96), tresh = 0.005, ma = False):
    df_ticker = download_historical_prices(ticker)
    create_ticker_dataset_bis(df_ticker, ticker, ma, n_days=window, target_size=target_size, tresh = tresh)

def load_all_ticker_data(dataset_dir):
    all_data = []

    for ticker in os.listdir(dataset_dir):
        ticker_path = os.path.join(dataset_dir, ticker)
        labels_path = os.path.join(ticker_path, "labels", "labels.csv")
        images_path = os.path.join(ticker_path, "images")

        if not os.path.exists(labels_path):
            continue

        df = pd.read_csv(labels_path)
       
        df['image_path'] = df.index.map(lambda x: os.path.join(images_path, f"{x}.png"))
        df['ticker'] = ticker
        all_data.append(df)

    # Combine all tickers into one DataFrame
    full_df = pd.concat(all_data).reset_index(drop=True)
    return full_df

def split_global_dataset(dataset_dir, test_ratio=0.15, val_ratio=0.15, seed=38):
    full_df = load_all_ticker_data(dataset_dir)

    # First, split off test set
    trainval_df, test_df = train_test_split(
        full_df,
        test_size=test_ratio,
        stratify=full_df['y'],
        random_state=seed
    )

    # Then, split train and validation
    val_size = val_ratio / (1 - test_ratio)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_size,
        stratify=trainval_df['y'],
        random_state=seed
    )

    # Print label distributions
    def print_distribution(name, df_):
        dist = df_['y'].value_counts().sort_index()
        total = len(df_)
        print(f"\n📊 {name.upper()} SET ({total} samples):")
        for label, count in dist.items():
            print(f"  Class {label}: {count} ({count / total:.2%})")

    print_distribution("train", train_df)
    print_distribution("val", val_df)
    print_distribution("test", test_df)

    return train_df, val_df, test_df


def dataframe_to_dataset(df, image_size= None):
    if image_size is None:
        images = np.array([
          img_to_array(load_img(path, color_mode='grayscale')) / 255.0
          for path in df['image_path']
      ])
    else:
      images = np.array([
          img_to_array(load_img(path, color_mode='grayscale', target_size=image_size)) / 255.0
          for path in df['image_path']
      ])
    labels = df['y'].values.astype(np.float32)
    return tf.data.Dataset.from_tensor_slices((images, labels))

def create_1d_dataframe(df):
    pass
    
def create_1d_dataset(ticker, n_days):
    df_ticker = download_historical_prices(ticker)


def preprocess_rgb(image, label):
    image = tf.image.grayscale_to_rgb(image)
    image = preprocess_input(image)  # very important!
    return image, label

def dataframe_to_dataset_color(df, image_size=(114,120)):
    # Create a tf.data.Dataset of (path, label) pairs
    paths = df['image_path'].values
    labels = df['y'].values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_and_preprocess_image(path, label):
        # Read image file
        image = tf.io.read_file(path)
        # Decode image (automatic RGB)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        # Resize image
        image = tf.image.resize(image, image_size)
        # Normalize
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Apply loading + preprocessing on-the-fly
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

import cv2

def superimpose_heatmap_on_image(original_image_path, heatmap, alpha=0.4):
    # Load the original image
    img = cv2.imread(original_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Normalize heatmap to 0-255
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply jet colormap
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Superimpose jet on the original image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, jet, alpha, 0)
    
    return superimposed_img, heatmap_resized


def plot_gradcam_side_by_side(image_path, heatmap, alpha=0.4):
    # Load original image
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Superimpose
    superimposed_img, heatmap_resized = superimpose_heatmap_on_image(image_path, heatmap, alpha=alpha)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Original Image
    axs[0].imshow(original_img)
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    # 2. Heatmap only + colorbar
    im = axs[1].imshow(heatmap_resized, cmap='jet', vmin=0, vmax=1)
    axs[1].axis('off')
    axs[1].set_title('Grad-CAM Heatmap')

    # Add colorbar next to the heatmap
    cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label('Attention intensity', rotation=270, labelpad=15)

    # 3. Superimposed
    axs[2].imshow(superimposed_img)
    axs[2].axis('off')
    axs[2].set_title('Superimposed on Original')

    plt.tight_layout()
    plt.show()


def plot_gradcam_side_by_side_with_label(image_path, heatmap, predicted_class, confidence, alpha=0.4):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Load original image
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Superimpose
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    axs[0].imshow(original_img)
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    im = axs[1].imshow(heatmap_resized, cmap='jet', vmin=0, vmax=1)
    axs[1].axis('off')
    axs[1].set_title('Grad-CAM Heatmap')

    axs[2].imshow(superimposed_img)
    axs[2].axis('off')
    axs[2].set_title('Superimposed Image')

    # Add colorbar for the heatmap
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04).set_label('Attention intensity', rotation=270, labelpad=15)

    # Set global title with prediction
    plt.suptitle(f'Predicted class: {predicted_class} (Confidence: {confidence:.2f})', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_gradcam_side_by_side_from_tensor(img_tensor, heatmap, predicted_class, confidence, alpha=0.4):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    # Prepare original image
    img = img_tensor.numpy()
    img = np.clip(img, 0, 1)  # Just to be sure

    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    img_uint8 = np.uint8(255 * img)
    superimposed_img = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    im = axs[1].imshow(heatmap_resized, cmap='jet', vmin=0, vmax=1)
    axs[1].axis('off')
    axs[1].set_title('Grad-CAM Heatmap')

    axs[2].imshow(superimposed_img)
    axs[2].axis('off')
    axs[2].set_title('Superimposed Image')

    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04).set_label('Attention intensity', rotation=270, labelpad=15)

    plt.suptitle(f'Predicted class: {predicted_class} (Confidence: {confidence:.2f})', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


