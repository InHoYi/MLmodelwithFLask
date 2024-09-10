import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compute_svd(channel_data):
    U, S, Vt = np.linalg.svd(channel_data, full_matrices=False)
    return U, S, Vt

def compress_image(image_data, red_percentage=0.10, green_percentage=0.10, blue_percentage=0.10):
    red_data = image_data[:, :, 0]
    R_U, R_S, R_Vt = compute_svd(red_data)

    green_data = image_data[:, :, 1]
    G_U, G_S, G_Vt = compute_svd(green_data)

    blue_data = image_data[:, :, 2]
    B_U, B_S, B_Vt = compute_svd(blue_data)

    red_k = int(red_percentage * len(R_S))
    green_k = int(green_percentage * len(G_S))
    blue_k = int(blue_percentage * len(B_S))

    compressed_red = reconstruct_image(R_U, R_S, R_Vt, red_k)
    compressed_green = reconstruct_image(G_U, G_S, G_Vt, green_k)
    compressed_blue = reconstruct_image(B_U, B_S, B_Vt, blue_k)
    compressed_image = np.stack((compressed_red, compressed_green, compressed_blue), axis=-1)

    return compressed_image

def reconstruct_image(U, S, Vt, k):
    reconstructed_channel = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    reconstructed_channel = np.clip(reconstructed_channel, 0, 255)
    return reconstructed_channel.astype(np.uint8)

def SVD_picture_path(image_data, output_path='static/compressed_data.png'):
    # 배열을 이미지로 변환하여 저장
    img = Image.fromarray(image_data.astype('uint8'))
    img.save(output_path)
    return output_path