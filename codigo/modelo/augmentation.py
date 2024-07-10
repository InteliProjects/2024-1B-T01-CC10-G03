import numpy as np
from scipy import ndimage
from skimage import feature, color
import math
import random

'''
Todas as funções recebem um numpy array com multiplas imagens
e um numpy array com suas respectivas labels, aplica os devidos
filtros/processamentos e retorna um numpy array de mesmo tamanho,
porém com quantidade aleatória de imagens/labels (média 50%),
onde os arrays possuem SOMENTE as imagens modificadas. O usuário deve chamar as funções,
guardar cada uma em variáveis, e depois unir todas as imagens utilizando numpy.concatenate((arr, arrN, ...), axis=0).
'''

# Espelha as imagens, utilize axis=0 para espelhar horizontalmente e axis=1 para espelhar verticalmente

def flip(images, labels, axis):
    
    result_images = np.empty_like(images, dtype=images.dtype, shape=[len(images), images.shape[1], images.shape[2], images.shape[3]])
    result_labels = np.empty_like(labels, dtype=labels.dtype, shape=[len(labels), labels.shape[1], labels.shape[2]])

    match axis:
        case 0:
            for index, value in enumerate(images):
                result_images[index] = np.fliplr(images[value])
                result_labels[index] = np.fliplr(labels[value])
        case 1:
            for index, value in enumerate(images):
                result_images[index] = np.flipud(images[value])
                result_labels[index] = np.flipud(labels[value])
        case _:
            raise Exception("Wrong axis, use 0 for x-axis or 1 for y-axis")
    return result_images, result_labels

# Rotaciona as imagens no sentido anti-horário, utilize degrees=90/180/270 para girar uma, duas ou três vezes, respectivamente

def rotate(images, labels, degrees):
    if ((degrees > 270) or not (degrees % 90 == 0)):
            raise Exception("Invalid degrees. Use 90, 180 or 270")
    result_images = np.empty_like(images, dtype=images.dtype, shape=[len(images), images.shape[1], images.shape[2], images.shape[3]])
    result_labels = np.empty_like(labels, dtype=labels.dtype, shape=[len(labels), labels.shape[1], labels.shape[2]])
    for index, value in enumerate(images):
        result_images[index] = np.rot90(images[value], (degrees/90))
        result_labels[index] = np.rot90(labels[value], (degrees/90))
    return result_images, result_labels

# Aplica o filtro gaussiano na imagem (borra), o sigma mínimo para evitar erros é 1, e quanto maior ele for, mais borrado a imagem ficará

def gauss(images, labels, sigma):
    result_images = np.empty_like(images, dtype=images.dtype, shape=[len(images), images.shape[1], images.shape[2], images.shape[3]])
    result_labels = np.empty_like(labels, dtype=labels.dtype, shape=[len(labels), labels.shape[1], labels.shape[2]])
    for index, value in enumerate(images):
        result_images[index] = ndimage.gaussian_filter(images[value], sigma=sigma)
        result_labels[index] = labels[value]
    return result_images, result_labels

# Vale lembrar que o corte ficara size x size, logo, a altura e largura serão os mesmos.

def cut(images, labels, size):
    cropped_images = []
    cropped_labels = []
    for image_data in images:
        # Extrair as dimensões da imagem atual
        height, width, _ = image_data.shape

        # Calculando o número de mini-imagens em cada direção
        num_mini_images_h = height // size
        num_mini_images_w = width // size

        # Loop para cortar a imagem grande em várias mini-imagens
        for i in range(num_mini_images_h):
            for j in range(num_mini_images_w):
                start_h = i * size
                start_w = j * size
                end_h = start_h + size
                end_w = start_w + size

                # Aplicando o corte apenas nas dimensões espaciais (altura e largura)
                croped_img = image_data[start_h:end_h, start_w:end_w, :]

                cropped_images.append(croped_img)

    for label in labels:
        height, width = label.shape

        # Calculando o número de mini-imagens em cada direção
        num_mini_images_h = height // size
        num_mini_images_w = width // size

        # Loop para cortar a imagem grande em várias mini-imagens
        for i in range(num_mini_images_h):
            for j in range(num_mini_images_w):
                start_h = i * size
                start_w = j * size
                end_h = start_h + size
                end_w = start_w + size

                # Aplicando o corte apenas nas dimensões espaciais (altura e largura)
                cropped_label = label[start_h:end_h, start_w:end_w]

                cropped_labels.append(cropped_label)

    # Converter a lista de imagens recortadas em um array numpy
    return np.array(cropped_images), np.array(cropped_labels)

def augmentate(images, labels, cutSize=0, flipPercentual=2, rotatePercentual=2, gaussPercentual=2):
    if cutSize == 0:
        print("ERROR: cutSize cant be 0")
        return images, labels

    cut_img, cut_lbl = cut(images, labels, cutSize)

    # O flip pega algumas das imagens cortadas aleatoriamente, vira elas e retorna as imagens e suas respectivas labels.
    # NOTA: O flip só retorna as imagens flipadas, e sua quantidade é aleatória.
    flip0_img, flip0_lbl = ag.flip(cut_img, cut_lbl, axis=0)
    flip1_img, flip1_lbl = ag.flip(cut_img, cut_lbl, axis=1)
    rotate90_img, rotate90_lbl = ag.rotate(cut_img, cut_lbl, degrees=90)
    rotate180_img, rotate180_lbl = ag.rotate(cut_img, cut_lbl, degrees=180)
    rotate270_img, rotate270_lbl = ag.rotate(cut_img, cut_lbl, degrees=270)
    gauss_img, gauss_lbl = ag.gauss(cut_img, cut_lbl, sigma=1)
    # Junção das imagens e labels cortadas com as viradas.
    # AVISO: Não esqueça o axis = 0, se não ele vai juntar tudo como se fosse uma array 1-D.
    cropped_images = np.concatenate((cut_img, flip0_img, flip1_img, rotate90_img, rotate180_img, rotate270_img, gauss_img), axis=0)
    cropped_labels = np.concatenate((cut_lbl, flip0_lbl, flip1_lbl, rotate90_lbl, rotate180_lbl, rotate270_lbl, gauss_lbl), axis=0)

    return cut_img, cut_lbl

def sample_and_apply_augmentation(aug_func, imgs, lbls, percentage, **kwargs):
    num_samples = max(1, int(len(imgs) * percentage / 100))
    sampled_indices = random.sample(range(len(imgs)), num_samples)
    sampled_imgs = imgs[sampled_indices]
    sampled_lbls = lbls[sampled_indices]
    augmented_imgs, augmented_lbls = aug_func(sampled_imgs, sampled_lbls, **kwargs)
    return augmented_imgs, augmented_lbls, num_samples

def augmentate(images, labels, cutSize=0, flipPercentual=2, rotatePercentual=2, gaussPercentual=2):
    if cutSize == 0:
        print("ERROR: cutSize cant be 0")
        return images, labels
    
    cut_img, cut_lbl = cut(images, labels, cutSize)

    flip0_img, flip0_lbl, num_flip0 = sample_and_apply_augmentation(flip, cut_img, cut_lbl, flipPercentual, axis=0)
    flip1_img, flip1_lbl, num_flip1 = sample_and_apply_augmentation(flip, cut_img, cut_lbl, flipPercentual, axis=1)
    rotate90_img, rotate90_lbl, num_rotate90 = sample_and_apply_augmentation(rotate, cut_img, cut_lbl, rotatePercentual, degrees=90)
    rotate180_img, rotate180_lbl, num_rotate180 = sample_and_apply_augmentation(rotate, cut_img, cut_lbl, rotatePercentual, degrees=180)
    rotate270_img, rotate270_lbl, num_rotate270 = sample_and_apply_augmentation(rotate, cut_img, cut_lbl, rotatePercentual, degrees=270)
    gauss_img, gauss_lbl, num_gauss = sample_and_apply_augmentation(gauss, cut_img, cut_lbl, gaussPercentual, sigma=1)

    cropped_images = np.concatenate((cut_img, flip0_img, flip1_img, rotate90_img, rotate180_img, rotate270_img, gauss_img), axis=0)
    cropped_labels = np.concatenate((cut_lbl, flip0_lbl, flip1_lbl, rotate90_lbl, rotate180_lbl, rotate270_lbl, gauss_lbl), axis=0)

    # Imprimir o número de imagens modificadas por cada técnica
    print(f"Número de imagens flip horizontal: {num_flip0}")
    print(f"Número de imagens flip vertical: {num_flip1}")
    print(f"Número de imagens rotacionadas 90 graus: {num_rotate90}")
    print(f"Número de imagens rotacionadas 180 graus: {num_rotate180}")
    print(f"Número de imagens rotacionadas 270 graus: {num_rotate270}")
    print(f"Número de imagens com ruído gaussiano: {num_gauss}")

    return cropped_images, cropped_labels