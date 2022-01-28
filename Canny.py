import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from skimage import color, io





# Wczytanie zdjęcia
image = color.rgb2gray(color.rgba2rgb(io.imread('CT_lungs.png')))

# Generowanie kwadratu
size_y = 256
size_x = 256
rectangle = np.zeros((size_y, size_x))
rectangle[76:180, 76:180] = 1
rectangle = rectangle + np.random.randn(size_y, size_x)*0.02

# plt.figure(dpi=150)
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.imshow(rectangle, cmap='gray')
# plt.axis('off')
# plt.show()



#Algorytm Canny


def gaussian_smoothing(image, sigma):
    return nd.gaussian_filter(image, sigma)

def calculate_gradients(image):
    gradient_y, gradient_x = np.gradient(image)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)   #długosc strzalki

    gradient_angle = np.arctan2(np.abs(gradient_y), np.abs(gradient_x))   #kierunek strzłki
    return gradient_magnitude, gradient_angle

def non_maximum_supression(gradient_magnitude, gradient_angle): #wyznacza potencjalne krawędzie
    size = gradient_magnitude.shape
    potential_edges = np.zeros_like(gradient_magnitude)

    angle = gradient_angle + 180

    for i in range(1, size[0]-1):
        for j in range(1, size[1]-1):

            # 0
            if 0 <= angle[i, j] < 22.5 or 337.5 <= angle[i, j] <= 360 or 157.5 <= angle[i, j] < 180:
                before = gradient_magnitude[i, j+1] # prawo lewo porownuje
                after = gradient_magnitude[i, j-1]

            # 45
            elif 22.5 <= angle[i, j] < 67.5:
                before = gradient_magnitude[i+1, j-1] # po przekatnej
                after = gradient_magnitude[i-1, j+1]

            # 90
            elif 67.5 <= angle[i, j] < 112.5:
                before = gradient_magnitude[i+1, j] # gora dol
                after = gradient_magnitude[i-1, j]

            # 135
            else:
                before = gradient_magnitude[i-1, j-1] # po przekatnej
                after = gradient_magnitude[i+1, j+1]

            if gradient_magnitude[i, j] >= before and gradient_magnitude[i, j] >= after:
                potential_edges[i, j] = gradient_magnitude[i, j]

    return potential_edges



def otsu_threshold(image):


    return

def double_threshold(potential_edges):  # bez magnitudy

    HT = potential_edges.max() * 0.12
    LT = HT * 0.07

    strong_edges = 1
    weak_edges = 0.1
    double = np.zeros(potential_edges.shape)

    st = np.where(potential_edges >= HT)
    we = np.where((potential_edges <= HT) & (potential_edges >= LT))


    double[st] = strong_edges
    double[we] = weak_edges
    return double, strong_edges, weak_edges

def edge_hysteresis(strong_edges, weak_edges, double):
    for i in range(1, double.shape[0]-1):
        for j in range(1, double.shape[1]-1):
            if (double[i, j] == weak_edges):
                if ((double[i + 1, j + 1] == strong_edges) or (double[i + 1, j - 1] == strong_edges) or (double[i + 1, j] == strong_edges) or (double[i, j + 1] == strong_edges)
                    or (double[i, j - 1] == strong_edges) or (double[i - 1, j + 1] == strong_edges) or (double[i - 1, j - 1] == strong_edges) or (double[i - 1, j] == strong_edges)):
                    double[i, j] = strong_edges
                else:
                    double[i, j] = 0
    return double



def canny(image, sigma=1.0):

    smooth_img = gaussian_smoothing(image, sigma)
    mag, ang = calculate_gradients(smooth_img)
    image2 = non_maximum_supression(mag, ang)
    # threshold = otsu_threshold(image)
    double, strong, weak = double_threshold(image2)
    img_out = edge_hysteresis(strong, weak, double)

    return img_out

plt.figure(dpi=150)
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(canny(image), cmap='gray')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(rectangle, cmap='gray')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(canny(rectangle), cmap='gray')
plt.axis('off')
plt.show()





#OTSU

def otsu_threshold(image, nbins=256):
    hist, bins = np.histogram(image.ravel(), nbins)          #lista ze wszytskimi wartosciami, wylicza wairancje, liczy dla kazdego kroku dwie wariancje, pozniej jak sie je doda to mamy dwa zbiory pikseli 50% czarnych 50% białych
    result = np.zeros(image.shape)

    weight = np.cumsum(hist)                                #otsu binaryzuje
    weight_inversed = np.cumsum(hist[::-1])[::-1]

    mean = np.cumsum(hist * bins[:-1]) / weight
    mean_inversed = (np.cumsum((hist * bins[:-1])) / weight_inversed[::-1])[::-1]

    variance = weight[:-1] * weight_inversed[1:] * (mean[:-1] - mean_inversed[1:]) ** 2
    index_max = np.argmax(variance)
    threshold = bins[:-1][index_max]

    result[image >= threshold] = 1
    return result, threshold