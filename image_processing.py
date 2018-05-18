import cv2
import matplotlib.pyplot as plt
import info
import numpy as np

path = "imgs/"
img_name_format = "img%s"
formats = [".ppm", ".pgm"]


def read_image(set, nr):
    file = img_name_format % (nr)
    for format in formats:
        file_name = path + set + "/" + file + format
        img = cv2.imread(file_name)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            info.show("Loaded file " + file_name)
            break
        info.show("Can't load file " + file_name)

    return img


def draw_points(image, points, size=3, color=(0, 255, 0), text=True):
    img = gray_to_rgb(image)
    for i in range(len(points)):
        [x, y] = points[i]
        cv2.circle(img, (x, y), size, color)
        if text:
            cv2.putText(img, str(i), (x + 3, y + 3), 3, cv2.FONT_HERSHEY_PLAIN, color, thickness=2)
    return img


def draw_point(image, point, size=3, color=(0, 255, 0), text=None):
    img = image.copy()
    [x, y] = point
    cv2.circle(img, (x, y), size, color)
    if text is not None:
        cv2.putText(img, text, (x + 3, y + 3), 3, cv2.FONT_HERSHEY_PLAIN, color, thickness=2)
    return img


def draw_line(image, p1, p2, size=1, color=(0, 255, 255)):
    img = image.copy()
    cv2.line(img, p1, p2, color, size)
    return img


def gray_to_rgb(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def read_images(set):
    imgs = []
    for i in range(1, 7):
        img = read_image(set, i)
        imgs.append(img)
    return imgs


def show(image, title=None):
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()


def get_color(point, image):
    (x, y) = point
    size = 3
    h, w = image.shape

    if y < size or x < size or x > w - 1 - size or y > h - 1 - size:
        return None
    subimg = image[y - size:y + size + 1, x - size:x + size + 1]

    return np.mean(subimg)


def draw_points_descriptors(image, refs):
    img = gray_to_rgb(image)
    for ref in refs:
        img = draw_points_descriptor(img, ref)
    return img


def draw_points_descriptor(image, ref):
    img = image.copy()
    for section in ref:
        point = section[0][0]
        for next_reference in section[1:]:
            [next_point, trend] = next_reference
            color = (0, 255, 0) if trend > 0 else ((255, 0, 0) if trend < 0 else (0, 0, 255))
            img = draw_line(img, point, next_point, color=color, size=2)
            point = next_point
    return img


def get_normalized_histogram(image):
    m, n = image.shape
    hist = [0.0 for i in range(256)]
    for i in range(m):
        for j in range(n):
            hist[image[i, j]] += 1
    return np.array(hist) / (m * n)


def cumulative_sum(hist):
    return [sum(hist[:i + 1]) for i in range(len(hist))]


def image_preprocess(image):
    hist = get_normalized_histogram(image)
    cdf = np.array(cumulative_sum(hist))
    sk = np.uint8(255 * cdf)
    h, w = image.shape
    output_image = np.zeros((h, w))
    for i in range(0, h):
        for j in range(0, w):
            output_image[i, j] = sk[image[i, j]]
    return output_image
