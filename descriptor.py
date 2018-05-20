import math
import image_processing
import info
import numpy as np
from sklearn.metrics import roc_auc_score

distance_start = 3 #10
distance_step = 1 #3
number_of_sections = 360
number_of_subsections = 8

angle_step = math.pi / number_of_sections

radius = 32

# trzeba zrobić żeby zwracał tylko jeden deskryptor
def extract(image, keypoints):
    image = image_processing.image_preprocess(image)
    descriptors = []
    references = []
    colors = []
    #descript = []
    for point in keypoints:
        desc, ref, col = get_descriptor(image, point)
        #descr = get_descriptor2(image, point)
        descriptors.append(desc)
        references.append(ref)
        colors.append(col)
        #descript.append(descr)
    return colors  # descriptors, references, colors, descript


# deskryptor okręgi od punktu do promienia 32 zczytywanie koloru
def get_descriptor2(image, point):
    angle_degree = math.pi / 360
    desc = []
    for r in range(distance_start, radius, distance_step):
        circle = []
        for i in range(360):
            angle = i * angle_degree
            sin = math.sin(2*angle)
            cos = math.cos(2*angle)
            checking_point = [int(r * sin) + point[0], (-1) * int(r * cos) + point[1]]
            checking_point_color = image_processing.get_color(checking_point, image)
            if checking_point_color is not None:
                circle.append(checking_point_color)
        desc.append(circle)
    return desc


def get_descriptor(image, point):
    reference_color = image_processing.get_color(point,image)
    descriptor = []
    reference_points = []
    colors = []

    if reference_color is None:
        for i in range(number_of_sections):
            tmp_descriptor = []
            tmp_points = []
            tmp_colors = []
            for j in range(number_of_subsections):
                tmp_descriptor.append(0)
                tmp_points.append([(-1, -1), 0])
                tmp_colors.append(-1)
            descriptor.append(tmp_descriptor)
            reference_points.append(tmp_points)
            colors.append(tmp_colors)
        return descriptor, reference_points, colors

    for i in range(number_of_sections):  # iterowanie po kącie
        angle = i*angle_step
        sin = math.sin(2*angle)
        cos = math.cos(2*angle)

        tmp_points = [[tuple(point),0]]
        tmp_descriptor = []
        tmp_colors = []

        distance = distance_start
        sign = 0
        last_trend = 0
        prev_point = point
        while len(tmp_descriptor) < number_of_subsections:  # dopóki nie ma danej liczby zmian, zwiększaj dystans
            checking_point = [int(distance*sin)+point[0], (-1)*int(distance*cos)+point[1]]  # punkt na okręgu oddalony o distance
            checking_point_color = image_processing.get_color(checking_point, image)

            if checking_point_color is None:  # wyszliśmy poza obraz
                tmp_descriptor.append((distance - distance_step)*last_trend)
                tmp_points.append([tuple(prev_point), last_trend])
                tmp_colors.append(reference_color)
                break

            new_trend = np.sign(checking_point_color - reference_color)  # 1 jeśli jaśniejszy, -1 jeśli ciemniejszy, 0 taki sam
            if new_trend*last_trend < 0:  # nastąpiło odwrócenie tendencji
                tmp_descriptor.append((distance - distance_step)*last_trend)
                tmp_points.append([tuple(prev_point), last_trend])
                tmp_colors.append(checking_point_color)

                last_trend = 0
                reference_color = checking_point_color
                continue

            if new_trend != 0:  # jeżeli nowy trend jest malejący lub rosnący
                last_trend = new_trend
            distance += distance_step
            prev_point = checking_point

            reference_color = checking_point_color  # model 2

        descriptor.append(tmp_descriptor)
        reference_points.append(tmp_points)
        colors.append(tmp_colors)

    return descriptor, reference_points, colors


# def normalize_descriptor(descriptor):
#     norm = [(section / max([abs(num) for num in section])) for section in descriptor]
#     incremental = [np.sign(norm[i])*(abs(norm[i]) - sum([abs(j) for j in norm[i]])) for i in range(len(norm))]
#     return incremental


# podliczanie auc dla deskryptorów dwóch obrazków
def distance2(descriptor1, descriptor2, file):
    desc = dict()
    for i in range(len(descriptor1)):
        for j in range(len(descriptor2)):
            desc[(i, j)] = distance(descriptor1[i], descriptor2[j])

    y_true = []
    y_scores = []
    for k, v in desc.items():
        if k[0] == k[1]:
            y_true.append(0)
        else:
            y_true.append(1)
        y_scores.append(v)

    auc = roc_auc_score(y_true, y_scores)
    print("AUC: ", auc)
    file.write("AUC: " + str(auc) + "\n")
    return auc


# różnica średnich kolorów w dwóch punktach
def distance(descriptor1, descriptor2):
    descriptor_in_line1 = []
    for i in range(len(descriptor1)):
        for j in range(len(descriptor1[i])):
            descriptor_in_line1.append(descriptor1[i][j])

    if len(descriptor_in_line1) == 0:
        s1 = 0
        max_s1 = 0
    else:
        s1 = np.mean(descriptor_in_line1)
        max_s1 = max(descriptor_in_line1)

    descriptor_in_line2 = []
    for i in range(len(descriptor2)):
        for j in range(len(descriptor2[i])):
            descriptor_in_line2.append(descriptor2[i][j])

    if len(descriptor_in_line2) == 0:
        s2 = 0
        max_s2 = 0
    else:
        s2 = np.mean(descriptor_in_line2)
        max_s2 = max(descriptor_in_line2)

    max_s = max(max_s1, max_s2)

    if math.isnan(max_s):
        max_s = 255

    s = abs(s1 - s2)

    if math.isnan(s):
        s = max_s
    return s
