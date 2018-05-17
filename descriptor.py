import math
import image_processing
import info
import numpy as np
import random
import operator
from sklearn.metrics import roc_auc_score

distance_start = 3 #10
distance_step = 1 #3
number_of_sections = 360
number_of_subsections = 8

angle_step = math.pi / number_of_sections

radius = 32


def extract(image, keypoints):
    descriptors = []
    references = []
    colors = []
    color_diff = []
    descript = []
    for point in keypoints:
        desc, ref, col, col_diff = get_descriptor(image, point)
        descr = get_descriptor2(image, point)
        descriptors.append(desc)
        references.append(ref)
        colors.append(col)
        color_diff.append(col_diff)
        descript.append(descr)
    return descriptors, references, colors, color_diff, descript


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
    color_diff = []

    if reference_color is None:
        for i in range(number_of_sections):
            tmp_descriptor = []
            tmp_points = []
            tmp_colors = []
            tmp_color_diff = []
            for j in range(number_of_subsections):
                tmp_descriptor.append(0)
                tmp_points.append([(-1, -1), 0])
                tmp_colors.append(-1)
                tmp_color_diff.append(255)
            descriptor.append(tmp_descriptor)
            reference_points.append(tmp_points)
            colors.append(tmp_colors)
            color_diff.append(tmp_color_diff)
        return descriptor, reference_points, colors, color_diff

    for i in range(number_of_sections):  # iterowanie po kącie
        angle = i*angle_step
        sin = math.sin(2*angle)
        cos = math.cos(2*angle)

        tmp_points = [[tuple(point),0]]
        tmp_descriptor = []
        tmp_colors = []
        tmp_color_diff = []

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
                #tmp_color_diff.append(max(abs(reference_color - 255), abs(reference_color)))
                break

            new_trend = np.sign(checking_point_color - reference_color)  # 1 jeśli jaśniejszy, -1 jeśli ciemniejszy, 0 taki sam
            if new_trend*last_trend < 0:  # nastąpiło odwrócenie tendencji
                tmp_descriptor.append((distance - distance_step)*last_trend)
                tmp_points.append([tuple(prev_point), last_trend])
                tmp_colors.append(checking_point_color)
                tmp_color_diff.append(checking_point_color - reference_color)

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
        color_diff.append(tmp_color_diff)

    return descriptor, reference_points, colors, color_diff


# def normalize_descriptor(descriptor):
#     norm = [(section / max([abs(num) for num in section])) for section in descriptor]
#     incremental = [np.sign(norm[i])*(abs(norm[i]) - sum([abs(j) for j in norm[i]])) for i in range(len(norm))]
#     return incremental


def distance2(descriptor1, descriptor2):
    desc = dict()
    for i in range(len(descriptor1)):
        for j in range(len(descriptor2)):
            desc[(i, j)] = distance(descriptor1[i], descriptor2[j])
            #print("(", i, ",", j, ") : ", desc[(i, j)])
    #print("\n\n")

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
    return auc


def distance(descriptor1, descriptor2):
    line_descriptor1 = []
    for i in range(len(descriptor1)):
        line_descriptor1.append(np.mean(descriptor1[i]))
    s1 = np.mean(line_descriptor1)
    max_s1 = max(line_descriptor1)

    line_descriptor2 = []
    for i in range(len(descriptor2)):
        line_descriptor2.append(np.mean(descriptor2[i]))
    s2 = np.mean(line_descriptor2)
    max_s2 = max(line_descriptor2)
    max_s = max(max_s1, max_s2)

    if math.isnan(max_s):
        max_s = 255

    s = abs(s1 - s2)
    s = round(s, 2)
    if math.isnan(s):
        s = max_s
    return s
