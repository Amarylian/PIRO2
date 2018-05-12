import math
import image_processing
import info
import numpy as np
import random
import operator
from sklearn.metrics import roc_auc_score

distance_start = 10
distance_step = 3
number_of_sections = 8
number_of_subsections = 8

angle_step = math.pi / number_of_sections


def extract(image, keypoints):
    descriptors = []
    references = []
    colors = []
    color_diff = []
    for point in keypoints:
        desc, ref, col, col_diff = get_descriptor(image, point)
        descriptors.append(desc)
        references.append(ref)
        colors.append(col)
        color_diff.append(col_diff)
    return descriptors, references, colors, color_diff


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
                tmp_color_diff.append(max(abs(reference_color - 255), abs(reference_color)))
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


def normalize_descriptor(descriptor):
    norm = [(section / max([abs(num) for num in section])) for section in descriptor]
    incremental = [np.sign(norm[i])*(abs(norm[i]) - sum([abs(j) for j in norm[i]])) for i in range(len(norm))]
    return incremental


def compare_lines(lines):
    scores = []

    for line in lines:
        score_diff = 0
        score_sign = 0
        for t in line:
            score_diff += t[0]
            score_sign += t[1]
        scores.append((score_diff, score_sign))

    max_index, max_val = max(enumerate(scores), key=operator.itemgetter(1))
    maxes_score = [i for i, j in enumerate(scores) if j[1] == max_val[1]]
    new_scores = []
    for i in range(len(maxes_score)):
        new_scores.append((scores[maxes_score[i]], maxes_score[i]))

    if len(maxes_score) > 1:
        min_index, min_val = min(enumerate(new_scores), key=operator.itemgetter(0))
        min_score = [j[1] for i, j in enumerate(new_scores) if j[0] == min_val[0]]

        if len(min_score) > 1:
            return random.choice(min_score)
        else:
            return min_score[0]
    else:
        return maxes_score[0]


def get_line_score(line1, line2):
    scores = []
    if len(line1) <= len(line2):
        for n in range(len(line1)):
            sign = 0
            if (line1[n] < 0 and line2[n] < 0) or (
                    line1[n] >= 0 and line2[n] >= 0):
                sign = 1
            diff = abs(line1[n] - line2[n])
            scores.append((diff, sign))
    else:
        for n in range(len(line2)):
            sign = 0
            if (line1[n] < 0 and line2[n] < 0) or (
                    line1[n] >= 0 and line2[n] >= 0):
                sign = 1
            diff = abs(line1[n] - line2[n])
            scores.append((diff, sign))
    return scores


def distance(descriptor1, descriptor2):
    for i in range(len(descriptor1)):
        random_line = random.randint(0, len(descriptor1[i]) - 1)
        all_points = []
        for j in range(len(descriptor2)):
            point = []
            for k in range(len(descriptor2[j])):
                line = get_line_score(descriptor1[i][random_line], descriptor2[j][k])
                point.append(line)
            line_index = compare_lines(point)

            lines_in_point = dict()
            for k in range(len(descriptor1[i])):
                k1 = (k + random_line) % len(descriptor1[i])
                k2 = (k + line_index) % len(descriptor2[j])
                lines_in_point[(k1, k2)] = get_line_score(descriptor1[i][k1], descriptor2[j][k2])

            all_points.append(point)


def distance2(descriptor1, descriptor2):
    desc = dict()
    for i in range(len(descriptor1)):
        for j in range(len(descriptor2)):
            desc[(i, j)] = distance3(descriptor1[i], descriptor2[j])  # distance4(descriptor1[i], descriptor2[j])
            print("(", i, ",", j, ") : ", desc[(i, j)])

    # for k, v in desc.items():
    #     if v == 0:
    #         print(k, ": ", v)
    print("\n\n")

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


def distance3(descriptor1, descriptor2):
    line_descriptor1 = []
    line_desc1 = []
    l_desc1 = []
    for i in range(len(descriptor1)):
        line = [j ** 2 for j in descriptor1[i]]
        line_descriptor1.append(np.mean(line))
        line_desc1.append(np.mean(descriptor1[i]))
        l = [j / 255 for j in descriptor1[i]]
        l_desc1.append(np.mean(l))

    if len(descriptor1) == 0:
        score1 = -1
        sc1 = -1
        s1 = -1
    else:
        score1 = np.mean(line_descriptor1)
        score1 = score1 / (255 ** 2)
        sc1 = np.mean(line_desc1)
        s1 = np.mean(l_desc1)

    line_descriptor2 = []
    line_desc2 = []
    l_desc2 = []
    for i in range(len(descriptor2)):
        line = [j ** 2 for j in descriptor2[i]]
        line_descriptor2.append(np.mean(line))
        line_desc2.append(np.mean(descriptor2[i]))
        l = [j / 255 for j in descriptor2[i]]
        l_desc2.append(np.mean(l))

    if len(descriptor2) == 0:
        score2 = -1
        sc2 = -1
        s2 = -1
    else:
        score2 = np.mean(line_descriptor2)
        score2 = score2 / (255 ** 2)
        sc2 = np.mean(line_desc2)
        s2 = np.mean(l_desc2)

    s = abs(score1 - score2)
    #print("s1: ", score1, "s2: ", score2)
    #print("sc1: ", sc1, "sc2: ", sc2)

    res = [-1, -1, -1]

    if s < 0.001:
        res[0] = 0

    if abs(sc1 - sc2) < 1:
        res[1] = 0

    if abs(s1 - s2) < 0.001:
        res[2] = 0

    n = 0
    for i in range(len(res)):
        if res[i] == 0:
            n += 1

    if n > 1:
        return 0
    else:
        return s


def distance4(descriptor1, descriptor2):
    line_descriptor1 = []
    for i in range(len(descriptor1)):
        line_descriptor1.append(np.mean(descriptor1[i]))
    s1 = np.mean(line_descriptor1)

    line_descriptor2 = []
    for i in range(len(descriptor2)):
        line_descriptor2.append(np.mean(descriptor2[i]))
    s2 = np.mean(line_descriptor2)

    s = abs(s1 - s2)

    if s < 0.1:
        return 0
    return s
