import math
import image_processing
import info
import numpy as np

distance_start = 10
distance_step = 3
number_of_sections = 8
number_of_subsections = 8

angle_step = math.pi / number_of_sections

def extract(image, keypoints):
    descriptors = []
    references = []
    for point in keypoints:
        desc, ref = get_descriptor(image,point)
        descriptors.append(desc)
        references.append(ref)
    return descriptors, references


def get_descriptor(image, point):
    reference_color = image_processing.get_color(point,image)
    descriptor = []
    reference_points = []


    for i in range(number_of_sections): #iterowanie po kącie
        angle = i*angle_step
        sin = math.sin(2*angle)
        cos = math.cos(2*angle)

        tmp_points = [[tuple(point),0]]
        tmp_descriptor = []

        distance = distance_start
        sign = 0
        last_trend = 0
        prev_point = point
        while len(tmp_descriptor)<number_of_subsections: #dopóki nie ma danej liczby zmian, zwiększaj dystans
            checking_point = [int(distance*sin)+point[0], (-1)*int(distance*cos)+point[1]] #punkt na okręgu oddalony o distance
            checking_point_color = image_processing.get_color(checking_point,image)

            if checking_point_color is None: #wyszliśmy poza obraz
                tmp_descriptor.append((distance - distance_step)*last_trend)
                tmp_points.append([tuple(prev_point), last_trend])
                break

            new_trend = np.sign(checking_point_color - reference_color) #1 jeśli jaśniejszy, -1 jeśli ciemniejszy, 0 taki sam
            if new_trend*last_trend < 0: #nastąpiło odwrócenie tendencji
                tmp_descriptor.append((distance - distance_step)*last_trend)
                tmp_points.append([tuple(prev_point), last_trend])


                last_trend = 0
                reference_color = checking_point_color
                continue

            if new_trend != 0: #jeżeli nowy trend jest malejący lub rosnący
                last_trend = new_trend
            distance+=distance_step
            prev_point = checking_point

            reference_color = checking_point_color  # model 2

        descriptor.append(tmp_descriptor)
        reference_points.append(tmp_points)

    return descriptor, reference_points

def normalize_descriptor(descriptor):
    norm = [section / max([abs(num) for num in section]) for section in descriptor]
    incremental = [np.sign(norm[i])*(abs(norm[i]) - sum([abs(j) for j in norm[:i]])) for i in range(len(norm))]
    return incremental


def distance(descriptor1, descriptor2):
    pass

