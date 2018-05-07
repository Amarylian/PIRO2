import sys
import image_processing
import info

import random

path = "imgs/"
transform_name_format = "H1to%sp"


def transform_point(point, transformation):
    lb_x = transformation[0][0] * point[0] + transformation[0][1] * point[1] + transformation[0][2]
    lb_y = transformation[1][0] * point[0] + transformation[1][1] * point[1] + transformation[1][2]
    lb = transformation[2][0] * point[0] + transformation[2][1] * point[1] + transformation[2][2]
    return [int(lb_x/lb), int(lb_y/lb)]


def transform_points(points, set, nr):
    transformation = read_transformation(set, nr)
    trans_points = [transform_point(point, transformation) for point in points]
    return trans_points


def read_transformation(set, nr):
    file_name = transform_name_format % nr
    f = open(path+set+"/"+file_name)
    t = [[float(num) for num in line.split()] for line in f]
    f.close()
    return t


def random_points(shape, number = 6):
    points = []
    for i in range(number):
        x = random.randint(32,shape[1] - 32)
        y = random.randint(32, shape[0] - 32)
        points.append([x,y])
    return points

def generate_points(imgs, set):
    if imgs[0] is None:
        exit(-1)

    shape = imgs[0].shape
    points = []
    points.append(random_points(shape))

    for i in range(1,len(imgs)):
        points.append(transform_points(points[0],set,i+1))

    return points

def main():
    set = "bikes"
    if len(sys.argv) > 1:
        set = sys.argv[1]
    imgs = image_processing.read_images(set)

    points = generate_points(imgs, set)

    for i in range(len(imgs)):
        if imgs[i] is None:
            break
        img = image_processing.draw_points(imgs[i],points[i])
        image_processing.show(img,"Image "+str(i))


if __name__ == "__main__":
    main()
