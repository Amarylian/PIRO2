import sys
import image_processing
import points_transformations
import descriptor
import info
import numpy as np


def main(name):
    set = name
    if len(sys.argv) > 1:
        set = sys.argv[1]
    imgs = image_processing.read_images(set)

    points = points_transformations.generate_points(imgs, set)

    descriptors = []
    color_descriptors = []
    color_diff_descriptors = []
    descript = []

    for i in range(len(imgs)):
        if imgs[i] is None:
            break
        img = image_processing.draw_points(imgs[i],points[i])
        image_processing.show(img,"Image "+str(i))
        desc, refs, colors, colors_diff, descr = descriptor.extract(imgs[i],points[i])
        info.show("-------------------- Image "+str(i)+"---------------------")
        info.show(desc)
        descriptors.append(desc)
        color_descriptors.append(colors)
        color_diff_descriptors.append(colors_diff)
        descript.append(descr)
        #img = image_processing.draw_points_descriptors(imgs[i], refs)
        #image_processing.show(img,"Image "+str(i))

    auces1 = []
    auces2 = []
    auces3 = []

    for i in range(len(color_diff_descriptors)):
        for j in range(len(color_diff_descriptors)):
            if i != j:
                auc1 = descriptor.distance2(color_diff_descriptors[i], color_diff_descriptors[j])
                auces1.append(auc1)

    for i in range(len(color_descriptors)):
        for j in range(len(color_descriptors)):
            if i != j:
                auc2 = descriptor.distance2(color_descriptors[i], color_descriptors[j])
                auces2.append(auc2)

    for i in range(len(descript)):
        for j in range(len(descript)):
            if i != j:
                auc3 = descriptor.distance2(descript[i], descript[j])
                auces3.append(auc3)

    m = np.mean(auces1)
    m2 = np.mean(auces2)
    m3 = np.mean(auces3)
    print(m, m2, m3)

    return m, m2, m3


def main2():
    l = []
    l2 = []
    l3 = []
    names = ["bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall"]

    for n in names:
        for i in range(10):
            m1, m2, m3 = main(n)
            l.append(m1)
            l2.append(m2)
            l3.append(m3)
    print(np.mean(l))
    print(np.mean(l2))
    print(np.mean(l3))


if __name__ == "__main__":
    main2()
    #main("wall")
