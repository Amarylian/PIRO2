import sys
import image_processing
import points_transformations
import descriptor
import info
import numpy as np


def main(name, file):
    set = name
    if len(sys.argv) > 1:
        set = sys.argv[1]
    imgs = image_processing.read_images(set, file)

    points = points_transformations.generate_points(imgs, set)

    descriptors = []
    color_descriptors = []

    for i in range(len(imgs)):
        if imgs[i] is None:
            break
        #img = image_processing.draw_points(imgs[i],points[i])
        #image_processing.show(img,"Image "+str(i))
        desc, refs, colors = descriptor.extract(imgs[i],points[i])
        #info.show("-------------------- Image "+str(i)+"---------------------")
        #info.show(desc)
        descriptors.append(desc)
        color_descriptors.append(colors)
        #img = image_processing.draw_points_descriptors(imgs[i], refs)
        #image_processing.show(img,"Image "+str(i))

    auces2 = []

    for i in range(len(color_descriptors)):
        for j in range(len(color_descriptors)):
            if i != j:
                auc2 = descriptor.distance2(color_descriptors[i], color_descriptors[j], file)
                auces2.append(auc2)

    m2 = np.mean(auces2)
    print(m2)
    file.write(str(m2) + "\n")
    return m2


def main2():
    file = open("results.txt", "w")
    l2 = []
    names = ["bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall"]

    for n in names:
        for i in range(10):
            m2 = main(n, file)
            l2.append(m2)
    m = np.mean(l2)
    print(m)
    file.write(str(m) + "\n")
    file.close()


if __name__ == "__main__":
    main2()
    #main("bark")
