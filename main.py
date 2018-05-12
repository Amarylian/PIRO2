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

    for i in range(len(imgs)):
        if imgs[i] is None:
            break
        img = image_processing.draw_points(imgs[i],points[i])
        image_processing.show(img,"Image "+str(i))
        desc, refs, colors, colors_diff = descriptor.extract(imgs[i],points[i])
        info.show("-------------------- Image "+str(i)+"---------------------")
        info.show(desc)
        descriptors.append(desc)
        color_descriptors.append(colors)
        color_diff_descriptors.append(colors_diff)
        img = image_processing.draw_points_descriptors(imgs[i], refs)
        image_processing.show(img,"Image "+str(i))


    #desc1 = descriptors[0][0]
    #desc2 = descriptors[1][0]
    #
    # normalized_descriptors = []
    #
    # for i in range(len(descriptors)):
    #     descript = []
    #     for j in range(len(descriptors[i])):
    #         info.show(descriptors[i][j])
    #         descript.append(descriptor.normalize_descriptor(descriptors[i][j]))
    #         info.show(descript[j])
    #     normalized_descriptors.append(descript)
    # print(normalized_descriptors)
    #
    # for i in range(1, len(normalized_descriptors)):
    #     descriptor.distance(normalized_descriptors[0], normalized_descriptors[i])

    auces = []

    for i in range(1, len(color_diff_descriptors)):
        auc = descriptor.distance2(color_diff_descriptors[0], color_diff_descriptors[i])
        auces.append(auc)

    m = np.mean(auces)
    print(m)

    return m


def main2():
    l = []
    names = ["bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall"]

    for n in names:
        for i in range(10):
            l.append(main(n))
    print(np.mean(l))


if __name__ == "__main__":
    #main2()
    main("bark")
