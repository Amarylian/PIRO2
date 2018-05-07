import sys
import image_processing
import points_transformations
import descriptor
import info


def main():
    set = "bikes"
    if len(sys.argv) > 1:
        set = sys.argv[1]
    imgs = image_processing.read_images(set)

    points = points_transformations.generate_points(imgs, set)

    descriptors = []

    for i in range(len(imgs)):
        if imgs[i] is None:
            break
        img = image_processing.draw_points(imgs[i],points[i])
        image_processing.show(img,"Image "+str(i))
        desc, refs = descriptor.extract(imgs[i],points[i])
        info.show("-------------------- Image "+str(i)+"---------------------")
        info.show(desc)
        descriptors.append(desc)
        img = image_processing.draw_points_descriptors(imgs[i], refs)
        image_processing.show(img,"Image "+str(i))


    desc1 = descriptors[0][0]
    desc2 = descriptors[1][0]

    info.show(desc1)
    info.show(descriptor.normalize_descriptor(desc1))



if __name__ == "__main__":
    main()