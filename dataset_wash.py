import os
import glob

object_types = ['sphere', 'cylinder', 'nut', 'bolt', 'spanner']  # tbc
root_dir = 'dataset'


# watch and delete bad images
# seems to be more straightforward inside windows...
def watch_images(object_type = 'spanner', img_type = 'raw'):
    # object_path_name = os.path.join(root_dir, object_type, img_type)
    # img_names = glob.glob(object_path_name)
    # for name in img_names:
        # img_path =  os.path.join(object_path_name, name)
        # img = mpimg.imread(name)
        # plt.imshow(img)
    pass


def rename_images_for_one_type(object_type = 'spanner', img_type = 'raw'):
    object_path_name = os.path.join(root_dir, object_type, img_type)
    img_names = glob.glob(os.path.join(object_path_name, '*'))
    ctr = 0
    for name in img_names:
        os.rename(name, os.path.join(object_path_name, 'raw_' + str(ctr).zfill(3) + '.png'))
        ctr += 1


def rename_images(img_type = 'raw'):
    for object_type in object_types:
        rename_images_for_one_type(object_type = object_type, img_type=img_type)


if __name__ == '__main__':
    rename_images()


