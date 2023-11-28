import glob
def get_images_list(logger):
    data_path = '/cluster/home/taheeraa/datasets/chestxray-14'
    image_files_list = []
    for i in range(1, 13):
        folder_name = f'{data_path}/images_{i:03}'
        files_in_subfolder = glob.glob(f'{folder_name}/images/*')
        image_files_list.extend(files_in_subfolder)
    logger.info(f"Image paths: {len(image_files_list)}")
    return image_files_list