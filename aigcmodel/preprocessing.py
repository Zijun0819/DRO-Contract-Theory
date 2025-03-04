import os
import random
import re
import shutil

from aigcmodel.evaluate import parse_args_and_config


def get_sample_eval_txt_file():
    data_folder = 'data\\low_light'
    sample_file_path = "data\\Sample_#200.txt"
    eval_file_path = "data\\Eval_#50.txt"
    sample_file_names = list()
    eval_file_names = list()

    Equipment_Strings = ["Bulldozer_", "Crane_", "Excavator_", "Truck_"]

    for equipment in Equipment_Strings:
        random.seed(42)
        inflation = 2 if equipment == "Excavator_" else 1

        equipment_images = [
            f for f in os.listdir(data_folder)
            if f.startswith(equipment) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # if len(equipment_images) < 50:
        #     raise ValueError(f"Not enough images with the prefix {equipment} to select 50.")

        selected_images = random.sample(equipment_images, 50*inflation)
        sample_file_names += selected_images[:40*inflation]
        eval_file_names += selected_images[40*inflation:]

    with open(sample_file_path, 'w') as file:
        for img_name in sample_file_names:
            file.write(os.path.join("data\\low_light", img_name) + '\n')

    with open(eval_file_path, 'w') as file:
        for img_name in eval_file_names:
            file.write(os.path.join("data\\low_light", img_name) + '\n')


# Copy images as per the designated txt file
def copy_files(config):
    src_dir = "data\\normal_light"
    copied_file_name = config.data.eval_dir
    dst_dir = config.data.copy_dir
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    with open(copied_file_name, "r") as file:
        lines = file.readlines()  # 逐行读取文件内容
        images_path= [line.strip() for line in lines]  # 去除换行符并生成列表

    for img_path in images_path:
        # 确保是文件而非目录
        if os.path.isfile(img_path):
            img_path = img_path.replace("\n", "")
            img_name = re.split("\\\\", img_path)[-1]
            src_path = os.path.join(src_dir, img_name)
            dst_path = os.path.join(dst_dir, img_name)

            # 复制文件
            shutil.copy(src_path, dst_path)
            print(f"Copied '{src_path}' to '{dst_path}'")


if __name__ == '__main__':
    args, config = parse_args_and_config()
    get_sample_eval_txt_file()
    copy_files(config=config)
