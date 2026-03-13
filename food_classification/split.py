import os
import shutil
import random

# 类别名与编号的映射（请根据实际类别调整）
class_map = {
    'broke': '00',
    'circle': '01',
    'good': '02',
    'lose': '03',
    'uncovered': '04'
    # ...补全所有类别
}

src_dir = r'd:\第四五节_分类代码\food_classification\JPEGImages'
train_dir = r'd:\第四五节_分类代码\food_classification\JPEGImages\training\labeled'
val_dir = r'd:\第四五节_分类代码\food_classification\JPEGImages\validation'

split_ratio = 0.8  # 80%训练，20%验证

for class_name, class_id in class_map.items():
    src_class_dir = os.path.join(src_dir, class_name)
    train_class_dir = os.path.join(train_dir, class_id)
    val_class_dir = os.path.join(val_dir, class_id)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    images = [f for f in os.listdir(src_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    split = int(len(images) * split_ratio)
    train_imgs = images[:split]
    val_imgs = images[split:]

    for img in train_imgs:
        shutil.copy(os.path.join(src_class_dir, img), os.path.join(train_class_dir, img))
    for img in val_imgs:
        shutil.copy(os.path.join(src_class_dir, img), os.path.join(val_class_dir, img))

print("划分完成！")