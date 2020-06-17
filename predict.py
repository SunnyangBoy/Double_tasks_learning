import os
import torch
from model import DetectAngleModel
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import math
import torch.nn as nn
import cv2
import torchvision.models as models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_rotate_mat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    v = vertices.reshape((4,2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def rotate_img(img_dir, img_savepath, file_path, lab_savepath, flag):
    image = Image.open(img_dir)
    if flag:
        image = image.rotate(180, Image.BILINEAR)
    image.save(img_savepath)

    with open(lab_savepath, 'w') as writer:
        with open(file_path, 'r') as lines:
            lines = lines.readlines()
            for l, line in enumerate(lines):
                line = line.split(';')
                vertice = [int(vt) for vt in line[1:-1]]
                vertice = np.array(vertice)
                if flag:
                    center_x = (image.width - 1) / 2
                    center_y = (image.height - 1) / 2
                    new_vertice = np.zeros(vertice.shape)
                    new_vertice[:] = rotate_vertices(vertice, - math.pi, np.array([[center_x], [center_y]]))
                    vertice = new_vertice
                new_line = []
                new_line.append(line[0])
                for v in vertice:
                    new_line.append(str(int(v)))
                new_line.append(line[-1])
                new_line = ';'.join(new_line)
                writer.write(new_line)
        writer.close()


if __name__ == '__main__':

    # resnet18 = models.resnet18(pretrained=False)
    # model = DetectAngleModel(resnet18)
    model = DetectAngleModel()
    model.load_state_dict(torch.load('/home/ubuntu/cs/checks_recognize_v2/pths/rotate_pths/rotated_43_0.0071.pth'))

    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()

    imgs_dir = '/home/ubuntu/cs/checks_recognize_v2/test_data/test'

    for root, dirs, files in os.walk(imgs_dir):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            img_name = file
            print(img_name)
            img_dir = file_path
            origin_img = Image.open(img_dir).convert('L')
            # origin_img = Image.open(img_dir)
            width = origin_img.width
            height = origin_img.height

            img = origin_img.resize((224, 224))
            # img = np.array(img)
            # normalize = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ])
            # img = normalize(img)
            img = np.array(img)
            img = img / 255
            img = torch.from_numpy(img).float()
            img = torch.unsqueeze(img, 0)
            img = torch.unsqueeze(img, 0)
            img = img.to(device)
            output_class, output_mode = model(img)
            text = ""
            if output_class[0][0] < output_class[0][1]:
                text += ("class :" + "up ")
            else:
                text += ("class :" + "dowm ")

            mode = torch.max(output_mode, 1)[1]
            text += ("mode :" + str(mode[0]+1))
            result_img = cv2.putText(np.array(origin_img).copy(), text, (50, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            cv2.imwrite(os.path.join('/home/ubuntu/cs/checks_recognize_v2/test_data/resultes', img_name), result_img)


