#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torchvision.transforms as transforms
from networks import *

import torchvision
import torch.optim
import numpy as np
from PIL import Image
from glob import glob
import scipy.misc as misc
import skimage.measure
import os
import scipy.misc
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from utils import *

process = transforms.ToTensor()
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
torch.cuda.set_device(0)

def save_images(rain_image_output, test_label_out, predicted_image_out, filepath):
    cat_image = np.column_stack((test_label_out, predicted_image_out))
    cat_image = np.column_stack((rain_image_output, cat_image))
    # cat_image=np.clip(255*cat_image,0,255)
    im = Image.fromarray(cat_image.astype('uint8'))
    im.save(filepath, 'png')


def test():
    save_path = 'real'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    rain_list = sorted(os.listdir('rain_data_train_Heavy/val/rain/'))
    label_list = sorted(os.listdir('rain_data_train_Heavy/val/label/'))
    rain_path = 'rain_data_train_Heavy/val/'


    ssim_sum = 0
    psnr_sum = 0
    for i in range(len(rain_list)):
        print(rain_list[i], "-----", label_list[i])

        data_rain = skimage.io.imread(rain_path + 'rain/' + rain_list[i])
        data_label = skimage.io.imread(rain_path + 'label/' + label_list[i])
        data_rain_yuan=data_rain
        data_rain=data_rain.astype(np.float32)/255.

        data_rain = process(data_rain)
        data_rain = data_rain.unsqueeze(dim=0)
        if torch.cuda.is_available():
            data_rain = data_rain.cuda()
            derain_net = Net().cuda()
            derain_net.eval()
            derain_net.load_state_dict(torch.load('./final_H/net_step330000.pth'))
        else:
            data_rain= data_rain
            derain_net = Net().cuda()
            derain_net.load_state_dict(torch.load('./final_H/net_latest.pth'))
        with torch.no_grad():

            predicted_image = derain_net(data_rain.detach())

            predicted_image = predicted_image[0, :, :, :]
            predicted_image = predicted_image.permute(1, 2, 0)
            predicted_image = predicted_image.cpu().detach().numpy()

            predicted_image = np.clip(predicted_image * 255, 0, 255).astype('uint8')
            clean = np.clip(data_label, 0, 255).astype('uint8')



            ssim = skimage.measure.compare_ssim(predicted_image, clean, gaussian_weights=True, sigma=1.5,
                                                use_sample_covariance=False, multichannel=True)
            psnr = skimage.measure.compare_psnr(predicted_image, clean)

            img = Image.fromarray(predicted_image.astype('uint8'))

            img.save('{0}/{1}'.format(save_path, label_list[i]))
            save_images(clean, data_rain_yuan, predicted_image,
                        os.path.join("./real", '%s_%.4f_%.4f.png' % (rain_list[i], psnr, ssim)))
            print("SSIM = %.4f" % ssim)
            print("PSNR = %.4f" % psnr)
            ssim_sum += ssim
            psnr_sum += psnr
           
    avg_ssim = ssim_sum / len(rain_list)
    avg_psnr = psnr_sum / len(rain_list)
    print("---- Average SSIM = %.4f----" % avg_ssim)
    print("---- Average PSNR = %.4f----" % avg_psnr)


if __name__ == '__main__':
    test()
    print("perfect,done!")
