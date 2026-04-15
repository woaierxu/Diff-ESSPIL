import pathlib
import concurrent.futures
from functools import partial
import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from networks.RNCL_Block.guided_diffusion.gaussian_diffusion import _extract_into_tensor


def getLargestCC(segmentation):
    labels = label(segmentation)
    #assert( labels.max() != 0 ) # assume at least 1 CC
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    else:
        largestCC = segmentation
    return largestCC


def var_all_case_guide_model(root_path, model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, dataset='LA'):
    if dataset == 'LA':
        with open('./Datasets/la/data_split/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [root_path + r"/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for
                      item in image_list]
    elif dataset == 'BraTS':
        with open('./Datasets/brats/val.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [root_path + r"/" + item.replace('\n', '') + ".h5" for item in
                      image_list]
    elif dataset =='KiTS19':
        with open('./Datasets/kits/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [root_path + r"/" + item.replace('\n', '') + ".h5" for item in image_list]

    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if dataset=='KiTS19':
            # preprocess
            image = image.swapaxes(0, 2)  # 192*192*64
            label = label.swapaxes(0, 2)
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - np.mean(image)) / np.std(image)
            label = (label > 0).astype(np.int8)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice


def var_all_case_RNCL(root_path, model_v, model_dfu, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                      model_choose = 0,
                      has_feature=False,
                      dataset = 'LA',
                      in_prob = False,
                      multi_step = True,
                      fuse_feature1=False,
                      cut_down_PIL = False
                      ):

    if dataset=='LA':
        with open('./Datasets/la/data_split/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [root_path + r"/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for
                      item in image_list]
    elif dataset =='KiTS19':
        with open('./Datasets/kits/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [root_path + r"/" + item.replace('\n', '') + ".h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if dataset=='KiTS19':
            # preprocess
            image = image.swapaxes(0, 2)  # 192*192*64
            label = label.swapaxes(0, 2)
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - np.mean(image)) / np.std(image)
            label = (label > 0).astype(np.int8)
        if not fuse_feature1:
            prediction, score_map = test_single_case_RNCL(model_v, model_dfu, image, stride_xy, stride_z, patch_size,
                                                          num_classes=num_classes,
                                                          model_choose = model_choose,
                                                          has_feature=has_feature,
                                                          in_prob= in_prob,
                                                          multi_step=multi_step,
                                                          cut_down_PIL= cut_down_PIL
                                                          )
        else:
            prediction, score_map = test_single_case_rncl_fuse_feature(model_v, model_dfu, image, stride_xy, stride_z, patch_size,
                                                                       num_classes=num_classes, model_choose=model_choose,
                                                                       has_feature=has_feature, in_prob=in_prob,
                                                                       multi_step=multi_step,
                                                                       cut_down_PIL = cut_down_PIL
                                                                       )
        if np.sum(prediction) == 0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def test_single_case_RNCL(model_v, model_dfu, image, stride_xy, stride_z, patch_size,
                          num_classes=1, model_choose=0, has_feature = False, in_prob = True,
                          multi_step = False, add_pred_prob = False,
                          cut_down_PIL = False):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    if model_choose ==0:
                        y1, _ = model_v(test_patch)
                    elif model_choose==1:
                        y1_mid, _ = model_v(test_patch)
                        y1_mid = F.softmax(y1_mid, dim=1)
                        if not in_prob:
                            y1_mid = (y1_mid>0.5).float()
                        if has_feature:
                            if multi_step:
                                if not cut_down_PIL:
                                    if add_pred_prob:
                                        combine_image = torch.cat([test_patch,y1_mid[:,1:2,...]],dim=1)
                                        y1, _ = model_dfu.module.inference(combine_image, y1_mid)
                                    else:
                                        y1, _ = model_dfu.module.inference(test_patch, y1_mid)
                                else:
                                    try:
                                        no_image = model_dfu.module.no_image
                                        test_patch_zeros = torch.zeros_like(test_patch)
                                        if add_pred_prob:
                                            combine_image = torch.cat([test_patch_zeros,y1_mid[:,1:2,...]],dim=1)
                                            y1, _ = model_dfu.module.inference(combine_image, y1_mid)
                                        else:
                                            y1, _ = model_dfu.module.inference(test_patch_zeros, y1_mid)
                                    except:
                                        zero_tensor = torch.zeros(y1_mid.shape).cuda()
                                        if add_pred_prob:
                                            combine_image = torch.cat([test_patch,zero_tensor[:,1:2,...]],dim=1)
                                            y1, _ = model_dfu.module.inference(combine_image, zero_tensor)
                                        else:
                                            y1, _ = model_dfu.module.inference(test_patch, zero_tensor)
                            else:
                                y1, _ = model_dfu(test_patch, y1_mid)
                        else:
                            if multi_step:
                                y1 = model_dfu.module.inference(test_patch, y1_mid)
                            else:
                                y1 = model_dfu(test_patch, y1_mid)


                    if y1.shape[1]==1:
                        y = F.sigmoid(y1)
                    else:
                        y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def test_single_case_rncl_fuse_feature(model_v, model_dfu, image, stride_xy, stride_z, patch_size,
                                       num_classes=1, model_choose=0, has_feature = False, in_prob = True,
                                       multi_step = False, add_pred_prob = False,
                                       cut_down_PIL = False):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    if model_choose ==0:
                        y1, _ = model_v(test_patch)
                    elif model_choose==1:
                        y1_mid, feature1 = model_v(test_patch)
                        y1_mid = F.softmax(y1_mid, dim=1)
                        if not in_prob:
                            y1_mid = (y1_mid>0.5).float()
                        if has_feature:
                            if multi_step:
                                if not cut_down_PIL:
                                    if add_pred_prob:
                                        combine_image = torch.cat([test_patch,y1_mid[:,1:2,...]],dim=1)
                                        y1, _ = model_dfu.module.inference(combine_image, y1_mid, feature1)
                                    else:
                                        y1, _ = model_dfu.module.inference(test_patch, y1_mid, feature1)
                                else:
                                    try:
                                        no_image = model_dfu.module.no_image
                                        test_patch_zeros = torch.zeros_like(test_patch)
                                        if add_pred_prob:
                                            combine_image = torch.cat([test_patch_zeros,y1_mid[:,1:2,...]],dim=1)
                                            y1, _ = model_dfu.module.inference(combine_image, y1_mid, feature1)
                                        else:
                                            y1, _ = model_dfu.module.inference(test_patch_zeros, y1_mid, feature1)
                                    except:
                                        zero_tensor = torch.zeros(y1_mid.shape).cuda()
                                        if add_pred_prob:
                                            combine_image = torch.cat([test_patch,zero_tensor[:,1:2,...]],dim=1)
                                            y1, _ = model_dfu.module.inference(combine_image, zero_tensor, feature1)
                                        else:
                                            y1, _ = model_dfu.module.inference(test_patch, zero_tensor, feature1)
                            else:
                                y1, _ = model_dfu(test_patch, y1_mid, feature1)
                        else:
                            if multi_step:
                                y1 = model_dfu.module.inference(test_patch, y1_mid, feature1)
                            else:
                                y1 = model_dfu(test_patch, y1_mid, feature1)


                    if y1.shape[1]==1:
                        y = F.sigmoid(y1)
                    else:
                        y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def test_all_case_DiffUV(model_v, model_dfu, num_classes, image_list, patch_size=(112, 112, 80),
                         stride_xy=18, stride_z=4,
                         nms=0,
                         model_choose = 0,
                         has_feature=False,
                         is_KiTS19 = False,
                         save_path = None,
                         in_prob = True,
                         multi_step = False,
                         add_pred_prob = False,
                         ablation_net = False,
                         fuse_feature1=False):
    loader = tqdm(image_list)
    total_metric = 0.0
    for i,image_path in enumerate(loader):
        h5f = h5py.File(image_path, 'r')
        name = pathlib.Path(image_path).parent.name
        image = h5f['image'][:]
        label = h5f['label'][:]
        if is_KiTS19:
            # preprocess
            image = image.swapaxes(0, 2)  # 192*192*64
            label = label.swapaxes(0, 2)
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - np.mean(image)) / np.std(image)
            label = (label > 0).astype(np.int8)
        if ablation_net:
            prediction, score_map = test_single_case_ablation_net(model_v, model_dfu,
                                                                  image, stride_xy, stride_z, patch_size,
                                                                  num_classes=num_classes,
                                                                  model_choose=model_choose,
                                                                  has_feature=has_feature,
                                                                  in_prob=in_prob,
                                                                  multi_step=multi_step,
                                                                  add_pred_prob=add_pred_prob)
        elif fuse_feature1:
            prediction, score_map = test_single_case_rncl_fuse_feature(model_v, model_dfu,
                                                                       image, stride_xy, stride_z, patch_size,
                                                                       num_classes=num_classes,
                                                                       model_choose=model_choose,
                                                                       has_feature=has_feature,
                                                                       in_prob=in_prob,
                                                                       multi_step=multi_step,
                                                                       add_pred_prob=add_pred_prob)
        else:
            prediction, score_map = test_single_case_RNCL(model_v, model_dfu,
                                                          image, stride_xy, stride_z, patch_size,
                                                          num_classes=num_classes,
                                                          model_choose = model_choose,
                                                          has_feature=has_feature,
                                                          in_prob = in_prob,
                                                          multi_step=multi_step,
                                                          add_pred_prob=add_pred_prob)
        if nms == 1:
            prediction = getLargestCC(prediction)
        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        print(single_metric)
        total_metric += np.asarray(single_metric)
    avg_metrics = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metrics))
    return avg_metrics


def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1, _ = model(test_patch)
                    y = F.softmax(y1, dim=1)

                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, jc, hd, asd



def test_single_sample_wrapper(image_path, model_v, model_dfu, num_classes, patch_size,
                               stride_xy, stride_z, nms, model_choose, has_feature,
                               is_KiTS19, save_path, in_prob, multi_step, index,add_pred_prob,cut_down_PIL,fuse_feature1):
    """
    单个样本测试的包装函数,用于并行处理
    """
    with torch.no_grad():
        try:
            h5f = h5py.File(image_path, 'r')
            name = pathlib.Path(image_path).parent.name
            image = h5f['image'][:]
            label = h5f['label'][:]
            h5f.close()

            if is_KiTS19:
                # preprocess
                image = image.swapaxes(0, 2)  # 192*192*64
                label = label.swapaxes(0, 2)
                image = (image - np.mean(image)) / np.std(image)
                label = (label > 0).astype(np.int8)
            if not fuse_feature1:
                prediction, score_map = test_single_case_RNCL(
                    model_v, model_dfu, image, stride_xy, stride_z, patch_size,
                    num_classes=num_classes, model_choose=model_choose,
                    has_feature=has_feature, in_prob=in_prob, multi_step=multi_step,add_pred_prob=add_pred_prob,
                    cut_down_PIL = cut_down_PIL
                )
            else:
                prediction, score_map = test_single_case_rncl_fuse_feature(
                    model_v, model_dfu, image, stride_xy, stride_z, patch_size,
                    num_classes=num_classes, model_choose=model_choose,
                    has_feature=has_feature, in_prob=in_prob, multi_step=multi_step, add_pred_prob=add_pred_prob,
                    cut_down_PIL=cut_down_PIL
                )
            if nms ==1:
                prediction = getLargestCC(prediction)
            if np.sum(prediction) == 0:
                single_metric = (0, 0, 0, 0)
            else:
                single_metric = calculate_metric_percase(prediction, label[:])

            print(f"Sample {index} ({name}): {single_metric}")

            return np.asarray(single_metric), name

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return np.asarray([0, 0, 0, 0]), pathlib.Path(image_path).parent.name


def test_all_case_DiffUV_parallel(model_v, model_dfu, num_classes,image_list,
                                  patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                  model_choose=0, has_feature=False, nms = 0,
                                  is_KiTS19=False, save_path=None, in_prob=True,
                                  multi_step=True, num_workers=4, add_pred_prob = False,cut_down_PIL = False,
                                  fuse_feature1 = False):
    """
    并行版本的测试函数

    Args:
        num_workers: 并行工作进程数量 (新增参数)

    Returns:
        avg_metrics: 平均指标
    """

    print(f"Testing {len(image_list)} cases with {num_workers} parallel workers...")

    total_metric = np.zeros(4)

    # 使用ThreadPoolExecutor进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务 - 修正参数传递方式
        futures = [
            executor.submit(
                test_single_sample_wrapper,
                image_path=image_path,  # 明确指定参数名
                model_v=model_v,
                model_dfu=model_dfu,
                num_classes=num_classes,
                patch_size=patch_size,
                stride_xy=stride_xy,
                stride_z=stride_z,
                nms = nms,
                model_choose=model_choose,
                has_feature=has_feature,
                is_KiTS19=is_KiTS19,
                save_path=save_path,
                in_prob=in_prob,
                multi_step=multi_step,
                index=i,
                add_pred_prob = add_pred_prob,
                cut_down_PIL = cut_down_PIL,
                fuse_feature1 = fuse_feature1
            )
            for i, image_path in enumerate(image_list)
        ]

        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(image_list),
                           desc="Processing samples"):
            single_metric, name = future.result()
            total_metric += single_metric

    avg_metrics = total_metric / len(image_list)
    print('Average metric is {}'.format(avg_metrics))
    return avg_metrics

def val_all_case_DiffUV_parallel(root_path, model_v, model_dfu, num_classes,
                                  patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                  model_choose=0, has_feature=False, image_list=None, nms = 0,
                                  is_KiTS19=False, save_path=None, in_prob=True,
                                  multi_step=True, num_workers=4, add_pred_prob = False,cut_down_PIL = False,
                                  fuse_feature1 = False):
    """
    并行版本的测试函数

    Args:
        num_workers: 并行工作进程数量 (新增参数)

    Returns:
        avg_metrics: 平均指标
    """
    if image_list is None:
        with open('./Datasets/la/data_split/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [
            root_path +
            r"/2018LA_Seg_Training Set/" +
            item.replace('\n', '') + "/mri_norm2.h5"
            for item in image_list
        ]

    print(f"Testing {len(image_list)} cases with {num_workers} parallel workers...")

    total_metric = np.zeros(4)

    # 使用ThreadPoolExecutor进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务 - 修正参数传递方式
        futures = [
            executor.submit(
                test_single_sample_wrapper,
                image_path=image_path,  # 明确指定参数名
                model_v=model_v,
                model_dfu=model_dfu,
                num_classes=num_classes,
                patch_size=patch_size,
                stride_xy=stride_xy,
                stride_z=stride_z,
                nms = nms,
                model_choose=model_choose,
                has_feature=has_feature,
                is_KiTS19=is_KiTS19,
                save_path=save_path,
                in_prob=in_prob,
                multi_step=multi_step,
                index=i,
                add_pred_prob = add_pred_prob,
                cut_down_PIL = cut_down_PIL,
                fuse_feature1 = fuse_feature1
            )
            for i, image_path in enumerate(image_list)
        ]

        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(image_list),
                           desc="Processing samples"):
            single_metric, name = future.result()
            total_metric += single_metric

    avg_metrics = total_metric / len(image_list)
    print('Average metric is {}'.format(avg_metrics))
    return avg_metrics


def test_single_case_ablation_net(model_v,model_dfu, image, stride_xy, stride_z, patch_size,
                            num_classes=1,model_choose=0,has_feature = False,in_prob = True,
                            multi_step = False,add_pred_prob = False,
                            cut_down_PIL = False):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    if model_choose ==0:
                        y1, _ = model_v(test_patch)
                    elif model_choose==1:
                        if model_dfu.module.PIL:
                            y1_mid, _ = model_v(test_patch)
                            y1_mid = F.softmax(y1_mid, dim=1)
                            if not in_prob:
                                y1_mid = (y1_mid>0.5).float()
                            try:
                                noise = model_dfu.module.noise
                                y1_mid_noised = y1_mid + noise * torch.randn_like(y1_mid)
                                y1 = model_dfu(test_patch, y1_mid_noised)
                            except:
                                y1 = model_dfu(test_patch, y1_mid)
                        else:
                            y1 = model_dfu(test_patch)





                    if y1.shape[1]==1:
                        y = F.sigmoid(y1)
                    else:
                        y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

