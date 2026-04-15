import torch
from torch.nn import functional as F
import torch.nn as nn
import contextlib
import pdb
import numpy as np
from scipy.ndimage import distance_transform_edt


# region Loss rncl
def get_weight(pred1, pred2, shift_step=16, ratio=(0.4, 0.6, 0.8)):
    import torch
    import torch.nn.functional as F
    from einops import rearrange

    assert pred1.shape == pred2.shape , 'MQX in get_weight func:input should be the same shape!'
    B,C,D,H,W = pred1.shape

    def pad_to_multiple(x):
        pad_D = (shift_step - D % shift_step) % shift_step
        pad_H = (shift_step - H % shift_step) % shift_step
        pad_W = (shift_step - W % shift_step) % shift_step
        return F.pad(x, (0, pad_W, 0, pad_H, 0, pad_D)), pad_D, pad_H, pad_W

    pred1, pad_D, pad_H, pad_W = pad_to_multiple(pred1)
    pred2, *_ = pad_to_multiple(pred2)
    B, C, D, H, W = pred1.shape



    patch1 = rearrange(pred1, 'b 1 (d p1) (h p2) (w p3) -> b (d h w) p1 p2 p3', p1=shift_step, p2=shift_step, p3=shift_step)
    patch2 = rearrange(pred2, 'b 1 (d p1) (h p2) (w p3) -> b (d h w) p1 p2 p3', p1=shift_step, p2=shift_step, p3=shift_step)

    nonzero1 = torch.count_nonzero(patch1, dim=(2, 3, 4))
    nonzero2 = torch.count_nonzero(patch2, dim=(2, 3, 4))
    total_voxels = shift_step ** 3
    same_mask = ((nonzero1 == 0) & (nonzero2 == 0)) | ((nonzero1 == total_voxels) & (nonzero2 == total_voxels))  # (1, 245)
    # XOR
    diff_mask = patch1 ^ patch2

    weights = torch.full_like(diff_mask, ratio[1], dtype=torch.float)


    weights[diff_mask] = ratio[2]


    same_mask = same_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    weights = torch.where(same_mask, torch.full_like(weights, ratio[0]), weights)

    weights = rearrange(weights, 'b (c d h w) p1 p2 p3 -> b c (d p1) (h p2) (w p3)', d=D // shift_step, h=H // shift_step, w=W // shift_step)
    weights = weights.repeat((1,2,1,1,1))
    return weights

class WeightedMSE(torch.nn.Module):
    def __init__(self):
        super(WeightedMSE, self).__init__()

    def forward(self, p1, p2, weight):
        """
        输入 patch1, patch2, weights，shape 必须一致
        """
        assert p1.shape == p2.shape == weight.shape, 'MQX in weighted_mse func:input should be the same shape!'
        diff_sq = (p1.float() - p2.float()) ** 2
        weighted = diff_sq * weight
        return weighted.mean()


class WeightedDiceLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, p1, p2, weight):
        """
        加权 Dice Loss（适用于 hard 01 值），所有维度一起算
        """
        assert p1.shape == p2.shape == weight.shape, 'MQX in weighted_dice_loss func:input should be the same shape!'
        p1_flat = p1.reshape(-1).float()
        p2_flat = p2.reshape(-1).float()
        weight_flat = weight.reshape(-1).float()

        inter = torch.sum(weight_flat * p1_flat * p2_flat)
        sum_p1 = torch.sum(weight_flat * p1_flat)
        sum_p2 = torch.sum(weight_flat * p2_flat)
        dice = (2 * inter + self.eps) / (sum_p1 + sum_p2 + self.eps)
        return 1 - dice


# endregion

class mask_DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(mask_DiceLoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    
    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            # bug found by @CamillerFerros at github issue#25
            mask = mask.repeat(1, self.n_classes, 1, 1).type(torch.float32)
            for i in range(0, self.n_classes): 
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes

class CrossEntropyLoss(nn.Module):
    def __init__(self, n_classes):
        super(CrossEntropyLoss, self).__init__()
        self.class_num = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.class_num):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def _one_hot_mask_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.class_num):
            temp_prob = input_tensor * i == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _ce_loss(slef, score, target, mask):
        target = target.float()
        loss = (-target * torch.log(score) * mask.float()).sum() / (mask.sum() + 1e-16)
        return loss

    def forward(self, inputs, target, mask):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        mask = self._one_hot_mask_encoder(mask)
        loss = 0.0
        for i in range(0, self.class_num):
            loss += self._ce_loss(inputs[:,i], target[:, i], mask[:, i])
        return loss / self.class_num 


def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def get_probability(logits):
    """ Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass

class Dice_Loss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(Dice_Loss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

class softDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(softDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice
        return loss / self.n_classes
        
@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d):
    # pdb.set_trace()
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8  ###2-p length of vector
    return d

class VAT2d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT2d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(4)

    def forward(self, model, x):
        with torch.no_grad():
            pred= F.softmax(model(x)[0], dim=1)

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d) 
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)[0]
                logp_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(logp_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            pred_hat = model(x + r_adv)[0]
            logp_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(logp_hat, pred)
        return lds

class VAT3d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT3d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = Binary_dice_loss
        
    def forward(self, model, x):
        with torch.no_grad():
            pred= F.softmax(model(x)[0], dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device) ### initialize a random tensor between [-0.5, 0.5]
        d = _l2_normalize(d) ### an unit vector
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)[0]
                p_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(p_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
            pred_hat = model(x + self.epi * d)[0]
            p_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(p_hat, pred)
        return lds

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)


# 一个接受 one-hot target 的 Dice Loss 实现
def dice_loss_one_hot(input, target):
    """
    input: a tensor of shape [B, C, D, H, W] (probabilities)
    target: a tensor of shape [B, C, D, H, W] (one-hot encoding)
    """
    smooth = 1e-5

    # 确保 input 是概率形式
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)

    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - (2. * intersection + smooth) / (A_sum + B_sum + smooth)

def create_focus_weight_mask_by_LPmodel(true_label, pred_label, high_penalty_weight=5.0, low_reward_weight=1.0,need_channel = False):
    """
    生成重点关注区域的权重掩码。

    Args:
        true_label (Tensor): 真实标签, shape (B, H, W)
        pred_label (Tensor): 初始预测标签, shape (B, H, W)
        high_penalty_weight (float): 对修正失败区域的惩罚权重。
        low_reward_weight (float): 对修正成功或保持正确的区域的权重。

    Returns:
        Tensor: 权重掩码, shape (B, H, W)
    """
    # 1. 找到 pred_label 和 true_label 不同的区域，即“重点关注区域” RoI
    # RoI 中的像素值为 True (or 1)
    roi_mask = (pred_label != true_label)

    # 2. 初始化权重掩码，所有像素的初始权重都为 low_reward_weight
    # 这个权重应用于那些 pred_label 已经预测正确的区域
    weight_mask = torch.full_like(true_label, fill_value=low_reward_weight, dtype=torch.float)

    # 3. 在重点关注区域 (RoI) 内，将权重提升为 high_penalty_weight
    # 这样，在这些区域计算损失时，会被放大
    weight_mask[roi_mask] = high_penalty_weight
    if need_channel:
        return weight_mask
    else:
        return weight_mask[:, 1, ...]

def focused_loss(final_pred_logits, true_label, pred_label, high_penalty_weight=5.0, low_reward_weight=1.0):
    """
    计算聚焦于修正错误的加权损失。

    Args:
        final_pred_logits (Tensor): 模型最终输出的logits, shape (B, C, H, W)
        true_label (Tensor): 真实标签, shape (B, H, W)
        pred_label (Tensor): 初始预测标签, shape (B, H, W)
        high_penalty_weight (float): 高惩罚权重。
        low_reward_weight (float): 低奖励权重。

    Returns:
        Tensor: 一个标量，表示最终的损失值。
    """
    # 1. 获取权重掩码
    weight_mask = create_focus_weight_mask_by_LPmodel(
        true_label, pred_label, high_penalty_weight, low_reward_weight
    )

    # 2. 计算逐像素的交叉熵损失，不进行任何聚合 (reduction='none')
    # loss_per_pixel 的形状将是 (B, H, W)
    loss_per_pixel = F.cross_entropy(final_pred_logits, true_label, reduction='none')

    # 3. 将权重掩码应用到逐像素损失上
    weighted_loss_per_pixel = loss_per_pixel * weight_mask

    # 4. 对加权后的损失进行求平均
    # 除以 weight_mask.sum() 是为了让损失的量级保持稳定，不受高权重值的影响
    # 也可以直接 .mean()，但前者在理论上更稳健
    final_loss = weighted_loss_per_pixel.sum() / weight_mask.sum()
    # final_loss = weighted_loss_per_pixel.mean() # 这是一个更简单的替代方案

    return final_loss





