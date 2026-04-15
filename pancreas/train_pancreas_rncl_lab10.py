import pathlib
from tqdm import tqdm as tqdm_load
from utils.losses import WeightedDiceLoss, WeightedMSE
from pancreas_utils import *
from test_util import *
from losses import *
from dataloaders import get_ema_model_and_dataloader_rncl
import torch.nn.functional as F





"""Global Variables"""
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed_test = 2020
seed_reproducer(seed = seed_test)
label_percent = 10
data_root, split_name = r'/data1/mengqingxu/Dataset/Pancreas/preprocess/', 'pancreas'
exp_name = pathlib.Path(__file__).name.replace('.py','')
result_dir = f'model/{exp_name}/'
mkdir(result_dir)
batch_size, lr = 2, 1e-3
pretraining_epochs, self_training_epochs = 101, 321
pretrain_save_step, st_save_step, pred_step = 10, 20, 5
alpha, consistency, consistency_rampup = 0.99, 0.1, 40
u_weight = 1.5
connect_mode = 2
try_second = 1
sec_t = 0.5
self_train_name = 'self_train'
lambda_dice = 1
# MQX
num_classes_dfu = 2
BIAS = 0.25

sub_batch = int(batch_size/2)
consistency_criterion = softmax_mse_loss
CE = nn.CrossEntropyLoss()
CE_r = nn.CrossEntropyLoss(reduction='none')
DICE = DiceLoss(nclass=2)
patch_size = 64

logger = None


def cmp_dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

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


def pretrain(net1, net2, optimizer1, optimizer2, lab_loader_a, lab_loader_b, test_loader):
    """pretrain image- & patch-aware network"""

    """Create Path"""
    save_path = Path(result_dir) / 'pretrain'
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger
    logger, writer = cutmix_config_log(save_path, tensorboard=True)
    logger.info("cutmix Pretrain, patch_size: {}, save path: {}".format(patch_size, str(save_path)))

    max_dice1 = 0
    max_dice2 = 0
    measures = CutPreMeasures(writer, logger)
    net2.module.fuse_feature = True
    for epoch in tqdm_load(range(1, pretraining_epochs + 1), ncols=70):
        measures.reset()
        """Testing"""
        if epoch % 5 == 0:
            net1.eval()
            net2.eval()
            avg_metric1, _ = test_calculate_metric_MQX(net1, net2, test_loader.dataset, s_xy=16, s_z=16, model_choose=0)
            avg_metric2, _ = test_calculate_metric_MQX(net1, net2, test_loader.dataset, s_xy=16, s_z=16, model_choose=1)

            logger.info('average metric is : {}'.format(avg_metric1))
            logger.info('average metric is : {}'.format(avg_metric2))
            val_dice1 = avg_metric1[0]
            val_dice2 = avg_metric2[0]

            if val_dice1 > max_dice1:
                save_net_opt(net1, optimizer1, save_path / f'best_ema{label_percent}_pre_vnet.pth', epoch)
                max_dice1 = val_dice1
                with open(str(save_path / f'epoch_{epoch}_{label_percent}_ValDice1_{round(val_dice1,4)}_self.txt'),'w')as f:
                    f.write(f'val_dice1:{round(val_dice1,4)}')

            if val_dice2 > max_dice2:
                save_net_opt(net2, optimizer2, save_path / f'best_ema{label_percent}_pre_diffu.pth', epoch)
                max_dice2 = val_dice2
                with open(str(save_path / f'epoch_{epoch}_{label_percent}_ValDice2_{round(val_dice2,4)}_self.txt'),'w')as f:
                    f.write(f'val_dice1:{round(val_dice2,4)}')

            logger.info('\nEvaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice1, max_dice1))
            logger.info('resnet Evaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice2, max_dice2))

        """Training"""
        net1.train()
        net2.train()
        logger.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b  = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            img_mask, loss_mask = generate_mask(img_a, patch_size)

            img = img_a * img_mask + img_b * (1 - img_mask)
            lab = lab_a * img_mask + lab_b * (1 - img_mask)

            out1 = net1(img)
            ce_loss1 = F.cross_entropy(out1, lab)
            dice_loss1 = DICE(out1, lab)
            loss1 = (ce_loss1 + dice_loss1) / 2
            l_label_batch = torch.unsqueeze(lab, dim=1).float()
            if num_classes_dfu == 2:
                l_in_cls1_label = l_label_batch
                l_in_cls0_label = 1 - l_in_cls1_label
                l_in_label = torch.cat((l_in_cls0_label, l_in_cls1_label), dim=1)
            else:
                l_in_label = l_label_batch

            with torch.no_grad():
                _, features = net1(img, need_features = True)
            output2_logits, _ = net2(img, l_in_label, features)
            loss_ce2 = F.cross_entropy(output2_logits, lab)
            loss_dice2 = DICE(output2_logits, lab)
            loss2 = (loss_ce2 + loss_dice2) / 2

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            logger.info("cur epoch: %d step: %d" % (epoch, step+1))
            measures.update(out1, lab, ce_loss1, dice_loss1, loss1)
            measures.update(output2_logits, lab, loss_ce2, loss_dice2, loss2, model_choose=1)
            measures.log(epoch, epoch * len(lab_loader_a) + step)


    return max_dice1

def ema_cutmix(net1, net2, ema_net1, optimizer1, optimizer2, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader):

    def get_XOR_region(mixout1, mixout2):
        s1 = torch.softmax(mixout1, dim = 1)
        l1 = torch.argmax(s1, dim = 1)

        s2 = torch.softmax(mixout2, dim = 1)
        l2 = torch.argmax(s2, dim = 1)

        diff_mask = (l1 != l2)
        return diff_mask

    """Create Path"""
    save_path = Path(result_dir) / self_train_name
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger
    logger, writer = config_log(save_path, tensorboard=True)
    logger.info("EMA_training, save_path: {}".format(str(save_path)))
    measures = CutmixFTMeasures(writer, logger)

    """Load Model"""
    pretrained_path = Path(f'./model/{exp_name}') / 'pretrain'
    load_net_opt(net1, optimizer1, pretrained_path / f'best_ema{label_percent}_pre_vnet.pth')
    load_net_opt(net2, optimizer2, pretrained_path / f'best_ema{label_percent}_pre_diffu.pth')
    load_net_opt(ema_net1, optimizer1, pretrained_path / f'best_ema{label_percent}_pre_vnet.pth')
    logger.info('Loaded from {}'.format(pretrained_path))
    net2.module.fuse_feature = True
    max_dice1 = 0
    max_list1 = None
    max_dice2 = 0
    weighted_dice_loss = WeightedDiceLoss()
    weighted_mse = WeightedMSE()

    for epoch in tqdm_load(range(1, self_training_epochs+1)):
        measures.reset()
        logger.info('')

        # Validate
        if (epoch % 20 == 0) | ((epoch >= 160) & (epoch % 5 ==0)):
            net1.eval()
            net2.eval()

            avg_metric1, _ = test_calculate_metric_MQX(net1, net2, test_loader.dataset, s_xy=16, s_z=16, model_choose=0)
            avg_metric2, _ = test_calculate_metric_MQX(net1, net2, test_loader.dataset, s_xy=16, s_z=16, model_choose=1)

            logger.info('average metric is : {}'.format(avg_metric1))
            logger.info('average metric is : {}'.format(avg_metric2))

            val_dice1 = avg_metric1[0]
            val_dice2 = avg_metric2[0]

            if val_dice1 > max_dice1:
                save_net(net1, str(save_path / f'best_ema_{label_percent}_self.pth'))
                max_dice1 = val_dice1
                max_list1 = avg_metric1
                with open(str(save_path / f'epoch_{epoch}_{label_percent}_ValDice1_{round(val_dice1,4)}_self.txt'),'w')as f:
                    f.write(f'val_dice1:{round(val_dice1,4)}')

            if val_dice2 > max_dice2:
                save_net(net2, str(save_path / f'best_ema_{label_percent}_self_diffu.pth'))
                max_dice2 = val_dice2
                with open(str(save_path / f'epoch_{epoch}_{label_percent}_ValDice2_{round(val_dice2,4)}_self.txt'),'w')as f:
                    f.write(f'val_dice1:{round(val_dice2,4)}')


            logger.info('\nEvaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice1, max_dice1))
            logger.info('resnet Evaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice2, max_dice2))

        """Training"""
        net1.train()
        net2.train()
        ema_net1.train()
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda([img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])
            """Generate Pseudo Label"""
            with torch.no_grad():
                unimg_a_out_1 = ema_net1(unimg_a)
                unimg_b_out_1 = ema_net1(unimg_b)

                uimg_a_plab = get_cut_mask(unimg_a_out_1, nms=True, connect_mode=connect_mode)
                uimg_b_plab = get_cut_mask(unimg_b_out_1, nms=True, connect_mode=connect_mode)


                img_mask, loss_mask = generate_mask(img_a, patch_size)


            """Mix input"""
            net3_input_l = unimg_a * img_mask + img_b * (1 - img_mask)
            net3_input_unlab = img_a * img_mask + unimg_b * (1 - img_mask)
            l_label_batch_ori = uimg_a_plab * img_mask + lab_b * (1 - img_mask)
            ul_label_batch_ori = lab_a * img_mask + uimg_b_plab * (1 - img_mask)

            """BCP"""
            """Supervised Loss"""
            mix_lab_out = net1(net3_input_l)
            mix_output_l = mix_lab_out
            loss_1 = mix_loss(mix_output_l, uimg_a_plab.long(), lab_b, loss_mask, unlab=True)


            """Unsupervised Loss"""
            mix_unlab_out = net1(net3_input_unlab)
            mix_output_2 = mix_unlab_out
            loss_2 = mix_loss(mix_output_2, lab_a, uimg_b_plab.long(), loss_mask)
            label_all = torch.concat([l_label_batch_ori.unsqueeze(1), ul_label_batch_ori.unsqueeze(1)], dim=0)
            outputs_1_prob = torch.softmax(torch.concat([mix_lab_out, mix_unlab_out], dim=0), dim=1)
            weight_1 = torch.ones_like(outputs_1_prob)*0.75
            label_all = torch.cat((1 - label_all, label_all), dim=1)
            loss_weight_mse_1 = weighted_mse(outputs_1_prob, label_all, weight_1)
            loss_weight_dice_1 = weighted_dice_loss(outputs_1_prob, label_all, weight_1)
            # region MQX model forward

            with torch.no_grad():
                p_l_label_batch, l_features = ema_net1(net3_input_l,need_features = True)
                p_ul_label_batch, ul_features = ema_net1(net3_input_unlab,need_features = True)
                p_l_label_batch, p_ul_label_batch = F.softmax(p_l_label_batch, dim=1)[:, 1:2, ...], F.softmax(
                    p_ul_label_batch, dim=1)[:, 1:2, ...]
                # p_l_label_batch, p_ul_label_batch = (p_l_label_batch > 0.5).float(), (p_ul_label_batch > 0.5).float()

            l_label_batch_ori, ul_label_batch_ori = torch.unsqueeze(l_label_batch_ori, dim=1), torch.unsqueeze(
                ul_label_batch_ori, dim=1)
            pred_prob_lab = p_l_label_batch
            pred_prob_unlab = p_ul_label_batch
            if num_classes_dfu == 2:
                l_in_cls1_label = pred_prob_lab
                l_in_cls0_label = 1 - l_in_cls1_label
                l_in_prob = torch.cat((l_in_cls0_label, l_in_cls1_label), dim=1)

                ul_in_cls1_label = pred_prob_unlab
                ul_in_cls0_label = 1 - ul_in_cls1_label
                ul_in_prob = torch.cat((ul_in_cls0_label, ul_in_cls1_label), dim=1)
            else:
                l_in_prob = pred_prob_lab
                ul_in_prob = pred_prob_unlab
            combine_prob = torch.cat([l_in_prob,ul_in_prob],dim=0)
            combine_true_label = torch.cat([l_label_batch_ori,ul_label_batch_ori],dim=0)
            combine_true_label_onehot = torch.cat((1-combine_true_label, combine_true_label), dim=1)
            combine_features = []
            for l_f,ul_f in zip(l_features,ul_features):
                combine_features.append(torch.cat([l_f, ul_f], dim=0))

            # endregion

            """Supervised Loss"""
            combined_images = torch.cat([net3_input_l,net3_input_unlab],dim=0)
            output2,_ = net2(
                combined_images,
                combine_prob,
                combine_features,
            )
            output2 = torch.softmax(output2, dim=1)
            loss_ce_2 = F.cross_entropy(output2, combine_true_label_onehot)
            loss_dice_2 = dice_loss_one_hot(output2, combine_true_label_onehot)
            weight_2 = torch.ones_like(output2)*0.75
            loss_weight_mse_2 = weighted_mse(output2, combine_true_label_onehot, weight_2)
            loss_weight_dice_2 = weighted_dice_loss(output2, combine_true_label_onehot, weight_2)

            loss1 = loss_1 + loss_2 + loss_weight_mse_1 + loss_weight_dice_1

            loss2 = loss_ce_2 + loss_dice_2 + loss_weight_dice_2 + loss_weight_mse_2

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            update_ema_variables(net1, ema_net1, alpha)

            logger.info("loss_1: %.4f, loss_2: %.4f" %
                (loss_1.item(), loss_2.item()))

        if epoch == self_training_epochs:
            save_net(net1, str(save_path / f'best_ema_{label_percent}_self_latest.pth'))
    return max_dice1, max_list1

if __name__ == '__main__':
    net1, net2, ema_net1, optimizer1, optimizer2, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader = get_ema_model_and_dataloader_rncl(data_root, split_name, batch_size, lr, labelp=label_percent)
    # -- Pre-training
    pretrain(net1, net2, optimizer1, optimizer2, lab_loader_a, lab_loader_b, test_loader)
    seed_reproducer(seed = seed_test)
    # -- Self-training
    ema_cutmix(net1, net2, ema_net1, optimizer1, optimizer2, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader)




