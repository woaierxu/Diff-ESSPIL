import pathlib
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import torch.optim as optim
import torch.backends.cudnn as cudnn
from skimage.measure import label
from torch.utils.data import DataLoader
from networks.DiffUVNet import DiffUnet_predMask_FuseFeature
from utils.losses import get_weight, WeightedMSE, dice_loss_one_hot
from utils import losses, ramps, test_3d_patch
from dataloaders.LADataset import LAHeart
from utils.LA_utils import to_cuda
from utils.BCP_utils import *
from pancreas.losses import *
from networks.Vnet_LA import VNet

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default=r'', help='Name of Dataset')
parser.add_argument('--model', type=str, default='RNCL', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=8000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=1e-3, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1345, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')

parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
parser.add_argument('--shift_step',type = int, default=2,help= 'the patch size for ESS loss')
parser.add_argument('--mse_ratio',type = list, default=[0.75, 0.75, 0.75],help= 'the diffenent weights for differnt region of ESS loss')
parser.add_argument('--lambda_dice',type = float, default=1,help= 'the weight of dice loss in loss_model2')
args = parser.parse_args()

patch_size = (112, 112, 80)
num_classes = 2

def create_guide_model(ema=False):
    net = VNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=False)
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def create_refinement_model(ema=False):
    net = DiffUnet_predMask_FuseFeature(
        in_channels=1,
        num_classes=num_classes,
        num_filters=(16, 32, 64, 128, 256, 32),
        infer_start_timestep=10, timesteps=20,
        fuse_mode='gate'
    )
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model




def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    batch_np = np.array(batch_list)

    return torch.Tensor(batch_np).cuda()


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)




def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))


def get_XOR_region(mixout1, mixout2):
    s1 = torch.softmax(mixout1, dim=1)
    l1 = torch.argmax(s1, dim=1)

    s2 = torch.softmax(mixout2, dim=1)
    l2 = torch.argmax(s2, dim=1)

    diff_mask = (l1 != l2)
    return diff_mask


def cmp_dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def pre_train(args, snapshot_path):
    # creat guide model and refinement model
    model = create_guide_model()
    model2 = create_refinement_model()

    c_batch_size = args.batch_size//2
    trainset_lab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split=f'train_lab{args.labelnum}', logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split=f'train_lab{args.labelnum}', reverse=True, logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)

    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    model2.train()
    logging.info("{} iterations per epoch".format(len(lab_loader_a)))
    iter_num = 0
    best_dice = 0.0
    best_dice2 = 0.0
    max_epoch = args.pre_max_iteration // len(lab_loader_a) +1
    iterator = tqdm(range(1, max_epoch), ncols=70)
    for epoch_num in iterator:
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs,_ = model(volume_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            l_label_batch = torch.unsqueeze(label_batch, dim=1).float()
            if num_classes == 2:
                l_in_cls1_label = l_label_batch
                l_in_cls0_label = 1 - l_in_cls1_label
                l_in_label = torch.cat((l_in_cls0_label, l_in_cls1_label), dim=1)
            else:
                l_in_label = l_label_batch
            with torch.no_grad():
                _,features1 = model(volume_batch)
            output2_logits, _ = model2(volume_batch, l_in_label,features1)

            loss_ce2 = F.cross_entropy(output2_logits, label_batch)
            loss_dice2 = DICE(output2_logits, label_batch)
            loss2 = (loss_ce2 + loss_dice2) / 2

            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            logging.info(
                'iteration %d : loss: %03f, loss_dice: %03f, loss2: %03f, loss_dice2: %03f,' % (
                    iter_num, loss, loss_dice, loss2, loss_dice2))
        # validate
        if epoch_num % 10 == 0:
            # val guide model
            model.eval()
            dice_sample = test_3d_patch.var_all_case_RNCL(
                args.root_path,
                model,                      # guide model
                model2,                     # refinement model
                model_choose=0,             # Test model  0：Vnet(guide model) 1：RNCL model(refinement-model).
                num_classes=num_classes,
                patch_size=patch_size,      # Training patch size.
                stride_xy=56, stride_z=40,  # Shifting window size during testing.
                fuse_feature1=True          # The FFAG model requires interaction between two models, so testing also needs to be matched accordingly.
            )

            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.txt'.format(iter_num, best_dice))
                save_best_path = os.path.join(snapshot_path, 'best_model.pth'.format(args.model))
                with open(save_mode_path,'w')as f:
                    f.write(f'Validation on model1: Dice={best_dice:.4f}')
                save_net_opt(model, optimizer, save_best_path, epoch_num)
                logging.info("save best model to {}".format(save_mode_path))
            model.train()

            model.eval()

            # val refinement model
            model2.eval()
            dice_sample2 = test_3d_patch.var_all_case_RNCL(
                args.root_path,
                model, model2,
                model_choose=1,
                num_classes=num_classes,
                patch_size=patch_size,
                stride_xy=56, stride_z=40,
                has_feature=True,
                fuse_feature1=True
             )
            if dice_sample2 > best_dice2:
                best_dice2 = round(dice_sample2, 4)
                save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}_diffu.txt'.format(iter_num, best_dice2))
                save_best_path = os.path.join(snapshot_path, 'best_model_diffu.pth'.format(args.model))
                with open(save_mode_path, 'w')as f:
                    f.write(f'Validation on model1: Dice={best_dice2:.4f}')
                save_net_opt(model2, optimizer2, save_best_path, epoch_num)
                logging.info("save best refine model to {}".format(save_mode_path))
            model.train()
            model2.train()



def self_train(args, pre_snapshot_path, self_snapshot_path):
    # init writer
    os.makedirs(os.path.join(self_snapshot_path, 'log'), exist_ok=True)
    writer = SummaryWriter(os.path.join(self_snapshot_path, 'log'))


    model1 = create_guide_model()
    model2 = create_refinement_model()
    ema_model1 = create_guide_model(ema=True).cuda()
    weighted_mse = WeightedMSE()

    c_batch_size = args.batch_size//2
    trainset_lab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split=f'train_lab{args.labelnum}', logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)
    trainset_lab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split=f'train_lab{args.labelnum}', reverse=True, logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)
    trainset_unlab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split=f'train_unlab{args.labelnum}', logging=logging)
    unlab_loader_a = DataLoader(trainset_unlab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)
    trainset_unlab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split=f'train_unlab{args.labelnum}', reverse=True, logging=logging)
    unlab_loader_b = DataLoader(trainset_unlab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    optimizer = optim.Adam(model1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)

    pretrained_model1 = os.path.join(pre_snapshot_path, 'best_model.pth')
    pretrained_model2 = os.path.join(pre_snapshot_path, 'best_model_diffu.pth')

    load_net_opt(model1, optimizer, pretrained_model1)
    load_net_opt(model2, optimizer2, pretrained_model2)
    load_net(ema_model1, pretrained_model1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.self_max_iteration, eta_min=1e-6)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.self_max_iteration, eta_min=1e-6)

    model1.train()
    model2.train()
    ema_model1.train()

    logging.info("{} iterations per epoch".format(len(lab_loader_a)))
    iter_num = 0
    best_dice = 0.0
    best_dice2 = 0.0

    max_epoch = args.self_max_iteration // len(lab_loader_a) + 1
    iterator = tqdm(range(1, max_epoch), ncols=70)

    for epoch in iterator:
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, _), (unimg_b, _)) in enumerate(
                zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):

            img_a, lab_a, img_b, lab_b, unimg_a, unimg_b = to_cuda([img_a, lab_a, img_b, lab_b, unimg_a, unimg_b])

            with torch.no_grad():
                unoutput_a_1, _ = ema_model1(unimg_a)
                unoutput_b_1, _ = ema_model1(unimg_b)
                plab_a = get_cut_mask(unoutput_a_1, nms=1)
                plab_b = get_cut_mask(unoutput_b_1, nms=1)
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            mixl_img = unimg_a * img_mask + img_b * (1 - img_mask)
            mixu_img = img_a * img_mask + unimg_b * (1 - img_mask)
            l_label_batch_ori = plab_a * img_mask + lab_b * (1 - img_mask)
            ul_label_batch_ori = lab_a * img_mask + plab_b * (1 - img_mask)

            outputs_l, _ = model1(mixl_img)
            outputs_u, _ = model1(mixu_img)
            loss_l = mix_loss(outputs_l, plab_a.long(), lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            loss_u = mix_loss(outputs_u, lab_a, plab_b.long(), loss_mask, u_weight=args.u_weight)

            outputs_1_prob = torch.softmax(torch.concat([outputs_l, outputs_u], dim=0), dim=1)
            outputs_1_pred = (outputs_1_prob > 0.5)
            label_all = torch.concat([l_label_batch_ori.unsqueeze(1), ul_label_batch_ori.unsqueeze(1)], dim=0)
            weight_1 = get_weight(outputs_1_pred[:, 1:2, ...], label_all > 0.5, shift_step=args.shift_step,
                                  ratio=args.mse_ratio)

            label_all = torch.cat((1 - label_all, label_all), dim=1)
            loss_weight_mse_1 = weighted_mse(outputs_1_prob, label_all, weight_1)
            loss_model1 = (loss_l + loss_u) + loss_weight_mse_1

            combined_images = torch.cat([mixl_img, mixu_img], dim=0)
            combined_true_labels = label_all

            with torch.no_grad():
                coarse_logits, features2 = ema_model1(combined_images)
                coarse_prob = F.softmax(coarse_logits, dim=1)
                coarse_logits = torch.clamp(coarse_prob, 0, 1)

            output2, _ = model2(combined_images, coarse_logits,features2)
            output2 = torch.softmax(output2, dim=1)

            outputs_2_pred = (output2 > 0.5)
            weight_2 = get_weight(outputs_2_pred[:, 1:2, ...], label_all[:, 1:2, ...] > 0.5,
                                  shift_step=args.shift_step,
                                  ratio=args.mse_ratio)
            loss_weight_mse_2 = weighted_mse(output2, label_all, weight_2)
            loss_ce = F.cross_entropy(output2, label_all.float())
            loss_dice_2 = dice_loss_one_hot(output2, label_all.float())

            lambda_dice = args.lambda_dice
            loss_model2 = loss_ce + lambda_dice * loss_dice_2 + loss_weight_mse_2

            iter_num += 1

            optimizer.zero_grad()
            loss_model1.backward()
            optimizer.step()
            scheduler.step()
            optimizer2.zero_grad()
            loss_model2.backward()
            optimizer2.step()
            scheduler2.step()

            update_ema_variables(model1, ema_model1, 0.99)


            writer.add_scalar('Loss/Model1_Total_Loss', loss_model1.item(), iter_num)
            writer.add_scalar('Loss/Model2_Refiner_MSE_Loss', loss_model2.item(), iter_num)
            logging.info(
                f"Epoch {epoch}, Iter {iter_num}: Model1 Loss: {loss_model1.item():.4f}, Model2 Loss: {loss_model2.item():.4f}")
            if iter_num %200 ==0:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f"Iter {iter_num}: ... Loss: ..., LR: {current_lr:.1e}")
        if epoch % 10 == 0:
            ema_model1.eval()
            dice_sample = test_3d_patch.var_all_case_guide_model(args.root_path, ema_model1, num_classes=num_classes, patch_size=patch_size,
                                                                 stride_xy=56, stride_z=40)
            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                save_best_path = os.path.join(self_snapshot_path, 'best_model.pth')
                torch.save(ema_model1.state_dict(), save_best_path)
                save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.txt'.format(iter_num, best_dice))
                with open(save_mode_path,'w')as f:
                    f.write(f'Validation on model1: Dice={best_dice:.4f}')
                logging.info(f"Save new best segmentation model with Dice: {best_dice}")
            ema_model1.train()

            ema_model1.eval()
            model2.eval()
            dice_sample2 = test_3d_patch.val_all_case_DiffUV_parallel(
                args.root_path,
                ema_model1,                 # guide model
                model2,                     # refinement model
                num_classes=num_classes,
                patch_size=patch_size,      # Training patch size.
                stride_xy=56, stride_z=40,  # Shifting window size during testing.
                model_choose=1,             # Test model  0：Vnet(guide model) 1：RNCL model(refinement-model).
                has_feature=True,           # Ensure that the model output format is (prediction, features)
                num_workers=4,              # Parallel testing works
                fuse_feature1=True          # The FFAG model requires interaction between two models, so validation and test also needs to be matched accordingly.
            )[0]

            if dice_sample2 > best_dice2:
                best_dice2 = round(dice_sample2, 4)
                save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}_diffu.txt'.format(iter_num, round(dice_sample2, 4)))
                with open(save_mode_path, 'w')as f:
                    f.write(f'useless content')
                save_best_path = os.path.join(self_snapshot_path, 'best_model_diffu.pth')
                torch.save(model2.state_dict(), save_best_path)
                logging.info(f"Save new best refiner model with Dice: {best_dice2}")
            writer.add_scalar('Val/Model1 Dice', dice_sample, iter_num)
            writer.add_scalar('Val/Model1 best Dice', best_dice, iter_num)
            writer.add_scalar('Val/Model2 Dice', dice_sample2, iter_num)
            writer.add_scalar('Val/Model2 best Dice', best_dice2, iter_num)
            ema_model1.train()
            model2.train()


if __name__ == "__main__":
    print('RNCL runing.')
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    ## make logger file
    args.exp = pathlib.Path(__file__).name.replace('.py','')
    pre_snapshot_path = "./experiments/LA/{}/pre_train".format(args.exp)
    self_snapshot_path = "./experiments/LA/{}/self_train".format(args.exp)
    print("Starting Diff-ESSPIL training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    # copy train files
    shutil.copyfile(__file__, f"./experiments/LA/{args.exp}/{args.exp}.py")

    # -- Pre-training
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # -- Self-training
    logging.info(f'pre trian path:{pre_snapshot_path}')
    self_train(args, pre_snapshot_path, self_snapshot_path)

