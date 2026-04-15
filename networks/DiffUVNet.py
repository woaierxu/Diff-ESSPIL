import torch
import torch.nn as nn

from networks.RNCL_Block.unet.basic_unet_denose import BasicUNetDe, BasicUNetDe_crossAttention, \
    BasicUNetDe_GateFusion, BasicUNetDe_FiLMFusion, GatedFusion, BasicUNetDe_GateFusion_removeFteacher, \
    BasicUNetDe_GateFusion_removeFimage, BasicUNetDe_GateFusion_removeFcombine, \
    BasicUNetDe_GateFusion_keepFcombine_only, BasicUNetDe_GateFusion_removeFcombine_IG
from networks.RNCL_Block.unet.basic_unet import BasicUNetEncoder
from networks.RNCL_Block.guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, \
    ModelVarType, LossType, _extract_into_tensor
from networks.RNCL_Block.guided_diffusion.respace import SpacedDiffusion, space_timesteps
from networks.RNCL_Block.guided_diffusion.resample import UniformSampler, UniformSampler_interval
import torch.nn.functional as F

from utils.losses import dice_loss_one_hot

class DiffUnet_predMask_FuseFeature(nn.Module):
    def __init__(self, in_channels,
                 num_classes,
                 num_filters ,
                 timesteps = 10,
                 infer_start_timestep = 1,
                 train_time_step = 1000,
                 fuse_mode = 'gate') -> None:
        super().__init__()
        self.num_classes = num_classes
        self.infer_start_timestep = infer_start_timestep
            # pass
        # ori:[64, 64, 128, 256, 512, 64]
        self.embed_model = BasicUNetEncoder(3, in_channels, num_classes, num_filters)
        if fuse_mode == 'gate':
            self.model = BasicUNetDe_GateFusion(3, in_channels + num_classes, num_classes, num_filters,
                                                act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
        elif fuse_mode == 'film':
            self.model = BasicUNetDe_FiLMFusion(3, in_channels + num_classes, num_classes, num_filters,
                                                act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", train_time_step)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(train_time_step, [train_time_step]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(train_time_step, [timesteps]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(train_time_step)

    def forward(self, image, label, embeddings2 = None, need_noise_pred_label = False,t_force = None,
                writer = None, iter_num = None, # visualization for gate weight
                t_step_features = False
                ):
        noise = torch.randn_like(label).to(label.device)
        if t_force==None:
            t, weight = self.sampler.sample(label.shape[0], label.device)
        else:
            t = torch.full((label.shape[0],), t_force, device=label.device, dtype=torch.long)
        noise_label = self.diffusion.q_sample(label, t, noise=noise)
        # return noise_label, t, noise
        embeddings = self.embed_model(image)
        if embeddings2==None:
            embeddings2 = []
            for embed in embeddings:
                embeddings2.append(torch.randn_like(embed))
        out_label = self.model(noise_label, t=t, image=image,
                               embeddings=embeddings,embeddings2 = embeddings2,
                               writer = writer, iter_num = iter_num)
        if need_noise_pred_label:
            return out_label, embeddings, noise_label,t
        return out_label ,embeddings

    def inference(self,image,label,embeddings2,start_timestep = None):
        if start_timestep == None:
            start_timestep = self.infer_start_timestep
        range_timestep = self.diffusion.num_timesteps // self.sample_diffusion.num_timesteps

        num_t = range_timestep * start_timestep
        t = torch.full((label.shape[0],), num_t, device=label.device, dtype=torch.long)
        noise = torch.randn_like(label).to(label.device)
        noise_label = self.diffusion.q_sample(label, t, noise=noise)
        # return noise_label, t, noise
        embeddings = self.embed_model(image)
        pred_label = self.model(noise_label, t=t, image=image, embeddings=embeddings, embeddings2=embeddings2)
        return pred_label, None


