import copy
import csv
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pickle


# from utils import logging, optimize, sampling
from aigcmodel.utils import get_optimizer, save_image
from .unet import DiffusionUNet
from .wavelet import DWT, IWT
from pytorch_msssim import ssim
from .mods import HFRM


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device

        self.high_enhance0 = HFRM(in_channels=3, out_channels=64)
        self.high_enhance1 = HFRM(in_channels=3, out_channels=64)
        self.Unet = DiffusionUNet(config)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // 10
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            # TODO: Might need to revise the update equation for DDIM
            # c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            # Deterministic Implicit de-noising method
            c1 = torch.zeros_like(x)
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, x):
        data_dict = {}
        dwt, idwt = DWT(), IWT()

        input_img = x[:, :3, :, :]
        n, c, h, w = input_img.shape
        input_img_norm = data_transform(input_img)
        input_dwt = dwt(input_img_norm)

        input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]

        input_high0 = self.high_enhance0(input_high0)

        input_LL_dwt = dwt(input_LL)
        input_LL_LL, input_high1 = input_LL_dwt[:n, ...], input_LL_dwt[n:, ...]
        input_high1 = self.high_enhance1(input_high1)

        b = self.betas.to(input_img.device)

        t = torch.randint(low=0, high=self.num_timesteps, size=(input_LL_LL.shape[0] // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_LL_LL.shape[0]].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(input_LL_LL)

        if self.training:
            gt_img_norm = data_transform(x[:, 3:, :, :])
            gt_dwt = dwt(gt_img_norm)
            gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]

            gt_LL_dwt = dwt(gt_LL)
            gt_LL_LL, gt_high1 = gt_LL_dwt[:n, ...], gt_LL_dwt[n:, ...]

            # Forward Process
            x = gt_LL_LL * a.sqrt() + e * (1.0 - a).sqrt()
            noise_output = self.Unet(torch.cat([input_LL_LL, x], dim=1), t.float())
            denoise_LL_LL = self.sample_training(input_LL_LL, b)

            pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))

            pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            data_dict["input_high0"] = input_high0
            data_dict["input_high1"] = input_high1
            data_dict["gt_high0"] = gt_high0
            data_dict["gt_high1"] = gt_high1
            data_dict["pred_LL"] = pred_LL
            data_dict["gt_LL"] = gt_LL
            data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = pred_x
            data_dict["e"] = e

        else:
            denoise_LL_LL = self.sample_training(input_LL_LL, b)
            pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))
            pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            data_dict["pred_x"] = pred_x

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters is: {total_params}")
        # Serialize the state_dict to a byte stream
        model_state_dict = pickle.dumps(self.model.state_dict())
        # Get the size of the byte stream in bytes
        state_dict_size = sys.getsizeof(model_state_dict) / (1024**2)
        print(f"The size of the state_dict is: {state_dict_size} bytes")
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()

        self.optimizer, self.scheduler = get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        # checkpoint = utils.load_checkpoint(load_path, None)
        # self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        # self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        self.model = torch.load(load_path, map_location=self.device)
        # if ema:
        #     self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))

    def server_train(self, DATASET):
        train_loader, val_loader = DATASET.server_get_loaders()
        self.common_part_fun(train_loader, val_loader)

    def common_part_fun(self, train_loader, val_loader):
        formatted_time = time.strftime('%m%d_%H%M', time.localtime(time.time()))
        cudnn.benchmark = True
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        loss_list = []
        time_list = []
        min_loss = self.config.training.min_loss
        flag = False
        time_start = time.time()

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            cnt_t = 0
            loss_avg = 0.0
            for i, (x, y) in enumerate(train_loader):
                cnt_t += 1
                print(f"Current batch round is {cnt_t}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)

                output = self.model(x)

                noise_loss, photo_loss, frequency_loss = self.estimation_loss(x, output)
                loss = noise_loss + photo_loss + frequency_loss
                loss_avg += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

            loss_avg /= cnt_t
            loss_list.append(loss_avg)
            if loss_avg <= min_loss:
                total_time = time.time() - time_start
                flag = True
                min_loss = loss_avg
                time_list.append(total_time)
                print(f"Time elapsed is {total_time}, current min_loss and avg_loss is {min_loss} : {loss_avg}")

            if (epoch + 1) % 1 == 0:
                print(
                    "epoch:{}, lr:{:.6f}, all_loss:{:.4f}".format(epoch + 1, self.scheduler.get_last_lr()[0], loss_avg))

            if (epoch + 1) % self.config.training.validation_freq == 0 or flag:
                flag = False
                self.model.eval()
                self.sample_validation_patches(val_loader, epoch + 1, formatted_time)

                torch.save(self.model, os.path.join(self.config.data.ckpt_dir, f'{self.config.data.data_volume}_{self.config.model.model_size}model_latest.pth'))
            self.scheduler.step()
        self.save_loss_file(loss_list, "loss")
        self.save_loss_file(time_list, "time")

    def train_client(self, client_loader):
        self.model.eval()
        local_model = copy.deepcopy(self.model)
        local_model = local_model.to(self.device)
        local_model.train()

        optimizer, _ = get_optimizer(self.config, local_model.parameters())

        loss_avg = 0.0
        cnt_e, cnt_t = 0, 0
        for epoch in range(self.config.training.fl_local_epochs):
            cnt_e += 1
            for i, (x, y) in enumerate(client_loader):
                cnt_t += 1
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x = x.to(self.device)

                optimizer.zero_grad()
                output = local_model(x)
                noise_loss, photo_loss, frequency_loss = self.estimation_loss(x, output)
                loss = noise_loss + photo_loss + frequency_loss
                loss_avg += loss.item()
                loss.backward()
                optimizer.step()

        return local_model, loss_avg/(cnt_e*cnt_t)

    def running_model_avg(self, current, next, scale):
        if current == None:
            current = next
            for key in current:
                current[key] = current[key] * scale
        else:
            for key in current:
                current[key] = current[key] + (next[key] * scale)
        return current

    def fl_train(self, DATASET):
        formatted_time = time.strftime('%m%d_%H%M', time.localtime(time.time()))
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.fl_get_loaders()
        loss_list = []
        time_list = []
        min_loss = self.config.training.min_loss
        flag = False
        time_start = time.time()
        for epoch in range(self.config.training.n_epochs):
            print('starting federated learning round: ', epoch)
            self.model.eval()
            self.model = self.model.to(self.config.device)

            running_avg = None
            loss_avg = 0.0

            for i in range(self.config.training.fl_clients):
                # train local client
                print("round {}, starting client {}/{}".format(epoch, i + 1, self.config.training.fl_clients))
                local_model, local_loss = self.train_client(train_loader[i])

                loss_avg += local_loss * (1 / self.config.training.fl_clients)

                # add local model parameters to running average
                running_avg = self.running_model_avg(running_avg, local_model.state_dict(), 1 / self.config.training.fl_clients)

            # set global model parameters for the next step
            self.model.load_state_dict(running_avg)
            self.ema_helper.update(self.model)

            if (epoch+1) % 1 == 0:
                print("epoch:{}, all_loss:{:.4f}".format(epoch+1, loss_avg))
                loss_list.append(loss_avg)

            if loss_avg <= min_loss:
                total_time = time.time() - time_start
                flag = True
                min_loss -= 0.05
                time_list.append(total_time)
                print(f"Time elapsed is {total_time}, current min_loss and avg_loss is {min_loss} : {loss_avg}")

            if (epoch + 1) % self.config.training.validation_freq == 0 or flag:
                flag = False
                self.model.eval()
                self.sample_validation_patches(val_loader, epoch+1, formatted_time)

            torch.save(self.model, os.path.join(self.config.data.ckpt_dir, f'{self.config.training.t_method}_model_latest.pth'))

        self.save_loss_file(loss_list, "loss")
        self.save_loss_file(time_list, "time")

    def single_fl_train(self, DATASET):
        train_loader, val_loader = DATASET.fl_get_loaders()
        self.common_part_fun(train_loader[0], val_loader)

    def estimation_loss(self, x, output):

        input_high0, input_high1, gt_high0, gt_high1 = output["input_high0"], output["input_high1"],\
                                                       output["gt_high0"], output["gt_high1"]

        pred_LL, gt_LL, pred_x, noise_output, e = output["pred_LL"], output["gt_LL"], output["pred_x"],\
                                                  output["noise_output"], output["e"]

        gt_img = x[:, 3:, :, :].to(self.device)
        # =============noise loss==================
        noise_loss = self.l2_loss(noise_output, e)

        # =============frequency loss==================
        frequency_loss = 0.1 * (self.l2_loss(input_high0, gt_high0) +
                                self.l2_loss(input_high1, gt_high1) +
                                self.l2_loss(pred_LL, gt_LL)) +\
                         0.01 * (self.TV_loss(input_high0) +
                                 self.TV_loss(input_high1) +
                                 self.TV_loss(pred_LL))

        # =============photo loss==================
        content_loss = self.l1_loss(pred_x, gt_img)
        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0).to(self.device)

        photo_loss = content_loss + ssim_loss

        return noise_loss, photo_loss, frequency_loss

    def sample_validation_patches(self, val_loader, step, formatted_time):

        image_folder = os.path.join(self.args.image_folder, f"{self.config.data.val_dataset}_{self.config.training.t_method}_{formatted_time}")
        self.model.eval()
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):

                b, _, img_h, img_w = x.shape
                img_h_32 = int(32 * np.ceil(img_h / 32.0))
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

                out = self.model(x.to(self.device))
                pred_x = out["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w]
                save_image(pred_x, os.path.join(image_folder, str(step), f"{y[0]}.png"))

    def save_loss_file(self, loss_list, save_type):
        formatted_time = time.strftime('%m%d_%H%M%S', time.localtime(time.time()))

        file_folder = os.path.join(self.args.image_folder,
                                   f"{self.config.data.val_dataset}_{self.config.training.t_method}")
        file_name = os.path.join(file_folder, f"{save_type}_{formatted_time}_{self.config.training.fl_local_epochs}.csv")

        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow([x]) for x in loss_list]

        print(f"===>File'{file_name}' has been saved!<===")

