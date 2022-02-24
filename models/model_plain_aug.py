from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G, define_A, define_D
from models.model_base import ModelBase
from models.loss import CharbonnierLoss, GANLoss, PerceptualLoss
from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip

"""
Author: Varun Jois
"""


class ModelPlainAug(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelPlainAug, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        self.netA = define_A(opt)
        self.netA = self.model_to_device(self.netA)
        # self.netAD = define_D(opt)
        # self.netAD = self.model_to_device(self.netAD)
        self.hard_ratio = opt['train']['hard_ratio_start']
        self.augmentation_wt = 1
        self.batch_size = opt['datasets']['train']['dataloader_batch_size']
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.netA.train()
        # self.netAD.train()
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log
        self.log_dict['G_loss_epoch'] = 0
        self.log_dict['A_loss_epoch'] = 0
        self.log_dict['G_loss_epoch'] = 0
        self.log_dict['L1_L'] = 0
        self.log_dict['L1_LA'] = 0
        # self.log_dict['F_loss_epoch'] = 0
        # self.log_dict['AD_loss_epoch'] = 0
        # self.log_dict['AD_loss_aug'] = 0
        # self.log_dict['l_d_real'] = 0
        # self.log_dict['l_d_fake'] = 0

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')

        load_path_A = self.opt['path']['pretrained_netA']
        if load_path_A is not None:
            print('Loading model for A [{:s}] ...'.format(load_path_A))
            self.load_network(load_path_A, self.netA, strict=self.opt_train['A_param_strict'], param_key='params')

        # load_path_AD = self.opt['path']['pretrained_netAD']
        # if load_path_AD is not None:
        #     print('Loading model for AD [{:s}] ...'.format(load_path_AD))
        #     self.load_network(load_path_AD, self.netAD, strict=self.opt_train['AD_param_strict'], param_key='params')

        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

        load_path_optimizerA = self.opt['path']['pretrained_optimizerA']
        if load_path_optimizerA is not None and self.opt_train['A_optimizer_reuse']:
            print('Loading optimizerA [{:s}] ...'.format(load_path_optimizerA))
            self.load_optimizer(load_path_optimizerA, self.A_optimizer)

        # load_path_optimizerAD = self.opt['path']['pretrained_optimizerAD']
        # if load_path_optimizerAD is not None and self.opt_train['AD_optimizer_reuse']:
        #     print('Loading optimizerAD [{:s}] ...'.format(load_path_optimizerAD))
        #     self.load_optimizer(load_path_optimizerAD, self.AD_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        self.save_network(self.save_dir, self.netA, 'A', iter_label)
        # self.save_network(self.save_dir, self.netAD, 'AD', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
        if self.opt_train['A_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.A_optimizer, 'optimizerA', iter_label)
        # if self.opt_train['AD_optimizer_reuse']:
        #     self.save_optimizer(self.save_dir, self.AD_optimizer, 'optimizerAD', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

        # augmentor loss
        # def augmentor_loss(E, E_A, H):
        #     loss_E = self.G_lossfn(E, H)
        #     loss_E_A = self.G_lossfn(E_A, H)
        #     aug_loss = torch.abs(1.0 - torch.exp(loss_E_A - self.hard_ratio * loss_E))
        #     loss = loss_E_A + self.augmentation_wt * aug_loss
        #     return loss
        # self.A_lossfn = augmentor_loss

        # perceptual loss for augmentor
        # self.F_lossfn = PerceptualLoss(feature_layer=[8, 35], weights=[1.05, -0.05], use_input_norm=False).to(self.device)
        # self.F_lossfn = PerceptualLoss(feature_layer=35, use_input_norm=False).to(self.device)

        # augmentor's discriminator loss
        # self.AD_lossfn = GANLoss(self.opt_train['gan_type'], 1.0, 0.0).to(self.device)
        # self.AD_lossfn_weight = self.opt_train['AD_lossfn_weight']
        # self.AD_update_ratio = self.opt_train['AD_update_ratio'] if self.opt_train['AD_update_ratio'] else 1
        # self.AD_init_iters = self.opt_train['AD_init_iters'] if self.opt_train['AD_init_iters'] else 0

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)

        # optimizer for the augmentor
        A_optim_params = []
        for k, v in self.netA.named_parameters():
            if v.requires_grad:
                A_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.A_optimizer = Adam(A_optim_params, lr=self.opt_train['A_optimizer_lr'], weight_decay=0)

        # optimizer for the augmentor's discriminator
        # AD_optim_params = []
        # for k, v in self.netAD.named_parameters():
        #     if v.requires_grad:
        #         AD_optim_params.append(v)
        #     else:
        #         print('Params [{:s}] will not optimize.'.format(k))
        # self.AD_optimizer = Adam(AD_optim_params, lr=self.opt_train['AD_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))
        # scheduler for the augmentor
        self.schedulers.append(lr_scheduler.MultiStepLR(self.A_optimizer,
                                                        self.opt_train['A_scheduler_milestones'],
                                                        self.opt_train['A_scheduler_gamma']
                                                        ))

        # scheduler for the augmentor's discriminator
        # self.schedulers.append(lr_scheduler.MultiStepLR(self.AD_optimizer,
        #                                                 self.opt_train['AD_scheduler_milestones'],
        #                                                 self.opt_train['AD_scheduler_gamma']
        #                                                 ))
    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L)

    # ----------------------------------------
    # feed L to netA
    # ----------------------------------------
    def netA_forward(self):
        self.L_A = self.netA(self.H)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        # torch.autograd.set_detect_anomaly(True)

        # # freezing the generator and discriminator graphs
        # for p in self.netA.parameters():
        #     p.requires_grad = True
        # # for p in self.netAD.parameters():
        # #     p.requires_grad = False
        # for p in self.netG.parameters():
        #     p.requires_grad = False


        # update hard_ratio
        epoch_to_update = 50
        if (current_step - 1) % ((800 / self.batch_size) * epoch_to_update) == 0:  # 200 steps is 1 epoch for div2k train and batch size of 4
            self.hard_ratio += 0.05
            print(f'Increased hard ratio to {self.hard_ratio}')
        #     self.F_lossfn.weights[0] -= 0.05
        #     self.F_lossfn.weights[1] += 0.05
        #     print(f'Weights are {self.F_lossfn.weights}')

        self.A_optimizer.zero_grad()
        self.L_A = self.netA(self.H)
        self.E = self.netG(self.L)
        self.E_A = self.netG(self.L_A)

        # calculate individual losses
        loss_E = self.G_lossfn(self.E, self.H)
        loss_E_A = self.G_lossfn(self.E_A, self.H)

        # get the loss from the discriminator
        # pred_fake = self.netAD(self.L_A)
        # AD_loss_aug = 0.5 * self.AD_lossfn(pred_fake, True)

        # Perceptual loss
        # F_loss = self.F_lossfn(self.L_A, self.L)

        # augmentor loss
        # A_loss = F_loss
        A_loss = loss_E_A + 2 * torch.abs(1.0 - torch.exp(loss_E_A - self.hard_ratio * loss_E))
        # A_loss = loss_E_A + torch.abs(1.0 - torch.exp(loss_E_A - self.hard_ratio * loss_E)) + AD_loss_aug
        # A_loss = torch.abs(1.0 - torch.exp(loss_E_A - self.hard_ratio * loss_E)) + F_loss / 10
        # A_loss = torch.exp(-(loss_E_A - loss_E))  # extreme loss
        A_loss.backward(retain_graph=True)
        self.A_optimizer.step()

        # freezing the augmentor and discriminator graphs
        # for p in self.netA.parameters():
        #     p.requires_grad = False
        # for p in self.netAD.parameters():
        #     p.requires_grad = True
        # for p in self.netG.parameters():
        #     p.requires_grad = False
        #
        # self.AD_optimizer.zero_grad()
        # # real
        # pred_d_real = self.netAD(self.L)
        # l_d_real = 0.5 * self.AD_lossfn(pred_d_real, True)
        # # fake
        # pred_d_fake = self.netAD(self.L_A.detach())  # 2) fake data, detach to avoid BP to G
        # l_d_fake = 0.5 * self.AD_lossfn(pred_d_fake, False)
        # AD_loss = l_d_real + l_d_fake
        # AD_loss.backward()
        # self.AD_optimizer.step()

        # optimize the generator (like this otherwise grads will be calculated for the Augmentor giving an error)
        # for p in self.netA.parameters():
        #     p.requires_grad = False
        # # for p in self.netAD.parameters():
        # #     p.requires_grad = False
        # for p in self.netG.parameters():
        #     p.requires_grad = True

        self.G_optimizer.zero_grad()
        G_loss = self.G_lossfn(self.netG(self.L), self.H) + self.G_lossfn(self.netG(self.L_A.detach()), self.H)
        # if current_step % 2 == 0:
        #     G_loss = self.G_lossfn(self.netG(self.L_A.detach()), self.H)
        # else:
        #     G_loss = self.G_lossfn(self.netG(self.L), self.H)
        # G_loss = G_loss
        G_loss.backward()
        # print(f'A after G back, should be unchanged: {self.netA.module.conv_last.weight[0][0][0].grad}')
        # print(f'G after G back, should be changed: {self.netG.module.conv_last.weight[0][0][0].grad}')

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        # G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        # if G_optimizer_clipgrad > 0:
        #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        # G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        # if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
        #     self.netG.apply(regularizer_orth)
        # G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        # if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
        #     self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['A_loss'] = A_loss.item()
        # self.log_dict['F_loss'] = F_loss.item()
        # self.log_dict['AD_loss'] = AD_loss.item()

        self.log_dict['G_loss_epoch'] += G_loss.item()
        self.log_dict['A_loss_epoch'] += A_loss.item()
        # self.log_dict['L1_L'] += loss_E.item()
        # self.log_dict['L1_LA'] += loss_E_A.item()
        # self.log_dict['F_loss_epoch'] += F_loss.item()
        # self.log_dict['AD_loss_epoch'] += AD_loss.item()
        # self.log_dict['AD_loss_aug'] += AD_loss_aug.item()
        # self.log_dict['l_d_real'] += l_d_real.item()
        # self.log_dict['l_d_fake'] += l_d_fake.item()

        self.log_dict['hard_ratio'] = self.hard_ratio

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()
        
        # self.netA.eval()
        # with torch.no_grad():
        #     self.netA_forward()
        # self.netA.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get epoch_stats
    # ----------------------------------------
    def get_epoch_stats(self):
        s = {'G_loss_epoch', 'A_loss_epoch', 'F_loss_epoch', 'AD_loss_epoch', 'AD_loss_aug', 'l_d_real',
             'l_d_fake', 'L1_L', 'L1_LA'}
        subset = {k: v for k, v in self.log_dict.items() if k in s}
        subset['hard_ratio'] = self.hard_ratio
        # message = ' '.join([f'{k:s}:{v:.3e}' for k, v in subset.items()])
        for k in subset:
            self.log_dict[k] = 0
        return subset

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        # out_dict['L_A'] = self.L_A.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg

    def update_hard_ratio(self):
        self.hard_ratio += 1
