import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from matplotlib.pylab import cm
import random


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        if self.isTrain:
            visual_names_A += ['noise_real_A', 'before_avg_heatmap', 'after_avg_heatmap']
            if opt.sp_attention:
                visual_names_A += ['sp_w']
            if opt.prob_add_patch > 0:
                visual_names_A += ['patchnoiseA', 'noise_patch']
            self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        else:
            if opt.only_fakephoto:
                self.visual_names = ['fake_B'] 
            else:
                visual_names_A += ['noise_real_A', 'before_avg_heatmap', 'after_avg_heatmap']
                visual_names_A += ['sp_w', 'jet_sp_w']
                self.visual_names = visual_names_A + visual_names_B

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.resnet_n_downsample,
                                        layerasfeat_in_bottle_neck = opt.layerasfeat_in_bottle_neck, opt=opt)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.resnet_n_downsample,
                                        layerasfeat_in_bottle_neck = opt.layerasfeat_in_bottle_neck)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), 
                                                                self.netG_B.parameters()),
                                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.only_fakephoto = opt.only_fakephoto


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.B_paths = input['B_paths' if AtoB else 'A_paths']
        self.noise = input['noise'].to(self.device)
        self.patchnoiseA = input['patchnoiseA'].to(self.device)
        if self.isTrain:
            self.epoch = input['epoch']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.noise_real_A = self.real_A
        if self.isTrain:
            assert self.opt.prob_add_noise >=0 and self.opt.prob_add_noise < 0.5
            dice = random.random()
            px = random.randint(0,78)
            py = random.randint(0,78)
            self.noise_patch = torch.ones(self.real_A.shape).cuda()
            self.noise_patch[:,:,px:px+50,py:py+50] = self.patchnoiseA[:,:,px:px+50,py:py+50]
            if dice < self.opt.prob_add_noise:
                self.noise_real_A = self.real_A * self.noise
            elif dice >= self.opt.prob_add_noise and dice <= self.opt.prob_add_noise + self.opt.prob_add_patch:
                self.noise_real_A = self.real_A * self.noise_patch

        self.fake_B, self.A_o_feat, self.A_c_feat, r_dic = self.netG_A({'x':self.noise_real_A}) 
        self.rec_A,  _, _, _ = self.netG_B(self.fake_B) 
        self.fake_A, _, _, _ = self.netG_B(self.real_B) 
        self.rec_B,  _, _, _ = self.netG_A(self.fake_A) 

        def jet_heatmap(heatmap, norm=False):
            if norm:
                heatmap = (heatmap - torch.min(heatmap)) / (torch.max(heatmap) - torch.min(heatmap)) #* 255.0
            heatmap = cm.jet(heatmap.squeeze().data.cpu().numpy())
            heatmap = torch.from_numpy(heatmap)
            heatmap = heatmap.transpose(1, 2).transpose(0, 1).unsqueeze(0)
            return heatmap

        if self.opt.sp_attention:
            sp_w = r_dic['sp_w']
            self.sp_w = torch.nn.functional.interpolate(sp_w, size=(128, 128), mode='bilinear')
            self.jet_sp_w = jet_heatmap(self.sp_w)

        if not self.only_fakephoto :
            self.before_avg_heatmap = self.A_o_feat
            self.before_avg_heatmap = torch.mean(self.before_avg_heatmap, 1).unsqueeze(0)
            self.before_avg_heatmap = jet_heatmap(self.before_avg_heatmap)[:,:3,:,:]
            self.before_avg_heatmap = torch.nn.functional.interpolate(self.before_avg_heatmap, size=(128, 128), mode='bilinear')

            self.after_avg_heatmap = self.A_c_feat
            self.after_avg_heatmap = torch.mean(self.after_avg_heatmap, 1).unsqueeze(0)
            self.after_avg_heatmap = jet_heatmap(self.after_avg_heatmap)[:,:3,:,:]
            self.after_avg_heatmap = torch.nn.functional.interpolate(self.after_avg_heatmap, size=(128, 128), mode='bilinear')


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A, _, _, _ = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B, _, _, _ = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
