import torch
import itertools, random
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torchvision.transforms as transforms
'''
    When adding or removing a model, you need to modify the following places' code.
    loss_names, model_names, optimizer, isTrain, optimize_parameters
'''


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
        if self.isTrain:
            self.loss_style_weight = opt.loss_style_weight
            self.loss_content_weight = opt.loss_content_weight
            self.pcploss_weight = opt.pcploss_weight
            self.L1loss_weight = opt.L1loss_weight

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'content', 'style', 'L1_rec', 'pcp']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'ref_image']
        visual_names_B = ['real_B']

        self.only_fakephoto = opt.only_fakephoto

        if self.isTrain :
            self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        else:
            if self.only_fakephoto:
                self.visual_names = ['fake_B']
            else:
                self.visual_names = ['real_A', 'fake_B', 'real_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'D_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # define networks
        self.net_cycleGAN = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 
                                        opt.resnet_n_downsample)

        netG_A_decoder = self.net_cycleGAN.module.decoder
        self.netG_A = networks.define_AdaINNet(self.gpu_ids, netG_A_decoder)
        
        def get_parameter_number(model, verbose=False):
            params = self.netG_A.parameters()
            result = {'sum_all':0, 'sum_need_grad':0}
            for k in params:
                result['sum_all'] += k.data.numel()
                if verbose:
                    print("k.data.shape={}".format(k.data.shape))
                if k.requires_grad:
                    result['sum_need_grad'] += k.data.numel()
                        
            return result

        if self.isTrain:  # define discriminators
            print('build:netD_A')
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.real_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), 
                                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'

        self.ref_image = input['ref_image'].to(self.device)
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            dice = random.random()
            input_dict = {'gray':self.real_A, 'ref':self.ref_image}
            if dice > 0.2:
                self.fake_B, self.loss_content, self.loss_style = self.netG_A(input_dict, withref=False)  # input gray with ref
                assert self.loss_content == 0.0
                assert self.loss_style == 0.0
            else:
                self.fake_B, self.loss_content, self.loss_style = self.netG_A(input_dict, withref=True)  # input gray with ref
                assert self.loss_content > 0.0
                assert self.loss_style > 0.0
        else:
            if self.only_fakephoto:
                input_dict = {'gray':self.real_A, 'ref':self.ref_image}
                self.fake_B, self.loss_content, self.loss_style = self.netG_A(input_dict, withref=False)  # input gray with ref
            else:
                input_dict = {'gray':self.real_A, 'ref':self.ref_image}
                self.fake_B, self.loss_content, self.loss_style = self.netG_A(input_dict, withref=True)  # input gray with ref

        if self.isTrain:
            self.fake_B_gray = (self.fake_B[:,0,:,:] * 0.299+ self.fake_B[:,1,:,:] * 0.587+ self.fake_B[:,2,:,:] * 0.114).repeat(1,3,1,1)

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

    def backward_G(self):
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        self.fake_B_gray = self.fake_B[:,0,:,:] * 0.299 + self.fake_B[:,1,:,:] * 0.587 + self.fake_B[:,2,:,:] * 0.114 
        self.fake_B_gray = torch.unsqueeze(self.fake_B_gray, 1).repeat(1,3,1,1)
        self.loss_L1_rec = torch.mean(torch.abs(self.fake_B_gray - self.real_A)) * self.L1loss_weight

        self.loss_content = self.loss_content * self.loss_content_weight * self.pcploss_weight
        self.loss_style = self.loss_style * self.loss_style_weight * self.pcploss_weight 

        self.loss_pcp = self.loss_content + self.loss_style
        self.loss_G = self.loss_G_A + self.loss_pcp + self.loss_L1_rec

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
