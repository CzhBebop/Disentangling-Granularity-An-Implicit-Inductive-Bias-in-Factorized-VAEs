import torch
import lib.dist as dist
import lib.flows as flows
import vae_quant_Vstab


def load_model_and_dataset(checkpt):
    print('Loading model and dataset.')
    #checkpt = torch.load(checkpt_filename, map_location=lambda storage, loc: storage)
    args = checkpt['args']
    state_dict = checkpt['state_dict']

    # backwards compatibility
    if not hasattr(args, 'conv'):
        args.conv = False

    if not hasattr(args, 'dist') or args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = flows.FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()

    # model
    if hasattr(args, 'ncon'):
        # InfoGAN
        model = infogan.Model(
            args.latent_dim, n_con=args.ncon, n_cat=args.ncat, cat_dim=args.cat_dim, use_cuda=True, conv=args.conv)
        model.load_state_dict(state_dict, strict=False)
        vae = vae_quant.VAE(
            z_dim=args.ncon, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist, conv=args.conv)
        vae.encoder = model.encoder
        vae.decoder = model.decoder
    else:
        vae = vae_quant_Vstab.VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
                  include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss,
                  newTC=args.newTC, contraintInfo=args.contraintInfo, nur_num=args.nur_num, layer=args.layer)
        """vae = vae_quant_stable.VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss , newTC=args.newTC)"""
        vae.load_state_dict(state_dict, strict=False)

    # dataset loader
    loader = vae_quant_Vstab.setup_data_loaders(args)
    return vae, loader.dataset, args
