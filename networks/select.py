# Select deconvolution method

from networks.G.FP import FP
from networks.G.Wiener import Wiener

def select_G(params, args):
    if args.G_network == 'FP':
        return FP(params, args)
    elif args.G_network == 'Wiener':
        return Wiener(params, args)
    else:
        assert False, ("Unsupported generator network")
