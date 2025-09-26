import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import math

class TinyShapes3DContinuous(Dataset):
    def __init__(self, factor_cards=(3,3,3,2,2,4), image_shape=(3,64,64), seed=0):
        self.factor_cards = factor_cards
        self.N = int(np.prod(factor_cards))
        self.image_shape = image_shape
        self.rng = np.random.RandomState(seed)
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        img = self.rng.randn(*self.image_shape).astype(np.float32)
        return torch.from_numpy(img)

class MockContinuousEncoder(torch.nn.Module):
    def __init__(self, H=16, W=16, seed=123):
        super().__init__()
        self.H = H
        self.W = W
        torch.manual_seed(seed)
        C, Hx, Wx = 3, 8, 8
        in_dim = C*Hx*Wx
        out_dim = H*W
        self.pool = torch.nn.AdaptiveAvgPool2d((Hx, Wx))
        self.lin = torch.nn.Linear(in_dim, out_dim, bias=False)
        with torch.no_grad():
            self.lin.weight.uniform_(-0.1, 0.1)
    @torch.no_grad()
    def encode(self, x):
        B = x.shape[0]
        y = self.pool(x)
        y = y.reshape(B, -1)
        z = self.lin(y)
        return z

def _digitize_by_quantile(x: np.ndarray, nbins: int) -> np.ndarray:
    if np.allclose(x, x[0]):
        return np.zeros_like(x, dtype=np.int64)
    qs = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(x, qs)
    edges = np.maximum.accumulate(edges)
    b = np.digitize(x, edges[1:-1], right=False).astype(np.int64)
    return b

def _digitize_by_range(x: np.ndarray, nbins: int) -> np.ndarray:
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if np.isclose(xmax, xmin):
        return np.zeros_like(x, dtype=np.int64)
    edges = np.linspace(xmin, xmax, nbins + 1)
    b = np.digitize(x, edges[1:-1], right=False).astype(np.int64)
    return b

def _entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    p = counts.astype(np.float64) + eps
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())

def _mi_discrete_from_bins(Z_bins: np.ndarray, group_masks, nbins_each):
    N, P = Z_bins.shape
    F = len(group_masks)
    Hz = np.zeros(P, dtype=np.float64)
    for j in range(P):
        Bj = nbins_each[j]
        cnt = np.bincount(Z_bins[:, j], minlength=Bj)
        Hz[j] = _entropy_from_counts(cnt)
    MI = np.zeros((F, P), dtype=np.float64)
    for f in range(F):
        Hz_given_v = np.zeros(P, dtype=np.float64)
        for a_mask in group_masks[f]:
            Na = int(a_mask.sum())
            if Na == 0:
                continue
            pa = Na / N
            Za = Z_bins[a_mask]
            for j in range(P):
                Bj = nbins_each[j]
                cnt = np.bincount(Za[:, j], minlength=Bj)
                Hz_a = _entropy_from_counts(cnt)
                Hz_given_v[j] += pa * Hz_a
        MI[f, :] = Hz - Hz_given_v
    return MI

import numpy as np, math

def code_perplexity(enc_idx, Kc):
    """看每个位置大概用了多少个 code（困惑度），≈1 说明基本只用同一个 code -> 崩塌"""
    N, P = enc_idx.shape
    perps = []
    for j in range(P):
        cnt = np.bincount(enc_idx[:, j], minlength=Kc)
        if cnt.sum() == 0:
            perps.append(0.0); continue
        p = cnt / cnt.sum()
        H = -(p[p > 0] * np.log(p[p > 0])).sum()
        perps.append(float(np.exp(H)))
    return float(np.mean(perps))

def compute_mig_from_indices(enc_indices_all, factor_cards):
    """
    enc_indices_all: (N, Ppos) 每列一个空间位置的 code 下标
    factor_cards: 因子基数 (例如 (3,3,3,2,2,4) 或真实 (10,10,10,8,4,15))
    返回: mig, per_factor_gaps(F,), MI(F,Ppos)
    """
    Z = np.asarray(enc_indices_all, dtype=np.int64)
    N, P = Z.shape
    F = len(factor_cards)
    assert int(np.prod(factor_cards)) == N

    def ent(counts, eps=1e-12):
        p = counts.astype(np.float64) + eps
        p /= p.sum()
        return float(-(p * np.log(p)).sum())

    # H(Z_j)
    Hz = np.zeros(P, dtype=np.float64)
    Kc = int(Z.max()) + 1
    for j in range(P):
        cnt = np.bincount(Z[:, j], minlength=Kc)
        Hz[j] = ent(cnt)

    # grid 分组
    idx_grid = np.arange(N).reshape(*factor_cards)
    MI = np.zeros((F, P), dtype=np.float64)
    for f, Cf in enumerate(factor_cards):
        Hz_giv = np.zeros(P, dtype=np.float64)
        for a in range(Cf):
            sl = [slice(None)] * F; sl[f] = a
            idxs = idx_grid[tuple(sl)].reshape(-1)
            Za = Z[idxs]
            pa = Za.shape[0] / N
            for j in range(P):
                cnt = np.bincount(Za[:, j], minlength=Kc)
                Hz_giv[j] += pa * ent(cnt)
        MI[f] = Hz - Hz_giv

    gaps = np.zeros(F, dtype=np.float64)
    for f, Cf in enumerate(factor_cards):
        vals = np.sort(MI[f])[::-1]
        top1, top2 = vals[0], (vals[1] if P > 1 else 0.0)
        gaps[f] = (top1 - top2) / math.log(Cf)

    mig = float(gaps.mean())
    return mig, gaps, MI


def compute_mig_by_binning(
    Z_continuous: np.ndarray,
    factor_cards=(3,3,3,2,2,4),
    provide_order="grid",
    factor_labels=None,
    binning="quantile",
    nbins=20,
):
    Z = np.asarray(Z_continuous)
    N, P = Z.shape
    F = len(factor_cards)

    if provide_order == "grid":
        assert int(np.prod(factor_cards)) == N, "N must equal product(factor_cards) for grid mode."

    Z_bins = np.zeros_like(Z, dtype=np.int64)
    nbins_each = [nbins]*P
    for j in range(P):
        x = Z[:, j]
        if binning == "quantile":
            b = _digitize_by_quantile(x, nbins)
        else:
            b = _digitize_by_range(x, nbins)
        Z_bins[:, j] = b
        if (b.max() == 0) and (b.min() == 0):
            nbins_each[j] = 1

    group_masks = []
    if provide_order == "grid":
        idx_grid = np.arange(N).reshape(*factor_cards)
        for f, Cf in enumerate(factor_cards):
            masks_f = []
            for a in range(Cf):
                sl = [slice(None)] * F
                sl[f] = a
                idxs = idx_grid[tuple(sl)].reshape(-1)
                mask = np.zeros(N, dtype=bool)
                mask[idxs] = True
                masks_f.append(mask)
            group_masks.append(masks_f)
    else:
        assert factor_labels is not None and len(factor_labels) == F
        for f, Cf in enumerate(factor_cards):
            v = np.asarray(factor_labels[f]).astype(int)
            masks_f = [(v == a) for a in range(Cf)]
            group_masks.append(masks_f)

    MI = _mi_discrete_from_bins(Z_bins, group_masks, nbins_each)

    per_factor_gaps = np.zeros(F, dtype=np.float64)
    for f, Cf in enumerate(factor_cards):
        vals = np.sort(MI[f])[::-1]
        top1 = vals[0]
        top2 = vals[1] if P > 1 else 0.0
        per_factor_gaps[f] = (top1 - top2) / math.log(Cf)

    mig = float(per_factor_gaps.mean())
    return mig, per_factor_gaps, MI
from disentanglement_lib.data.ground_truth import shapes3d,cars3d,mpi3d


class CustomTensorDataset_3d(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        img1 = self.data_tensor[index1]
        if self.transform is not None:
            img1 = self.transform(img1)
        return img1

    def __len__(self):
        return self.data_tensor.size(0)
class CustomTensorDataset_3dcar(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        img1 = self.data_tensor[index1]
        if self.transform is not None:
            img1 = self.transform(img1)
        return img1

    def __len__(self):
        return self.data_tensor.size(0)

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)
import os
def load_dsprite():
    root = 'E:/python3.8.10/disentanglement_lib-master/bin/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    if not os.path.exists(root):
        import subprocess
        print('Now download dsprites-dataset')
        subprocess.call(['./download_dsprites.sh'])
        print('Finished')
    data = np.load(root, encoding='bytes')
    data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
    train_kwargs = {'data_tensor':data}
    dset = CustomTensorDataset
    train_data = dset(**train_kwargs)
    return train_data


def load_3dcars():
    cars_data = cars3d.Cars3D()
    data = cars_data.images
    data = torch.from_numpy(data).float().transpose(1, 3).transpose(2, 3)
    train_kwargs = {'data_tensor':data}
    dset = CustomTensorDataset_3dcar
    train_data = dset(**train_kwargs)
    return train_data

def load_3dshapes():
    cars_data = shapes3d.Shapes3D()
    data = cars_data.images
    data = torch.from_numpy(data).float().transpose(1, 3).transpose(2, 3)
    train_kwargs = {'data_tensor':data}
    dset = CustomTensorDataset_3d
    train_data = dset(**train_kwargs)
    return train_data


def load_3dmpi():
    cars_data = mpi3d.MPI3D()
    data = cars_data.images
    data = torch.from_numpy(data).float().transpose(1, 3).transpose(2, 3)
    train_kwargs = {'data_tensor':data}
    dset = CustomTensorDataset_3d
    train_data = dset(**train_kwargs)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = DataLoader(dataset=train_data,
        batch_size=1000, shuffle=False, **kwargs)
    return train_loader


def computeMIG(COUNT = 0,DATANAME='car'):
    #DATANAME = "CAR"
    if DATANAME == "car":
        dataset = load_3dcars()
        factor_cards_demo = (4, 24, 183)#17,568
    elif DATANAME == "SHAPE":
        dataset = load_3dshapes()
        factor_cards_demo = (10, 10, 10, 8, 4, 15)#480,000
    elif DATANAME == "dsprite":
        dataset = load_dsprite()
        factor_cards_demo = (3, 6, 40, 32, 32)#737,280
    elif DATANAME == "3dmpi":
        dataset = load_3dmpi()
        factor_cards_demo = (4, 4, 2, 3, 3, 40, 40)#460,800
    # Run the pipeline

    #dataset = TinyShapes3DContinuous(factor_cards=factor_cards_demo, image_shape=(3,64,64), seed=7)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    MO = "QL"
    device = torch.device("cuda:0")
    if DATANAME == "dsprite":
        in_channels = 1
    else:
        in_channels = 3
    if MO == "VQ":

        from vqvae import VQVAE
        
        use_ema = True
        model_args = {
            "in_channels": in_channels,
            "num_hiddens": 128,
            "num_downsampling_layers": 4,
            "num_residual_layers": 2,
            "num_residual_hiddens": 32,
            "embedding_dim": 16,
            "num_embeddings": 64,
            "use_ema": use_ema,
            "decay": 0.99,
            "epsilon": 1e-5,
        }
        model = VQVAE(**model_args).to(device)
        model.load_state_dict(torch.load('E:\\python3.8.10\\disentanglement_metric\\vqvae_model\\best_model_'+DATANAME+'_'+str(COUNT)+'.pth'))
    elif MO == "QL":
        from qlae import Model
        model = Model().to(device)
        model.load_state_dict(torch.load('E:\\python3.8.10\\disentanglement_metric\\results\\shapes3d\\trial'+str(COUNT)+'\\model'+str(COUNT)+'.pt'))


    #model_n_vali = DualDecoderAutoencoder(latent_n,dim=3)
    #E:\\python3.8.10\\disentanglement_metric\\results\\shapes3d\\trial0\\model0.pt
    #E:\\python3.8.10\\disentanglement_metric\\vqvae_model\\best_model3.pth
    #encoder = MockContinuousEncoder(H=16, W=16, seed=123)

    model.eval()

    """Z_list = []
    with torch.no_grad():
        for xs in loader:
            xs = xs.view(xs.size(0), 3, 64, 64)
            #z = model.encode(xs)      # (B,256)
            if MO == "VQ":
                (z_quantized, _, _, _) = model.quantize(xs.cuda())
            else:

            #print("z_quantized",z_quantized.shape)
            z = z_quantized.reshape(xs.size(0), -1)
            #print(z.shape)
            Z_list.append(z.cpu().numpy())
    Z = np.concatenate(Z_list, axis=0)  # (N,256)

    stds = Z.std(axis=0)
    print("零方差维度数:", (stds < 1e-8).sum(), "/", Z.shape[1], " | 均值std:", stds.mean())

    mig, gaps, MI = compute_mig_by_binning(
        Z_continuous=Z,
        factor_cards=factor_cards_demo,
        provide_order="grid",
        binning="quantile",
        nbins=20
    )

    print("== Continuous-latent MIG via binning (demo) ==")
    print("factor_cards:", factor_cards_demo)
    print("Z shape:", Z.shape)
    print("Overall MIG:", mig)
    print("Per-factor MIG gaps:", gaps)
    print("MI matrix shape (F,P):", MI.shape)"""


    enc_idx_list = []
    with torch.no_grad():
        for xs in loader:
            xs = xs.to(device).view(xs.size(0), in_channels, 64, 64)
            if MO == "VQ":
                zq, _, _, enc_idx = model.quantize(xs)     # ★ 第四个返回值是 indices
                #print("enc_idx",enc_idx.shape)
                # enc_idx shape: (B, H'*W')，在你的设置下 H'=W'=4，因此 Ppos=16
                enc_idx_list.append(enc_idx.cpu().numpy())

            else:
                outs = model.forward(xs)
                enc_idx_list.append(outs["z_indices"].T.cpu().numpy())
                #print("outs",outs["z_indices"].T.shape)
    enc_idx_all = np.concatenate(enc_idx_list, axis=0)    # (N, 16)
    print("enc_idx_all shape:", enc_idx_all.shape)

    # 先看 code 使用是否崩塌
    if MO == "VQ":
        avg_perp = code_perplexity(enc_idx_all, Kc=model.vq.num_embeddings)
        print("平均 code perplexity:", avg_perp, "(最大可能:", model.vq.num_embeddings, ")")
        #model.vq.num_embeddings
        # 用“离散下标”计算 MIG（不用分箱了）
        mig_idx, gaps_idx, MI_idx = compute_mig_from_indices(enc_idx_all, factor_cards_demo)
        print("MIG (from indices):", mig_idx)
        print("Per-factor gaps:", gaps_idx)
        print("MI shape:", MI_idx.shape)
    else:
        avg_perp = code_perplexity(enc_idx_all, Kc=12)
        print("平均 code perplexity:", avg_perp, "(最大可能:", 12, ")")
        #model.vq.num_embeddings
        # 用“离散下标”计算 MIG（不用分箱了）
        mig_idx, gaps_idx, MI_idx = compute_mig_from_indices(enc_idx_all, factor_cards_demo)
        print("MIG (from indices):", mig_idx)
        print("Per-factor gaps:", gaps_idx)
        print("MI shape:", MI_idx.shape)
    return mig_idx



for DATASET in ["SHAPE"]:
    MIG_LIST = []
    for i in range(10):
        mig_idx = computeMIG(COUNT = i,DATANAME=DATASET)
        MIG_LIST.append(mig_idx)
    print(DATASET)
    print(MIG_LIST)