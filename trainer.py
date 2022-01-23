import os
import wandb
import torch
import plotly as plt
from tqdm import tqdm

from metrics import *


def _mask(x):
    B, N, C = x.shape
    return torch.zeros((B, N)).bool().to(x.device)


def _unmask(x, x_mask):
    B, N = x.size(0), (~x_mask).sum(-1)[0]
    return x[~x_mask].reshape((B, N, -1))


def compute_loss(o, x, kls, epoch, kl_warmup_epoch):
    cd_loss = compute_cd(o, x, reduce_func=torch.sum).mean()
    kl_loss = torch.stack(kls, dim=1).sum(dim=1).mean()
    if kl_warmup_epoch > 0:
        kl_loss *= min(1.0, (epoch + 1) / kl_warmup_epoch)
    return cd_loss + kl_loss


@torch.no_grad()
def compute_metrics(x, y, batch_size):
    cd_xy, emd_xy = compute_pairwise_cd_emd(x, y, batch_size)
    cd_xx, emd_xx = compute_pairwise_cd_emd(x, x, batch_size)
    cd_yy, emd_yy = compute_pairwise_cd_emd(y, y, batch_size)
    mmd_cd, cov_cd = compute_mmd_cov(cd_xy)
    mmd_emd, cov_emd = compute_mmd_cov(emd_xy)
    acc_cd = compute_knn(cd_yy, cd_xy, cd_xx, k=1)
    acc_emd = compute_knn(emd_yy, emd_xy, emd_xx, k=1)
    return {
        "1NN-CD-acc": acc_cd.cpu(),
        "1NN-EMD-acc": acc_emd.cpu(),
        "cov-CD": cov_cd.cpu(),
        "cov-EMD": cov_emd.cpu(),
        "mmd-CD": mmd_cd.cpu(),
        "mmd-EMD": mmd_emd.cpu(),
    }


@torch.no_grad()
def plot_samples(samples, num=4, rows=2, cols=2):
    fig = plt.subplots.make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "Scatter3d"} for _ in range(cols)] for _ in range(rows)],
    )
    for i, sample in enumerate(samples[:num].cpu()):
        fig.add_trace(
            plt.graph_objects.Scatter3d(
                x=sample[:, 0],
                y=sample[:, 2],
                z=sample[:, 1],
                mode="markers",
                marker=dict(size=3, opacity=0.8),
            ),
            row=i // cols + 1,
            col=i % cols + 1,
        )
    fig.update_layout(showlegend=False)
    return fig


class Trainer:
    def __init__(
        self,
        net,
        opt,
        sch,
        max_epoch,
        kl_warmup_epoch,
        log_every_n_step,
        val_every_n_epoch,
        ckpt_every_n_epoch,
        batch_size,
        ckpt_dir,
        device,
    ):
        self.net = net.to(device)
        self.opt = opt
        self.sch = sch
        self.step = 0
        self.epoch = 0
        self.max_epoch = max_epoch
        self.kl_warmup_epoch = kl_warmup_epoch
        self.log_every_n_step = log_every_n_step
        self.val_every_n_epoch = val_every_n_epoch
        self.ckpt_every_n_epoch = ckpt_every_n_epoch
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_dir
        self.device = device

    def state_dict(self):
        return {
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "sch": self.sch.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "max_epoch": self.max_epoch,
            "kl_warmup_epoch": self.kl_warmup_epoch,
        }

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict["net"])
        self.opt.load_state_dict(state_dict["opt"])
        self.sch.load_state_dict(state_dict["sch"])
        (
            self.step,
            self.epoch,
            self.max_epoch,
            self.kl_warmup_epoch,
        ) = (state_dict[k] for k in ("step", "epoch", "max_epoch", "kl_warmup_epoch"))

    def save_checkpoint(self):
        ckpt_path = os.path.join(self.ckpt_dir, f"{self.epoch}.pth")
        torch.save(self.state_dict(), ckpt_path)

    def load_checkpoint(self, ckpt_path=None):
        if not ckpt_path:
            ckpt_paths = [f for f in os.listdir(self.ckpt_dir) if f.endswith(".pth")]
            assert len(ckpt_paths) > 0, "No checkpoints found."
            ckpt_path = sorted(ckpt_paths, key=lambda f: int(f[:-4]))[-1]
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_path)
        self.load_state_dict(torch.load(ckpt_path))

    def train_step(self, x, mu, std):
        o, o_mask, kls = self.net(x, _mask(x))
        o = _unmask(o, o_mask)
        return compute_loss(o, x, kls, self.epoch, self.kl_warmup_epoch)

    def train(self, train_loader, val_loader):
        while self.epoch < self.max_epoch:

            if self.epoch % self.val_every_n_epoch == 0:
                metrics, samples = self.test(val_loader)
                wandb.log({**metrics, "samples": samples, "epoch": self.epoch})

            if self.epoch % self.ckpt_every_n_epoch == 0:
                self.save_checkpoint()

            with tqdm(train_loader) as t:
                self.net.train()
                for batch in t:
                    loss = self.train_step(*(t.to(self.device) for t in batch))
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                    t.set_description(f"Epoch:{self.epoch}|Loss:{loss.item():.2f}")
                    if self.step % self.log_every_n_step == 0:
                        wandb.log(
                            {"loss": loss.cpu(), "step": self.step, "epoch": self.epoch}
                        )

                    self.step += 1
                self.sch.step()
            self.epoch += 1

    def test_step(self, x, mu, std):
        o_size = torch.ones(x.size(0)).to(x) * x.size(1)
        o, o_mask, _, _ = self.net.sample(o_size)
        x, o = x * std + mu, o * std + mu  # denormalize
        return _unmask(o, o_mask), x

    def test_end(self, o, x):
        metrics = compute_metrics(o, x, self.batch_size)
        samples = plot_samples(o)
        return metrics, samples

    @torch.no_grad()
    def test(self, test_loader):
        results = []
        self.net.eval()
        for batch in tqdm(test_loader):
            results.append(self.test_step(*(t.to(self.device) for t in batch)))
        return self.test_end(*(torch.cat(_, dim=0) for _ in zip(*results)))
