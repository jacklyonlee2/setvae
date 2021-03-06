import os
import wandb
import torch
from tqdm import tqdm

from utils import plot_samples
from metrics import compute_cd, compute_metrics


def _mask(x):
    B, N, C = x.shape
    return torch.zeros((B, N)).bool().to(x.device)


def _unmask(x, x_mask):
    B, N = x.size(0), (~x_mask).sum(-1)[0]
    return x[~x_mask].reshape((B, N, -1))


class Trainer:
    def __init__(
        self,
        net,
        device,
        batch_size,
        opt=None,
        sch=None,
        max_epoch=None,
        kl_warmup_epoch=None,
        log_every_n_step=None,
        val_every_n_epoch=None,
        ckpt_every_n_epoch=None,
        ckpt_dir=None,
    ):
        self.net = net.to(device)
        self.device = device
        self.batch_size = batch_size
        self.opt = opt
        self.sch = sch
        self.step = 0
        self.epoch = 0
        self.max_epoch = max_epoch
        self.kl_warmup_epoch = kl_warmup_epoch
        self.log_every_n_step = log_every_n_step
        self.val_every_n_epoch = val_every_n_epoch
        self.ckpt_every_n_epoch = ckpt_every_n_epoch
        self.ckpt_dir = ckpt_dir

    def _state_dict(self):
        return {
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "sch": self.sch.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "max_epoch": self.max_epoch,
            "kl_warmup_epoch": self.kl_warmup_epoch,
        }

    def _load_state_dict(self, state_dict):
        for k, m in {"net": self.net, "opt": self.opt, "sch": self.sch}.items():
            m and m.load_state_dict(state_dict[k])
        self.step, self.epoch, self.max_epoch, self.kl_warmup_epoch = map(
            state_dict.get,
            (
                "step",
                "epoch",
                "max_epoch",
                "kl_warmup_epoch",
            ),
        )

    def save_checkpoint(self):
        ckpt_path = os.path.join(self.ckpt_dir, f"{self.epoch}.pth")
        torch.save(self._state_dict(), ckpt_path)

    def load_checkpoint(self, ckpt_path=None):
        if not ckpt_path:  # Find last checkpoint in ckpt_dir
            ckpt_paths = [p for p in os.listdir(self.ckpt_dir) if p.endswith(".pth")]
            assert ckpt_paths, "No checkpoints found."
            ckpt_path = sorted(ckpt_paths, key=lambda f: int(f[:-4]))[-1]
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_path)
        self._load_state_dict(torch.load(ckpt_path))

    def _train_step(self, x, mu, std):
        o, o_mask, kls = self.net(x, _mask(x))
        o = _unmask(o, o_mask)
        cd_loss = compute_cd(o, x, reduce_func=torch.sum).mean()
        kl_loss = torch.stack(kls, dim=1).sum(dim=1).mean()
        if self.kl_warmup_epoch > 0:
            kl_loss *= min(1.0, (self.epoch + 1) / self.kl_warmup_epoch)
        return cd_loss + kl_loss

    def train(self, train_loader, val_loader):
        while self.epoch < self.max_epoch:

            if self.epoch % self.val_every_n_epoch == 0:
                metrics, samples = self.test(val_loader)
                wandb.log(
                    {
                        **metrics,
                        "samples": samples,
                        "step": self.step,
                        "epoch": self.epoch,
                    }
                )
            if self.epoch % self.ckpt_every_n_epoch == 0:
                self.save_checkpoint()

            self.net.train()
            with tqdm(train_loader) as t:
                for batch in t:
                    batch = [_.to(self.device) for _ in batch]

                    loss = self._train_step(*batch)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                    t.set_description(f"Epoch:{self.epoch}|Loss:{loss.item():.2f}")
                    if self.step % self.log_every_n_step == 0:
                        wandb.log(
                            {
                                "loss": loss.cpu(),
                                "step": self.step,
                                "epoch": self.epoch,
                            }
                        )

                    self.step += 1
            self.sch.step()
            self.epoch += 1

    def _test_step(self, x, mu, std):
        B, N, _ = x.shape
        o_size = torch.ones(B).to(x) * N
        o, o_mask, _, _ = self.net.sample(o_size)
        x, o = x * std + mu, o * std + mu  # denormalize
        return _unmask(o, o_mask), x

    def _test_end(self, o, x):
        metrics = compute_metrics(o, x, self.batch_size)
        samples = plot_samples(o)
        return metrics, samples

    @torch.no_grad()
    def test(self, test_loader):
        results = []
        self.net.eval()
        for batch in tqdm(test_loader):
            batch = [_.to(self.device) for _ in batch]
            results.append(self._test_step(*batch))
        results = [torch.cat(_, dim=0) for _ in zip(*results)]
        return self._test_end(*results)
