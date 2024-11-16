import torch
import torch.nn.functional as F


def load_data():
    words = open("./makemore-rs/names.txt").read().splitlines()
    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}

    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs), torch.tensor(ys), stoi, itos


def train_model(xs, ys, epochs=100):
    num = xs.nelement()
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g, requires_grad=True)

    for k in range(epochs):
        xenc = F.one_hot(xs, num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
        print(f"Loss at epoch {k}: {loss.item()}")

        W.grad = None
        loss.backward()
        W.data += -50 * W.grad

    return W


def generate_names(W, itos, num_samples=5):
    g = torch.Generator().manual_seed(2147483647)

    for i in range(num_samples):
        out = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W
            counts = logits.exp()
            p = counts / counts.sum(1, keepdims=True)

            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=g
            ).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print("".join(out))


def main():
    xs, ys, stoi, itos = load_data()
    print(f"Number of examples: {xs.nelement()}")

    W = train_model(xs, ys)
    generate_names(W, itos)


if __name__ == "__main__":
    main()
