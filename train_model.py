import argparse
import os
import numpy as np
import torchinfo

from data_process import TextDataset
from model import LossFn, OptimizedSemanticCommunicationSystem
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

torch.set_float32_matmul_precision('high')


def main():
    run = wandb.init(name="Train", project="Trang-DeepSC")
    if not os.path.exists("./trainedModel"):
        os.makedirs("./trainedModel")

    if not os.path.exists("./dataset"):
        os.makedirs("./dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ' + str(device).upper())

    batch_size = 256
    num_epoch = 2
    lr = 1e-3

    corpus_data = TextDataset()

    dataloader = DataLoader(corpus_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    input_size = corpus_data.num_classes

    K = 8
    save_path = './trainedModel/'
    os.makedirs(save_path, exist_ok=True)

    lossFn = LossFn().to(device)
    net = OptimizedSemanticCommunicationSystem(vocab_size=input_size, embed_dim=16, snr=12, K=K).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    torchinfo.summary(net, input_data=torch.randint(0, input_size, (batch_size, 100)).to(device))

    for epoch in range(num_epoch):
        train_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epoch}")
        net.train()
        for i, data in enumerate(train_bar):
            [inputs, sentence_length] = data  # get length of sentence without padding
            inputs = inputs.long().to(device, non_blocking=True)
            sentence_length = sentence_length.long().to(device, non_blocking=True)

            label = F.one_hot(inputs, num_classes=input_size).float().to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=True):
                s_predicted = net(inputs)
                loss = lossFn(s_predicted, label, sentence_length)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            train_bar.set_postfix(loss=loss.item())
            run.log({"loss": loss.item()})

        torch.save(net.state_dict(), save_path + f'model_{epoch}.pth')
    torch.save(net.state_dict(), os.path.join(save_path, 'model.pth'))
    wandb.save(os.path.join(save_path, 'model.pth'))
    print("All done!")


if __name__ == '__main__':
    main()
