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


def knowledge_distillation_loss(student_logits, teacher_logits, inputs, T=2.0, alpha=0.5):
    """
    Compute the knowledge distillation loss.
    Args:
        student_logits: logits of the student model (shape: [batch_size, seq_len, vocab_size])
        teacher_logits: logits of the teacher model (shape: [batch_size, seq_len, vocab_size])
        inputs: true labels (shape: [batch_size, seq_len])
        T: temperature for softening probability distributions
        alpha: weight for balancing soft and hard targets
    """
    # Soft loss
    soft_targets = F.softmax(teacher_logits / T, dim=-1)
    soft_prob = F.log_softmax(student_logits / T, dim=-1)
    soft_targets_loss = -torch.sum(soft_targets * soft_prob, dim=-1).mean()

    # Hard loss
    hard_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), inputs.view(-1), ignore_index=0)

    return (alpha * T * T * soft_targets_loss) + ((1 - alpha) * hard_loss)


def main():
    run = wandb.init(name="Train_KnowledgeDistillation", project="Trang-DeepSC")
    if not os.path.exists("./trainedModel"):
        os.makedirs("./trainedModel")

    if not os.path.exists("./dataset"):
        os.makedirs("./dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ' + str(device).upper())

    batch_size = 256
    num_epoch = 50
    lr = 1e-3

    corpus_data = TextDataset()

    dataloader = DataLoader(corpus_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    input_size = corpus_data.num_classes

    K = 8
    save_path = './trainedModel/'
    os.makedirs(save_path, exist_ok=True)

    # Initialize teacher model (128 embed dim)
    teacher_model = OptimizedSemanticCommunicationSystem(vocab_size=input_size, embed_dim=128, snr=12, K=K).to(device)
    teacher_model.load_state_dict(torch.load('trainedModel/teacher_model.pth'))
    teacher_model.eval()

    # Initialize student model (32 embed dim)
    student_model = OptimizedSemanticCommunicationSystem(vocab_size=input_size, embed_dim=16, snr=12, K=K).to(device)
    student_model.load_state_dict(torch.load('trainedModel/model.pth'))

    optim = torch.optim.Adam(student_model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    torchinfo.summary(student_model, input_data=torch.randint(0, input_size, (batch_size, 100)).to(device))

    for epoch in range(num_epoch):
        train_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epoch}")
        student_model.train()
        for i, data in enumerate(train_bar):
            [inputs, sentence_length] = data  # get length of sentence without padding
            inputs = inputs.long().to(device, non_blocking=True)
            sentence_length = sentence_length.long().to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=True):
                # Get teacher predictions
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)

                # Get student predictions
                student_logits = student_model(inputs)

                # Compute knowledge distillation loss
                loss = knowledge_distillation_loss(student_logits, teacher_logits, inputs, T=2.0, alpha=0.5)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            train_bar.set_postfix(loss=loss.item())
            run.log({"loss": loss.item()})

        torch.save(student_model.state_dict(), save_path + f'student_model_{epoch}.pth')
    print("Knowledge distillation training completed!")


if __name__ == '__main__':
    main()