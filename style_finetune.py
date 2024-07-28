import os

import torchinfo
from torch.utils.data import DataLoader
import torch
from data_process import CorpusData, TextDataset
import numpy as np
from openai import OpenAI
from textgrad.engine.local_model_openai_api import ChatExternalClient
import textgrad as tg

from model import LossFn, OptimizedSemanticCommunicationSystem
from transformers import AutoTokenizer
import textgrad as tg


def main():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    corpus_data = TextDataset()
    input_size = corpus_data.num_classes
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    engine = ChatExternalClient(client=client, model_string='lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF')

    K = 8
    save_path = './trainedModel/'
    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 1e-3
    instruction = ("You will evaluate a reconstructed sentence. Made the sentence concise "
                   "a much as possible while keeping the original meaning.")
    loss_fn = tg.TextLoss(instruction, engine=engine)
    net = OptimizedSemanticCommunicationSystem(vocab_size=input_size, embed_dim=32, snr=12, K=K).to(device)
    optim = tg.TextLoss(list(net.parameters()), engine=engine)



if __name__ == '__main__':
    main()