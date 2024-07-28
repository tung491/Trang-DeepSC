import argparse
import pickle
from datetime import datetime
import time

import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

import wandb

from data_process import CorpusData, TextDataset
from model import OptimizedSemanticCommunicationSystem
# ignore all warnings
import warnings

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate text model')
    parser.add_argument('--data_path', type=str, default='text_dataset', help='Path to evaluation data')
    parser.add_argument('--model_path', type=str, default="trainedModel/model.pth", help='Path to model')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on (cpu or cuda)')
    parser.add_argument('--snr', type=int, default=12, help='Signal to noise ratio')
    return parser.parse_args()


def padding_hiddden_state(hidden_state, max_length=256):
    pad_amount_input = (0, 0, 0, max_length - hidden_state.shape[1])
    padded_input_pooler = F.pad(hidden_state, pad_amount_input)
    return padded_input_pooler


def calculate_sem_similarity(bert_input, bert_output):
    return torch.sum(bert_input * bert_output) / (
            torch.sqrt(torch.sum(bert_input * bert_input))
            * torch.sqrt(torch.sum(bert_output * bert_output)))


def calBLEU(n_gram, s_predicted, s, length):
    weights = [1 / n_gram] * n_gram
    BLEU = nltk.translate.bleu_score.sentence_bleu([s[:length]], s_predicted[:length], weights=weights)
    return BLEU


def main():
    args = parse_args()
    run = wandb.init(name=f"Text_Eval", project=f'Trang-DeepSC',
                     config=vars(args))
    device = args.device
    print("Config:", dict(run.config))
    test_snr_range = np.arange(-6, 19, 3)
    tokenizer = BertTokenizer.from_pretrained('sentence-transformers/msmarco-bert-base-dot-v5')
    bert_model = BertModel.from_pretrained('sentence-transformers/msmarco-bert-base-dot-v5').to(device)

    corpus_data = TextDataset(split="test")
    batch_size = args.batch_size

    input_size = corpus_data.num_classes
    net = OptimizedSemanticCommunicationSystem(vocab_size=input_size, embed_dim=16, snr=12, K=8).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()

    BLEU_1_SS_per_testSNR = []
    BLEU_2_SS_per_testSNR = []
    BLEU_3_SS_per_testSNR = []
    BLEU_4_SS_per_testSNR = []
    SS_per_testSNR = []
    dataloader = DataLoader(corpus_data, batch_size=batch_size, shuffle=False)

    for test_snr in test_snr_range:
        print("Evaluating model with SNR:", test_snr)

        net.set_snr(test_snr)

        BLEU_1_list = []
        BLEU_2_list = []
        BLEU_3_list = []
        BLEU_4_list = []
        semantic_similarity_list = []

        train_bar = tqdm(dataloader)
        for batch_idx, data in enumerate(train_bar):
            # if batch_idx >= 8:
            #     break

            sentence_list = []
            sentence_length_list = []

            inputs, sentence_lenths = data
            for i in range(len(data)):
                sentence_ids = inputs[i]
                sentence_length = sentence_lenths[i]
                sentence = corpus_data.convert_id_sentence_to_word(sentence_ids, sentence_length)
                sentence_list.append(sentence)
                sentence_length_list.append(sentence_length)
            inputs = inputs.to(device)

            s_predicted = net(inputs)
            s_predicted = torch.argmax(s_predicted, dim=2)

            for i in range(len(data)):
                sentence = sentence_list[i]
                sentence_length = sentence_length_list[i]

                output_as_id = s_predicted[i, :]  # get the id list of most possible word
                origin_sentence_as_id = inputs[i, :]

                BLEU1 = calBLEU(1, output_as_id.cpu().detach().numpy(), origin_sentence_as_id.cpu().detach().numpy(),
                                sentence_length)
                BLEU_1_list.append(BLEU1)

                if sentence_length >= 2:
                    BLEU2 = calBLEU(2, output_as_id.cpu().detach().numpy(),
                                    origin_sentence_as_id.cpu().detach().numpy(), sentence_length)
                    BLEU_2_list.append(BLEU2)

                    if sentence_length >= 3:
                        BLEU3 = calBLEU(3, output_as_id.cpu().detach().numpy(),
                                        origin_sentence_as_id.cpu().detach().numpy(), sentence_length)
                        BLEU_3_list.append(BLEU3)

                        if sentence_length >= 4:
                            BLEU4 = calBLEU(4, output_as_id.cpu().detach().numpy(),
                                            origin_sentence_as_id.cpu().detach().numpy(),
                                            sentence_length)  # calculate BLEU
                            BLEU_4_list.append(BLEU4)

                output_sentence = corpus_data.convert_id_sentence_to_word(sentence_as_id=output_as_id,
                                                                          sentence_length=sentence_length)
                with torch.no_grad():
                    encoded_input = tokenizer(sentence, return_tensors='pt').to(device)  # encode sentence to fit bert model
                    bert_input = padding_hiddden_state(
                        bert_model(**encoded_input).last_hidden_state)
                    encoded_output = tokenizer(output_sentence, return_tensors='pt').to(device)
                    bert_output = padding_hiddden_state(bert_model(**encoded_output).last_hidden_state)
                    semantic_similarity = calculate_sem_similarity(bert_input, bert_output)
                    semantic_similarity_list.append(semantic_similarity.cpu().numpy())

        avg_BLEU_1 = np.mean(BLEU_1_list)
        avg_BLEU_2 = np.mean(BLEU_2_list)
        avg_BLEU_3 = np.mean(BLEU_3_list)
        avg_BLEU_4 = np.mean(BLEU_4_list)
        avg_SS = np.mean(semantic_similarity_list)

        BLEU_1_SS_per_testSNR.append(avg_BLEU_1)
        BLEU_2_SS_per_testSNR.append(avg_BLEU_2)
        BLEU_3_SS_per_testSNR.append(avg_BLEU_3)
        BLEU_4_SS_per_testSNR.append(avg_BLEU_4)
        SS_per_testSNR.append(avg_SS)

        print("Result of SNR:", test_snr)
        print('BLEU 1 = {}'.format(avg_BLEU_1))
        print('BLEU 2 = {}'.format(avg_BLEU_2))
        print('BLEU 3 = {}'.format(avg_BLEU_3))
        print('BLEU 4 = {}'.format(avg_BLEU_4))
        print('Semantic Similarity = {}'.format(avg_SS))
        run.log(
            {'BLEU_1': avg_BLEU_1, 'BLEU_2': avg_BLEU_2, 'BLEU_3': avg_BLEU_3, 'BLEU_4': avg_BLEU_4, "SNR": test_snr})

    latencies = []
    with torch.no_grad() and torch.cuda.amp.autocast(enabled=True):
        for _ in range(1000):
            start = time.time()
            net(inputs)
            end = time.time()
            latencies.append(end - start)
    print("Inference latency:", np.mean(latencies))


if __name__ == '__main__':
    main()
