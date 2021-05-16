# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
from tqdm import tqdm
from Engaging_classifier import analyze_engagement
from persona_selector_no_decode import prepare_persona_selector, select_persona
import torch
import torch.nn.functional as F
# import torch.optim.adam as Adam
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from os.path import join
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer, BertModel, BertTokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_, pad_dataset, get_data_loaders
from utils import get_dataset, download_pretrained_model

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, arg, current_output=None):
    model.eval()
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
    # print("At sample_sequence, personality is ", personality)
    # print("At sample_sequence, len personality is ", len(personality))
    # print("personality is ", tokenizer.decode(chain(*personality)))
    # print("At sample_sequence, history is ", history)
    # print("At sample_sequence, len history is ", len(history))
    # print("history is ",tokenizer.decode(chain(* history)))
    # print()
    for i in range(arg.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)
        input_ids = torch.tensor(instance["input_ids"], device = arg.device_1).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device = arg.device_1).unsqueeze(0)
        # print("input_ids is ", input_ids)
        # print("token_type_ids is ", token_type_ids)
        # input_ids = input_ids.to("cuda:0")
        # token_type_ids = token_type_ids.to("cuda")
        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / arg.temperature
        logits = top_filtering(logits, top_k=arg.top_k, top_p=arg.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if arg.no_sample else torch.multinomial(probs, 1)
        if i < arg.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def prepare_chatbot(check_point):
    # parser = ArgumentParser()
    # parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    # arg = {
    class ARG:
        
        def __init__(self):
                
            self.dataset_path = ''
            self.dataset_cache = './dataset_cache'
            self.max_history = 2
            self.num_candidates = 2
            self.device = "cuda:0"
            self.device_1 = "cuda:0"
            if torch.cuda.device_count() > 1:
                self.device_1 = "cuda:1"  
            self.no_sample = False
            self.max_length = 20
            self.min_length = 1
            self.seed = 0
            self.temperature = 0.7
            self.top_k = 0
            self.top_p= 0.9
            self.distributed = False
            self.personality_permutations = 1
            self.local_rank = -1
            self.train_batch_size = 4
            self.valid_batch_size = 4
    arg = ARG()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)

    # random.seed(arg.seed)
    # torch.random.manual_seed(arg.seed)
    # torch.cuda.manual_seed(arg.seed)


    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) #if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(check_point)
    model = model_class.from_pretrained(check_point)
    interlucuter = model_class.from_pretrained(check_point)
    model.to(arg.device_1)
    interlucuter.to(arg.device_1)
    model.eval()
    interlucuter.eval()

    add_special_tokens_(model, tokenizer)
    add_special_tokens_(interlucuter, tokenizer)
    logger.info("Sample a personality")
    dataset = get_dataset(tokenizer, arg.dataset_path, arg.dataset_cache)
    personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
    personality = random.choice(personalities)
    logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
    # print('-------------------------------------------')
    # print(tokenizer.decode(chain(*personality)))
    # print('-------------------------------------------')
    return model, interlucuter, tokenizer, personalities, arg
def generate_loss(model, tokenizer, interlocuter, bert_model, bert_tokenizer, persona_selector, persona_pool, arg, input_ids, personalities):
    # history = []
    model.eval()
    interlocuter.eval()
    # print("In generate_history, data:", data)
    # data_encoded = []
    persona_history = []
    # for i_data in data[0].strip().split('.'):
    #     if (i_data.strip() != ''):
    #         tmp = i_data + '.'
    #         persona_history.append(["", tmp])  # [[pad_0, s0], [pad_0, s0], [pad_0, s0], [pad_0, s0]]
    #         # print(tmp)
    #         # data_encoded.append(tokenizer.encode(""))
    #         enc = tokenizer.encode_plus(
    #             text = tmp,
    #             add_special_tokens=True,
    #             max_length = 32,
    #             pad_to_max_length = True,
    #             return_attention_mask = False
    #         )
    #         # dialogue_enc.append(enc['input_ids'])
    #         data_encoded.append(enc['input_ids'])
    # if len(data_encoded) != arg.train_batch_size: # wrong input shape
    #     return "wrong shape", "wrong shape"
    inter_persona = []
    history_enc = []
    for i_batch in input_ids:
        # print('batch\n', tokenizer.decode(i_batch[0]))
        persona = []
        sen = []
        per = True
        for i in range(1, len(i_batch[0])):
            if per:
                if i_batch[0][i] == 50261:  # <speaker2>
                    # sen.append(int(50261))
                    per = False
                else:
                    persona.append(int(i_batch[0][i]))
            else:
                if i_batch[0][i] == 50260:  # <speaker1>
                    break
                sen.append(int(i_batch[0][i]))
        # print('persona\n', tokenizer.decode(persona))
        inter_persona.append(persona)
        history_enc.append([sen])
    # print('history s0')
    for i in range(len(history_enc)):
        persona_history.append(["", tokenizer.decode(history_enc[i][0])]) # [[pad_0, s0], [pad_0, s0], [pad_0, s0], [pad_0, s0]]
        # print(tokenizer.decode(history_enc[i][0]))
        # print(tokenizer.decode(inter_persona[i]))
    # exit()
    # index = 0
    for i in range(len(history_enc)):
        # history_enc[i] is [s0]
        # print(f"Persona of {i} is ", tokenizer.decode(inter_persona[i]))
        with torch.no_grad():
            s1 = sample_sequence([inter_persona[i]], history_enc[i], tokenizer, interlocuter, arg)
        # temp = tokenizer.decode(interlocuter_response, skip_special_tokens=True)
        # print("temp is ",temp)
        persona_history[i].append(tokenizer.decode(s1, skip_special_tokens=True)) # [[pad_0, s0, s1], [pad_0, s0, s1], [pad_0, s0, , s1], [pad_0, s0, , s1]]
        history_enc[i].append(s1) # [[s0, s1], [s0, s1], [s0, s1], [s0, s1]]
        
        # inter_persona.append(personality[1])
        # history_sentence.append(history)# [[s0, s1], [s0, s1], [s0, s1], [s0, s1]]
    
    # print("In select persona")
    # print("persona_history is :\n", persona_history)
    # print("persona_history is :\n", np.shape(persona_history))
    selected_persona, log_prob = select_persona(persona_selector, persona_pool, persona_history, bert_tokenizer, bert_model)
    # print("selected persona is ", selected_persona)
    personalities = [tokenizer.encode(persona) for persona in selected_persona]
    
    persona_history_2 = []
    for hist in persona_history:
        persona_history_2.append([hist[2]]) # [[s1], [s1], [s1], [s1]]
    
    for i in range(len(personalities)):
        with torch.no_grad():
            # s2
            s2 = sample_sequence([personalities[i]], history_enc[i], tokenizer, model, arg)
        history_enc[i].append(s2)# [[s0, s1, s2], [s0, s1, s2], [s0, s1, s2], [s0, s1, s2]]
        persona_history_2[i].append(tokenizer.decode(s2, skip_special_tokens=True)) # [[s1, s2], [s1, s2], [s1, s2], [s1, s2]]
        with torch.no_grad():
            s3 = sample_sequence([inter_persona[i]], history_enc[i], tokenizer, interlocuter, arg)
        history_enc[i].append(s3)# [[s0, s1, s2, s3], [s0, s1, s2, s3], [s0, s1, s2, s3], [s0, s1, s2, s3]]
        persona_history_2[i].append(tokenizer.decode(s3, skip_special_tokens=True)) # [[s1, s2, s3], [s1, s2, s3], [s1, s2, s3], [s1, s2, s3]]

    # print("In select persona")
    # print("persona_history_2 is :\n", persona_history_2)
    # print("persona_history_2 is :\n", np.shape(persona_history_2))
    selected_persona_2, log_prob = select_persona(persona_selector, persona_pool, persona_history_2, bert_tokenizer, bert_model)
    # print("selected persona 2 is ", selected_persona_2)  
    
    
    replies = []
    queries = []
    personalities_2 = [tokenizer.encode(persona) for persona in selected_persona_2]    
    for i in range(len(personalities_2)):
        with torch.no_grad():
            # s4
            s4 = sample_sequence([personalities_2[i]], history_enc[i], tokenizer, model, arg)
        history_enc[i].append(s4)# [[s0, s1, s2, s3, s4], [s0, s1, s2, s3, s4], [s0, s1, s2, s3, s4], [s0, s1, s2, s3, s4]]
        queries.append(tokenizer.decode(s4, skip_special_tokens=True))# [s4, s4, s4, s4]
        with torch.no_grad():
            # s5
            s5 = sample_sequence([inter_persona[i]], history_enc[i], tokenizer, interlocuter, arg)
        replies.append(tokenizer.decode(s5, skip_special_tokens=True))
    
    # print("Queries is :\n", queries)
    # print("Replies is :\n", replies)
    
    score = analyze_engagement(replies, queries)
    score = torch.tensor(score, device = arg.device)
    loss = 0
    
    # print("score is ", score)
    # print("log_prob is ", log_prob)
    loss += -sum((1-score) * log_prob)
    
    # for k in range(arg.train_batch_size):
    #     loss -= score[k] * log_prob[k]
    # print("loss is ", loss)
    return loss, score


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type = str, default="/work/b07u1234/Engaging_Chatbot/transfer-learning-conv-ai/gpt2_persona_model/")
    parser.add_argument("--batch_size", type = int, default=4) # should not be reivsed
    parser.add_argument("--epoch", type = int, default=3)
    parser.add_argument("--lr", type = float, default=0.01)
    parser.add_argument("--data_portion", type = int, default=10000)
    parser.add_argument("--save_dir", type = str, default="/work/b07u1234/Engaging_Chatbot/persona_selector_dir/")
    parser.add_argument("--dir_name", type = str, default="partial_data_lr_001_test")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir + args.dir_name, exist_ok=True)
    # Prepare anything we need for training
    
    model, interlucuter, tokenizer, persona_list, arg = prepare_chatbot(args.model_checkpoint)
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(arg, tokenizer)
    persona_selector, persona_pool = prepare_persona_selector()    
    optimizer = torch.optim.Adam(persona_selector.id_selector.parameters(), lr = args.lr)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    bert_model = BertModel.from_pretrained('bert-large-uncased')
    bert_model.eval()
    
    
    # Start training
    for i in range(args.epoch):
        loss_for_plot = []
        score_for_plot = []
        batch_count = 0
        for input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids in tqdm(train_loader):
            
            # data = [tokenizer.decode(input_ids[0][0][:]).split("<speaker")[0].replace("<bos>", "")]
            batch_count += 1
            loss, score = generate_loss(model, tokenizer, interlucuter, bert_model, bert_tokenizer, persona_selector, persona_pool, arg, input_ids, persona_list)
            if loss == "wrong shape":
                loss = 0
                continue
            score = list(score.detach().cpu().numpy())
            optimizer.zero_grad()
            # print(score)
            # print(log_prob)
            persona_selector.train()
            persona_selector.id_selector.train()
            loss.backward()
            optimizer.step()
            if batch_count % 3 == 0:
                score_for_plot.append(sum(score) / len(score))
                loss_for_plot.append(loss.item())
                print(f"Engaging score at {batch_count}th batch is ", sum(score) / len(score))
                print(f"loss at {batch_count}th batch is ", loss.item())
            # print("args.data_portion is ", args.data_portion)
            if batch_count == args.data_portion:
                print("Break")
                print("Break")
                print("Break")
                
                
                plt.plot(score_for_plot)
                plt.title("Engaging every 3 batch")
                plt.xlabel("batch")
                plt.ylabel("loss")
                plt.savefig(join(args.save_dir + args.dir_name,f"{i}th_epoch_loss.jpg"))
                plt.clf()
                
                plt.plot(loss_for_plot)
                plt.title("Loss every 3 batch")
                plt.xlabel("batch")
                plt.ylabel("loss")
                plt.savefig(join(args.save_dir + args.dir_name,f"{i}th_epoch_reward.jpg"))
                with open(join(args.save_dir + args.dir_name,"param.txt"), "w") as f:
                    f.writelines(f"Batch size is {args.batch_size}\n")
                    f.writelines(f"epoch is {args.epoch}\n")
                    f.writelines(f"LR is {args.lr}\n")
                    f.writelines(f"data_portion is {args.data_portion}\n")
                    f.writelines(f"Dir name is {args.dir_name}\n")
                    
                torch.save(persona_selector, join(args.save_dir + args.dir_name, f"{i}_epoch.pkl"))
                break
        
            
            # print("###########################################")
            # print("delete loss !!!!!!!!!!!!")
            # print(torch.cuda.memory_summary(device = torch.device("cuda:0")))
            # # print(torch.cuda.memory_summary(device = torch.device("cuda:1")))
            # print("###########################################")
            # log_prob.to(torch.device("cpu"))
            # score.to(torch.device("cpu"))
            # loss.to(torch.device("cpu"))
            
            # if batch_count % args.save_time_step == 0:
                # if !os.path.exists(join(args.save_dir + args.dir_name):
                
if __name__ == "__main__":
    main()    
    