import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from transformers import BertTokenizer, BertModel

device_0 = torch.device("cuda:0")
#===== Pytorch Model definition ======================
class PersonaSelector(nn.Module):
    def __init__(self):
        super(PersonaSelector, self).__init__()
        #input dim [# of turn * # of emb_size]

        # not using dropout for RL training stability
        self.id_selector = nn.Sequential(
             nn.Linear(1024*3, 1024*5),
            nn.ReLU(),
            nn.Linear(1024*5, 1024*5),
            nn.ReLU(),
            nn.Linear(1024*5, 1024*5),
            nn.ReLU(),
            nn.Linear(1024*5, 6732) # output dim is len(persona_sentences_set)
        )

    def forward(self, x):
        x = self.id_selector(x)
        out = F.log_softmax(x, dim=-1)   # output probabilities for each persona id
        return out

def prepare_persona_selector():
    #==========Training Prepare===========================
    persona_selector = PersonaSelector().cuda()
    persona_selector.train()
    persona_selector.id_selector.train()
    
    #==========setting IO=================================
    persona_data_path = './data/personachat_self_original.json'
    persona_data_file = open(persona_data_path)
    persona_data = json.load(persona_data_file)

    #==========read persona sentences=====================
    data_type_list = ['train', 'valid']
    persona_set = set()
    for data_type in data_type_list:
        count = 0
        for i in range(len(persona_data[data_type])):
            count += len(persona_data[data_type][i]['personality'])
            for i_sentence in persona_data[data_type][i]['personality']:
                persona_set.add(i_sentence)
        print(data_type, count)
    print('total # of persona: ', len(persona_set))
    persona_pool = sorted(list(persona_set))

    return persona_selector, persona_pool

def select_persona(persona_selector, persona_pool, history_sentences, tokenizer, model):
    persona_selector.train()
    persona_selector.to(device_0)
    model.to(device_0)
    model.eval()
    # print("history_sentences is \n :", history_sentences)
    # print("np.shape(history_sentences) is", np.shape(history_sentences))
    encoded_input = []
    for sentences in history_sentences:
        dialogue_enc = []
        for sen in sentences:
            enc = tokenizer.encode_plus(
                text = sen,
                add_special_tokens=True,
                max_length = 32,
                pad_to_max_length = True,
                return_attention_mask = False
            )
            dialogue_enc.append(enc['input_ids'])
        encoded_input.append(dialogue_enc)
    # print("history_sentences.squeeze(-1) is :\n", history_sentences.squeeze(-1))
    # print("history_sentences.squeeze(-1) is :\n", history_sentences.squeeze(-1))
    
    encoded_input = torch.tensor(encoded_input, device = "cuda:0")
    ps_input_arr = []
    for dialogue in encoded_input:
        logits = model(dialogue)
        if isinstance(logits, tuple):
            # logits = logits[0]
            temp = logits[0]
        ps_input = torch.mean(temp, 1).squeeze(1)
        ps_input = torch.cat((ps_input[0], ps_input[1], ps_input[2]), 0)
        ps_input_arr.append(ps_input)
        
    # model.
    ps_input_arr = torch.stack((ps_input_arr[0], ps_input_arr[1], ps_input_arr[2], ps_input_arr[3])).to(torch.device("cuda:0"))
    # print('ps_input_arr\n', ps_input_arr)
    # print('ps_input_arr.size()\n', ps_input_arr.size())

    chatbot_persona_pred = persona_selector(ps_input_arr)
    # print('chatbot_persona_pred\n', chatbot_persona_pred)
    log_prob = []
    count = 0
    selected_persona_id = np.argmax(chatbot_persona_pred.cpu().data.numpy(), axis=-1)
    # print('selected_persona_id: ', selected_persona_id)
    for id in selected_persona_id:
        log_prob.append(chatbot_persona_pred[count][id])
        count += 1
    selected_persona = [persona_pool[id] for id in selected_persona_id]
    log_prob = torch.tensor(log_prob, device = "cuda:0", requires_grad = True)
    
    
    # print('selected_persona: ', selected_persona)
    return selected_persona, log_prob


#===== testing functions =============================
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# model = BertModel.from_pretrained('bert-large-uncased')
# persona_selector, persona_pool = prepare_persona_selector()

# history_sentences = [['i love to listen to frank sinatra.', 'how is life going for you?', 'it is going well. i am not poor, my wife hates me.'], ['i am a older lady.', "how old are you? i'm 32", "i'm a 32 year old male."], ['i love to eat cheese.', 'my girlfriend just broke up with me.', 'i love to read. i can not afford a television'], ['i like to cook stews.', 'i love making french fries', 'i like to shop.']]

# selected_persona, log_prob = select_persona(persona_selector, persona_pool, history_sentences, tokenizer, model)
# print(select_persona, log_prob)





#===== sundry ========================================
# for i_history_sentences in [['hi.', 'hello.', 'hello world!'], ['hi hi.', 'hello. hello.', 'hi hello world!']]:
    # encoded_input = []
    # max_l = max(len(tokenizer.encode(sentences)) for sentences in history_sentences) 
    # print("Max l is ", max_l)
    # for sentences in history_sentences:
    #     # encoded_input.append([])
    #     enc = tokenizer.encode(sentences)
    #     if len(enc) < max_l:
    #         for i in range(max_l - len(enc)):
    #             enc.append(0)
    #     encoded_input.append(enc)
    #     # for sentence in sentences:
    #     #     enc = tokenizer.encode(sentence)
    #     #     if len(enc) < max_l:
    #     #         for i in range(max_l - len(enc)):
    #     #             enc.append(0)
    #     #     encoded_input[-1].append(enc)
    # # for sentences in history_sentences:
    # #     encoded_input.append([])
    # #     for sentence in sentences:
    # #         encoded_input[-1].append(tokenizer.encode(history_sentences, padding = True))
            
    # print(encoded_input)       
    # encoded_input = torch.tensor(encoded_input, device='cuda')
    # print('encoded_input:  \n', encoded_input)
    # print('encoded_input: ', np.shape(encoded_input))
    # selected_persona = []
    # # for history in encoded_input:
    #     # history = torch.tensor(history, device='cuda').squeeze(-1)
    #     # print(history)
    #     # print("history.size() is ", history.size())
    # logits = model(encoded_input)
    # if isinstance(logits, tuple):
    #     logits = logits[0]
    # print("logits.size() is ", logits.size())
    # ps_input = torch.mean(logits, 1)
    # # history_embeddings = torch.Tensor(np.array(history_embeddings_arr).flatten())
    # # print(history_embeddings)
    # # tensor([-0.3671, -0.8464, -0.4059,  ...,  0.1756, -1.2973,  0.7627])

    # # chatbot_persona_pred = persona_selector(history_embeddings.cuda())
    # chatbot_persona_pred = persona_selector(ps_input)
    # print("chatbot_persona_pred is ", chatbot_persona_pred)
    # print("chatbot_persona_pred is ", chatbot_persona_pred.size())
    # # tensor([0.0001, 0.0001, 0.0001,  ..., 0.0002, 0.0002, 0.0001], device='cuda:0',
    # #        grad_fn=<SoftmaxBackward>)

    # selected_persona_id = np.argmax(chatbot_persona_pred.cpu().data.numpy(), axis=-1)
    # print("selected_persona_id is ", selected_persona_id)
    # selected_persona = [persona_pool[id] for id in selected_persona_id]

    # log_prob = []
    # print('selected_persona_id: ', selected_persona_id)
    # print('selected_persona: ', selected_persona)
    # # selected_persona_id:  3647
    # # selected_persona:  i sleep most of the day .
    # # model.to("cpu")
    # # exit()
    # count = 0
    # for id in selected_persona_id:
    #     log_prob.append(chatbot_persona_pred[count][id])
    #     count += 1
    # log_prob = torch.tensor(log_prob, device = "cuda", requires_grad = True)
    # return selected_persona, log_prob