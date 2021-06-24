import transformers
import torch
import os
import json
import time
import random
import argparse
import numpy as np
from datetime import datetime
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F
from tqdm import tqdm
import subprocess
# from torchcrf import CRF
import wandb

from gpt2_model import GPT2LMHeadModel
from OT import IPOT_distance2

# wandb.init(project="attribute_label")

BOS = 50257
EOS = 50256
PAD_ID = 15636
MAX_LEN = 512
HIDDEN_SIZE = 768

# FIELD_LABEL_NUM = 6302 # wikibio dataset
# FIELD_LABEL_NUM = 134 # human 50 dataset
# FIELD_LABEL_NUM = 190 # human 100 dataset
# FIELD_LABEL_NUM = 259 # human 200 dataset
FIELD_LABEL_NUM = 457 # human 500 dataset
# FIELD_LABEL_NUM = 732 # human 2000 dataset
# FIELD_LABEL_NUM = 49 # books 50 dataset
# FIELD_LABEL_NUM = 73 # books 100 dataset
# FIELD_LABEL_NUM = 86 # books 200 dataset
# FIELD_LABEL_NUM = 108 # books 500 dataset
# FIELD_LABEL_NUM = 163 # books 2000 dataset
# FIELD_LABEL_NUM = 25 # songs 50 dataset
# FIELD_LABEL_NUM = 27 # songs 100 dataset
# FIELD_LABEL_NUM = 29 # songs 200 dataset
# FIELD_LABEL_NUM = 39 # songs 500 dataset
# FIELD_LABEL_NUM = 39 # songs 2000 dataset

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)

class FieldLabelClassify(nn.Module):
    def __init__(self):
        super(FieldLabelClassify, self).__init__()
        self.linear_classify = nn.Linear(HIDDEN_SIZE, FIELD_LABEL_NUM, bias=True)
        self.loss_fct = nn.CrossEntropyLoss()
        self.dropout = torch.nn.Dropout(0.1)
        # self.crf = CRF(FIELD_LABEL_NUM, batch_first=True)
    
    def forward(self, hidden_states, field_labels, label_masks, entities_attr=None, entities_list=None, entities_len=None, entities_mask=None):
        '''
        shift_labels = torch.masked_select(field_labels, label_masks.bool()).contiguous()
        label_masks = label_masks.unsqueeze(-1).expand(hidden_states.size())
        attribute_hidden_states = torch.masked_select(hidden_states, label_masks.bool()).contiguous()
        shift_logits = self.linear_classify(attribute_hidden_states.view(-1, HIDDEN_SIZE))
        mask_loss = self.loss_fct(shift_logits, shift_labels.view(-1))
        '''
        if entities_attr is None:
            # hidden_states = self.dropout(hidden_states)
            # batch_size x seq_length x tag_num
            mc_logits = self.linear_classify(hidden_states)
            # mask_loss = -1 * self.crf(mc_logits, field_labels, mask=label_masks.bool(), reduction='mean')
            shift_labels = torch.masked_select(field_labels, label_masks.bool()).contiguous()
            label_masks = label_masks.unsqueeze(-1).expand(mc_logits.size())
            shift_logits = torch.masked_select(mc_logits, label_masks.bool()).contiguous()
            mask_loss = self.loss_fct(shift_logits.view(-1, FIELD_LABEL_NUM), shift_labels.view(-1))
        else:
            # calculate the entity hidden by get the sum of hidden from same entity 
            batch_ent, s_len_batch, num_entities = entities_list.size()
            batch_len_ent, s_len_entities = entities_len.size()
            aeq(batch_len_ent, batch_ent)
            aeq(num_entities, s_len_entities)
            ent_hidden_states = hidden_states.unsqueeze(1).expand(-1, num_entities, -1, -1)
            ent_dim = entities_list.transpose(1, 2).unsqueeze(3).expand(-1, -1, -1, HIDDEN_SIZE)
            ent_len_dim = entities_len.unsqueeze(2).expand(-1, -1, HIDDEN_SIZE)
            ent_hidden = (ent_hidden_states * ent_dim).sum(2)
            ent_hidden = ent_hidden / ent_len_dim
            # map the hidden size to the label num
            mc_logits = self.linear_classify(ent_hidden)
            shift_labels = torch.masked_select(entities_attr, entities_mask).contiguous()
            entities_mask = entities_mask.unsqueeze(-1).expand(mc_logits.size())
            shift_logits = torch.masked_select(mc_logits, entities_mask).contiguous()
            mask_loss = self.loss_fct(shift_logits.view(-1, FIELD_LABEL_NUM), shift_labels.view(-1))
        return mask_loss

def rebuild_sent(line):
    ws = []
    for i, w in enumerate(line.split()):
        if w[-1] == ',':
            ws.append(w[:-1])
            ws.append(',')
        elif i == len(line.split()) - 1:
            if w[-1] == '.':
                ws.append(w[:-1])
                ws.append('.')
            else:
                ws.append(w)
        else:
            ws.append(w)
    return ' '.join(ws)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='data/train.json', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_train_path', default='data/', type=str, required=True,
                        help='训练集目标tokenized语料存放位置')
    parser.add_argument('--src_tokenized_train_path', default='data/', type=str, required=True,
                        help='训练集src tokenized语料存放位置')
    parser.add_argument('--text_mask_train_path', default='data/', type=str, required=True,
                        help='训练集构造文本mask文件存放位置')
    parser.add_argument('--entity_mask_train_path', default='data/', type=str, required=True,
                        help='训练集构造实体mask文件存放位置')
    parser.add_argument('--field_label_train_path', default='data/', type=str, required=True,
                        help='训练集构造tokenized语料存放位置')
    parser.add_argument('--label_mask_train_path', default='data/', type=str, required=True,
                        help='训练集构造tokenized语料存放位置')
    parser.add_argument('--tokenized_dev_path', default='data/', type=str, required=False,
                        help='验证集tokenized语料存放位置')
    parser.add_argument('--src_dev', default='data/', type=str, required=False,
                        help='验证集输入语料存放位置')
    parser.add_argument('--tgt_dev', default='data/', type=str, required=False,
                        help='验证集输出语料存放位置')
    parser.add_argument('--log_file', default='data/', type=str, required=False,
                        help='log文件存放位置')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--seed', default=1234, type=int, required=False, help='random seed')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--start_save_epoch', default=1, type=int, required=False, help='开始保存模型的轮数')
    parser.add_argument('--start_eval_epoch', default=1, type=int, required=False, help='开始计算验证集BLEU值的轮数')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--shuffle', action='store_true', help='是否在每个epoch打乱batch顺序')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())

    full_tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    full_tokenizer.add_tokens(['<table2text>'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    tokenized_train_path = args.tokenized_train_path
    src_tokenized_train_path = args.src_tokenized_train_path
    text_mask_train_path = args.text_mask_train_path
    entity_mask_train_path = args.entity_mask_train_path
    field_label_train_path = args.field_label_train_path
    label_mask_train_path = args.label_mask_train_path
    src_dev = args.src_dev
    tgt_dev = args.tgt_dev
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  # 不支持半精度的显卡请勿打开
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    output_dir = args.output_dir

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        # model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.transformer.output_hidden_states = True
    model.resize_token_embeddings(len(full_tokenizer))
    model.train()
    model.to(device)
    multi_gpu = False

    field_label_classify = FieldLabelClassify()
    field_label_classify.train()
    field_label_classify.to(device)
    
    print('calculating total steps')
    with open(tokenized_train_path, 'r') as f:
        gold_train_token_lines = [[int(id) for id in line.strip().split()] for line in f.readlines()]
        total_steps = len(gold_train_token_lines) * epochs / batch_size
    with open(src_tokenized_train_path, 'r') as f:
        train_src_token_lines = [[int(id) for id in line.strip().split()] for line in f.readlines()]
    with open(entity_mask_train_path, 'r') as f:
        entity_mask_train_lines = [[int(id) for id in line.strip().split()] for line in f.readlines()]
    with open(text_mask_train_path, 'r') as f:
        text_mask_train_lines = [[int(id) for id in line.strip().split()] for line in f.readlines()]
        for line1, line2 in zip(gold_train_token_lines, text_mask_train_lines):
            assert len(line1) == len(line2)
    with open(field_label_train_path, 'r') as f:
        replace_flag_train_lines = [[int(id) for id in line.strip().split()] for line in f.readlines()]
    with open(label_mask_train_path, 'r') as f:
        replace_mask_train_lines = [[int(id) for id in line.strip().split()] for line in f.readlines()]
    optimizer = transformers.AdamW([{'params': model.parameters()}, {'params': field_label_classify.parameters(), 'lr': 1.5e-4}], lr=lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    # scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    print('total steps = {}'.format(total_steps))
    with open(src_dev, 'r') as fr:
        dev_srcs = [line.strip() for line in fr.readlines()]
    log_file = open(args.log_file, 'a')

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model)
        field_label_classify = DataParallel(field_label_classify)
        multi_gpu = True
    print('starting training')
    running_loss = 0
    
    # prepare train batch data 
    train_batch_data = []
    for step in range(len(gold_train_token_lines) // batch_size):
        gold_batch = gold_train_token_lines[step * batch_size: (step + 1) * batch_size]
        src_batch = train_src_token_lines[step * batch_size: (step + 1) * batch_size]
        text_mask_batch = text_mask_train_lines[step * batch_size: (step + 1) * batch_size]
        entity_mask_batch = entity_mask_train_lines[step * batch_size: (step + 1) * batch_size]
        field_label_batch = replace_flag_train_lines[step * batch_size: (step + 1) * batch_size]
        label_mask_batch = replace_mask_train_lines[step * batch_size: (step + 1) * batch_size]
        gold_max_length = max([len(ids) for ids in gold_batch])
        src_max_length = max([len(ids) for ids in src_batch])
        # tgt_max_length = max([sum(ms) for ms in text_mask_batch])
        # gold_max_length = max([ms.index(1) + tgt_max_length for ms in text_mask_batch])
        batch_labels = []
        batch_inputs = []
        text_masks = []
        entity_masks = []
        gold_attention_masks = []
        field_labels = []
        label_masks = []
        batch_entities_len = []
        batch_entities_lst = []
        batch_entities_attr = []
        src_batch_labels = []
        tgt_batch_masks = []
        for gold_ids, src_idxs, text_mask_ids, entity_mask_ids, field_label_ids, label_mask_ids in zip(gold_batch, src_batch, text_mask_batch, entity_mask_batch, field_label_batch, label_mask_batch):
            int_ids_for_labels = [PAD_ID] * gold_max_length
            int_ids_for_inputs = [PAD_ID] * gold_max_length
            text_mask = [0] * gold_max_length
            entity_mask = [0] * gold_max_length
            field_label = [0] * gold_max_length
            gold_attention_mask = [0] * gold_max_length
            label_mask = [0] * gold_max_length
            src_batch_label = [PAD_ID] * src_max_length
            tgt_batch_mask = [0] * gold_max_length
            entities_len = []
            entities_index = []
            entities_attr = []
            entity_len = 0
            entity_id = -1
            attribute = -1
            for x_i, x in enumerate(gold_ids):
                int_ids_for_labels[x_i] = x
                int_ids_for_inputs[x_i] = x
                text_mask[x_i] = text_mask_ids[x_i] 
                entity_mask[x_i] = entity_mask_ids[x_i]
                gold_attention_mask[x_i] = 1 
                field_label[x_i] = field_label_ids[x_i]
                label_mask[x_i] = label_mask_ids[x_i]
                # tgt_batch_mask[x_i] = text_mask_ids[x_i] if sum(tgt_batch_mask) < tgt_min_length else 0
                tgt_batch_mask[x_i] = text_mask_ids[x_i]
            '''
            for x_i in range(text_mask_ids.index(1), text_mask_ids.index(1) + tgt_max_length):    
                tgt_batch_mask[x_i] = 1
            '''
            for x_i, x in enumerate(src_idxs):
                src_batch_label[x_i] = x
            for attr_id in field_label:
                if attribute != attr_id:
                    attribute = attr_id
                    entities_attr.append(attr_id)
                    if entity_len != 0:
                        entities_len.append(entity_len)
                        entities_index.extend([entity_id] * entity_len)
                    entity_len = 1
                    entity_id += 1
                else:
                    entity_len += 1
            if entity_len != 0:
                entities_len.append(entity_len)
                entities_index.extend([entity_id] * entity_len)
            batch_labels.append(int_ids_for_labels)
            batch_inputs.append(int_ids_for_inputs)
            text_masks.append(text_mask)
            entity_masks.append(entity_mask)
            gold_attention_masks.append(gold_attention_mask)
            field_labels.append(field_label)
            label_masks.append(label_mask)
            batch_entities_len.append(entities_len)
            batch_entities_lst.append(entities_index)
            batch_entities_attr.append(entities_attr)
            src_batch_labels.append(src_batch_label)
            tgt_batch_masks.append(tgt_batch_mask)
        max_entity_cnt = max([len(entities_len) for entities_len in batch_entities_len])
        batch_entities_mask = [[0] * max_entity_cnt for _ in range(batch_size)]
        new_batch_entities_len = [[1] * max_entity_cnt for _ in range(batch_size)]
        new_batch_entities_attr = [[0] * max_entity_cnt for _ in range(batch_size)]
        for b_i, entities_len in enumerate(batch_entities_len):
            for e_i, entity_len in enumerate(entities_len):
                new_batch_entities_len[b_i][e_i] = entity_len
                new_batch_entities_attr[b_i][e_i] = batch_entities_attr[b_i][e_i]
            for e_i in range(len(entities_len[:-1])):    
                batch_entities_mask[b_i][e_i] = 1
        train_batch_data.append([batch_labels, batch_inputs, text_masks, entity_masks, gold_attention_masks, field_labels, label_masks, batch_entities_lst, new_batch_entities_len, new_batch_entities_attr, batch_entities_mask, src_batch_labels, tgt_batch_masks])
    
    dev_epoch2bleu = {}
    temperature = 1.0
    start_epoch = 10
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        piece_num = 0
        if args.shuffle:
            random.shuffle(train_batch_data)
        for step in range(len(gold_train_token_lines) // batch_size):
            #  prepare data
            batch_labels = train_batch_data[step][0]
            batch_inputs = train_batch_data[step][1]
            text_masks = train_batch_data[step][2]
            entity_masks = train_batch_data[step][3]
            gold_attention_masks = train_batch_data[step][4]
            field_labels = train_batch_data[step][5]
            label_masks = train_batch_data[step][6]
            entities_lst = train_batch_data[step][7]
            entities_len = train_batch_data[step][8]
            entities_attr = train_batch_data[step][9]
            entities_mask = train_batch_data[step][10]
            src_batch_labels = train_batch_data[step][11]
            tgt_batch_masks = train_batch_data[step][12]
            batch_labels = torch.tensor(batch_labels).long().to(device)
            batch_inputs = torch.tensor(batch_inputs).long().to(device)
            text_masks = torch.tensor(text_masks).bool().to(device)
            entity_masks = torch.tensor(entity_masks).bool().to(device)
            gold_attention_masks = torch.tensor(gold_attention_masks).bool().to(device)
            field_labels = torch.tensor(field_labels).long().to(device)
            label_masks = torch.tensor(label_masks).bool().to(device)
            # src_batch_labels = torch.tensor(src_batch_labels).long().to(device)
            src_batch_labels = [torch.tensor(src_labels).long().to(device) for src_labels in src_batch_labels]
            tgt_batch_masks = torch.tensor(tgt_batch_masks).bool().to(device)
            src_size = len(entities_lst[0])
            entities_size = len(entities_len[0])
            entities_mapping = torch.zeros(batch_size, src_size, entities_size).to(device)
            for i, sent in enumerate(entities_lst):
                for j, t in enumerate(sent):
                    entities_mapping[i, j, t]
            entities_len = torch.tensor(entities_len).float().to(device)
            entities_attr = torch.tensor(entities_attr).long().to(device)
            entities_mask = torch.tensor(entities_mask).bool().to(device)

            # LM forward pass
            assert batch_inputs.size() == batch_labels.size() == gold_attention_masks.size()
            outputs = model.forward(input_ids=batch_inputs, labels=batch_labels, attention_mask=gold_attention_masks, loss_mask=text_masks)
            lm_loss, logits = outputs[:2]
            hidden_states = outputs[3][-1]
            if epoch >= start_epoch:
                flc_loss = field_label_classify(hidden_states, field_labels, label_masks)
            # flc_loss = field_label_classify(hidden_states, field_labels, label_masks, entities_attr, entities_mapping, entities_len, entities_mask)
            
            #  get loss
            if multi_gpu:
                lm_loss = lm_loss.mean()
                if epoch >= start_epoch:
                    flc_loss = flc_loss.mean()
            if gradient_accumulation > 1:
                lm_loss = lm_loss / gradient_accumulation
                if epoch >= start_epoch:
                    flc_loss = flc_loss / gradient_accumulation

            if epoch >= start_epoch:
                if multi_gpu:
                    gpt_embeddings = model.module.get_input_embeddings()
                else:
                    gpt_embeddings = model.get_input_embeddings()
                # src_batch_words = gpt_embeddings(src_batch_labels)
                src_batch_words = [gpt_embeddings(src_labels) for src_labels in src_batch_labels]
                logits = F.softmax(logits / temperature, 2)
                logits = logits[..., :-1, :].contiguous()
                seq_batch_words = torch.matmul(logits, gpt_embeddings.weight)
                '''
                tgt_batch_masks = tgt_batch_masks[..., 1:].contiguous()
                tgt_batch_masks = tgt_batch_masks.unsqueeze(-1).expand(seq_batch_words.size())
                '''
                entity_masks = entity_masks[..., 1:].contiguous()
                entity_masks = entity_masks.unsqueeze(-1).expand(seq_batch_words.size())
                # tgt_words = torch.masked_select(seq_words, tgt_batch_masks).view(seq_words.size(0), -1, seq_words.size(2)).contiguous()
                # seq_index = outputs[-1]
                # tgt_batch_gen = torch.nn.utils.rnn.pad_sequence([torch.masked_select(x[..., :-1], m[..., 1:]).long() for x, m in zip(seq_index, text_masks)], batch_first=True, padding_value=PAD_ID)
                # tgt_words = gpt_embeddings(tgt_batch_gen)
                distance = []
                for src_words, seq_words, tgt_masks in zip(src_batch_words, seq_batch_words, entity_masks):
                    src_words = src_words.unsqueeze(0)
                    seq_words = seq_words.unsqueeze(0)
                    tgt_masks = tgt_masks.unsqueeze(0)
                    tgt_words = torch.masked_select(seq_words, tgt_masks).view(seq_words.size(0), -1, seq_words.size(2)).contiguous()
                    src_words = F.normalize(src_words, p=2, dim=2, eps=1e-12)
                    tgt_words = F.normalize(tgt_words, p=2, dim=2, eps=1e-12)
                
                    # get optimal transport loss
                    cosine_cost = 1 - torch.einsum('aij,ajk->aik', src_words, tgt_words.transpose(1,2))
                    distance.append(IPOT_distance2(cosine_cost, device, beta=0.5, k_steps=3).mean())
                ot_loss = sum(distance) / float(len(distance))
                loss = lm_loss + 0.2 * flc_loss + 0.2 * ot_loss
            else:
                # loss = lm_loss + 0.2 * flc_loss
                loss = lm_loss

            #  loss backward
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            #  optimizer step
            if (step + 1) % gradient_accumulation == 0:
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            if (step + 1) % log_step == 0:
                print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {}'.format(
                    datetime.now().hour,
                    datetime.now().minute,
                    (step + 1) // gradient_accumulation,
                    piece_num,
                    epoch + 1,
                    running_loss / log_step))
                running_loss = 0
            piece_num += 1

        if epoch + 1 >= args.start_save_epoch: 
            print('saving model for epoch {}'.format(epoch + 1))
            if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
                os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))

        if epoch + 1 >= args.start_eval_epoch:   
            new_model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
            new_model.to(device)
            new_model.eval()
            total_steps = len(dev_srcs)
            output_lst = []
            with torch.no_grad():
                for step in tqdm(range(total_steps)):
                    dev_inputs = dev_srcs[step: step + 1]
                    input_ids = []
                    for dev_input in dev_inputs:
                        input_ids.append(full_tokenizer.encode(dev_input))
                    if len(input_ids[0]) > MAX_LEN:
                        input_ids[0] = input_ids[0][:MAX_LEN] + [BOS]
                        print('source input over max length')
                    src_lengths = len(input_ids[0])
                    batch_input = torch.tensor(input_ids).long().to(device)
                    # output = new_model.generate(batch_input, do_sample=False, max_length=src_lengths + 50, num_beams=5)
                    output = new_model.generate(batch_input, do_sample=False, max_length=src_lengths + 50, num_beams=5, eos_token_ids=EOS)
                    output_ids = output.tolist()[0]
                    try:
                        tgt_ids = output_ids[(output_ids.index(BOS) + 1): output_ids.index(EOS)]
                    except:
                        tgt_ids = output_ids[(output_ids.index(BOS) + 1):]
                    output_sent = rebuild_sent(full_tokenizer.decode(tgt_ids))
                    output_lst.append(output_sent)
                save_time = time.time()
                with open('gen/dev/dev_gen_%f.txt'%save_time, 'w') as fw:
                    fw.write('\n'.join(output_lst))
                cmd = "perl %s %s" % ("multi-bleu.perl", tgt_dev)
                p = subprocess.Popen(cmd.split(), stdin=open('gen/dev/dev_gen_%f.txt'%save_time), stdout=subprocess.PIPE)
                lines = p.stdout.readlines()
                if len(lines) > 0:
                    print(lines[0].decode("utf-8"))
                    dev_bleu = float(str(lines[0]).split()[2].split(",")[0])
                    dev_epoch2bleu[epoch + 1] = dev_bleu
                    # log_file.write('epoch%d bleu: %.2f\n'%(epoch + 1, dev_bleu))
                    log_file.write('epoch%d '%(epoch + 1) + lines[0].decode("utf-8"))
                    log_file.flush()
                    # wandb.log({'epoch': epoch + 1, 'bleu': dev_bleu})
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    sorted_dev_epoch2bleu = sorted(dev_epoch2bleu.items(), key=lambda item: item[1], reverse=True)
    max_bleu_epoch, max_bleu_score = sorted_dev_epoch2bleu[0]
    log_file.write('epoch%d model has highest bleu score: %.2f'%(max_bleu_epoch, max_bleu_score))
    log_file.close()

if __name__ == '__main__':
    main()