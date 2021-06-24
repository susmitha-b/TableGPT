import transformers
import torch
import os
import json
import time
import copy
import random
import argparse
import numpy as np
from datetime import datetime
from torch.nn import DataParallel
import torch.nn.functional as F
from tqdm import tqdm
import subprocess
import wandb

from gpt2_model import GPT2LMHeadModel
from OT import IPOT_distance2

# wandb.init(project="ot")

BOS = 50257
EOS = 50256
PAD_ID = 15636
MAX_LEN = 512

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
                        help='训练集tokenized语料存放位置')
    parser.add_argument('--src_tokenized_train_path', default='data/', type=str, required=True,
                        help='训练集src tokenized语料存放位置')
    parser.add_argument('--tgt_tokenized_train_path', default='data/', type=str, required=True,
                        help='训练集tgt tokenized语料存放位置')
    parser.add_argument('--text_mask_train_path', default='data/', type=str, required=True,
                        help='训练集构造文本mask文件存放位置')
    parser.add_argument('--entity_mask_train_path', default='data/', type=str, required=True,
                        help='训练集构造实体mask文件存放位置')
    parser.add_argument('--src_dev', default='data/', type=str, required=False,
                        help='验证集输入语料存放位置')
    parser.add_argument('--tgt_dev', default='data/', type=str, required=False,
                        help='验证集输出语料存放位置')
    parser.add_argument('--log_file', default='data/', type=str, required=False,
                        help='log文件存放位置')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--start_epochs', default=0, type=int, required=False, help='开始训练的轮数')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--seed', default=1234, type=int, required=False, help='random seed')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--start_save_epoch', default=1, type=int, required=False, help='开始保存模型的轮数')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--shuffle', action='store_true', help='是否在每个epoch打乱batch顺序')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    full_tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    full_tokenizer.add_tokens(['<table2text>'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    tokenized_train_path = args.tokenized_train_path
    src_tokenized_train_path = args.src_tokenized_train_path
    tgt_tokenized_train_path = args.tgt_tokenized_train_path
    text_mask_train_path = args.text_mask_train_path
    entity_mask_train_path = args.entity_mask_train_path
    src_dev = args.src_dev
    tgt_dev = args.tgt_dev
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  # 不支持半精度的显卡请勿打开
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    output_dir = args.output_dir

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        # model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.resize_token_embeddings(len(full_tokenizer))
    model.train()
    model.to(device)
    multi_gpu = False
    
    print('calculating total steps')
    with open(tokenized_train_path, 'r') as f:
        train_token_lines = [[int(id) for id in line.strip().split()] for line in f.readlines()]
        total_steps = len(train_token_lines) * epochs / batch_size
    with open(src_tokenized_train_path, 'r') as f:
        train_src_token_lines = [[int(id) for id in line.strip().split()] for line in f.readlines()]
    with open(tgt_tokenized_train_path, 'r') as f:
        train_tgt_token_lines = [[int(id) for id in line.strip().split()] for line in f.readlines()]
    with open(text_mask_train_path, 'r') as f:
        text_mask_train_lines = [[int(id) for id in line.strip().split()] for line in f.readlines()]
    with open(entity_mask_train_path, 'r') as f:
        entity_mask_train_lines = [[int(id) for id in line.strip().split()] for line in f.readlines()]
    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                          num_training_steps=total_steps)
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
        multi_gpu = True
    print('starting training')
    running_loss = 0
    
    # prepare train batch data 
    train_batch_data = []
    for step in range(len(train_token_lines) // batch_size):
        batch = train_token_lines[step * batch_size: (step + 1) * batch_size]
        src_batch = train_src_token_lines[step * batch_size: (step + 1) * batch_size]
        tgt_batch = train_tgt_token_lines[step * batch_size: (step + 1) * batch_size]
        text_mask_batch = text_mask_train_lines[step * batch_size: (step + 1) * batch_size]
        entity_mask_batch = entity_mask_train_lines[step * batch_size: (step + 1) * batch_size]
        max_length = max([len(ids) for ids in batch])
        # src_max_length = max([len(ids) for ids in src_batch])
        # tgt_min_length = min([sum(ms) for ms in text_mask_batch])
        # tgt_max_length = max([sum(ms) for ms in text_mask_batch])
        # max_length = max([ms.index(1) + tgt_max_length for ms in text_mask_batch])
        batch_labels = []
        batch_inputs = []
        text_masks = []
        entity_masks = []
        attention_masks = []
        tgt_batch_masks = []
        for ids, text_mask_ids, entity_mask_ids in zip(batch, text_mask_batch, entity_mask_batch):
            int_ids_for_labels = [PAD_ID] * max_length
            int_ids_for_inputs = [PAD_ID] * max_length
            text_mask = [0] * max_length
            entity_mask = [0] * max_length
            attention_mask = [0] * max_length
            tgt_batch_mask = [0] * max_length
            for x_i, x in enumerate(ids):
                int_ids_for_labels[x_i] = x
                int_ids_for_inputs[x_i] = x
                text_mask[x_i] = text_mask_ids[x_i] 
                attention_mask[x_i] = 1 
                # tgt_batch_mask[x_i] = text_mask_ids[x_i] if sum(tgt_batch_mask) < tgt_min_length else 0
                tgt_batch_mask[x_i] = text_mask_ids[x_i]
                entity_mask[x_i] = entity_mask_ids[x_i]        
            batch_labels.append(int_ids_for_labels)
            batch_inputs.append(int_ids_for_inputs)
            text_masks.append(text_mask)
            entity_masks.append(entity_mask)
            attention_masks.append(attention_mask)
            tgt_batch_masks.append(tgt_batch_mask)
        train_batch_data.append([batch_labels, batch_inputs, attention_masks, text_masks, src_batch, tgt_batch, tgt_batch_masks, entity_masks])
    
    dev_epoch2bleu = {}
    temperature = 1.0
    for epoch in range(args.start_epochs, epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        piece_num = 0
        if args.shuffle:
            random.shuffle(train_batch_data)
        for step in range(len(train_token_lines) // batch_size):
            #  prepare data
            batch_labels = train_batch_data[step][0]
            batch_inputs = train_batch_data[step][1]
            attention_masks = train_batch_data[step][2]
            text_masks = train_batch_data[step][3]
            src_batch_labels = train_batch_data[step][4]
            tgt_batch_labels = train_batch_data[step][5]
            tgt_batch_masks = train_batch_data[step][6]
            entity_masks = train_batch_data[step][7]
            batch_labels = torch.tensor(batch_labels).long().to(device)
            batch_inputs = torch.tensor(batch_inputs).long().to(device)
            attention_masks = torch.tensor(attention_masks).bool().to(device)
            text_masks = torch.tensor(text_masks).bool().to(device)
            src_batch_labels = [torch.tensor(src_labels).long().to(device) for src_labels in src_batch_labels]
            # src_batch_labels = torch.tensor(src_batch_labels).long().to(device)
            tgt_batch_labels = [torch.tensor(tgt_labels).long().to(device) for tgt_labels in tgt_batch_labels]
            tgt_batch_masks = torch.tensor(tgt_batch_masks).bool().to(device)
            entity_masks = torch.tensor(entity_masks).bool().to(device)

            #  forward pass
            outputs = model.forward(input_ids=batch_inputs, labels=batch_labels, attention_mask=attention_masks, loss_mask=text_masks)
            lm_loss, logits = outputs[:2]

            #  get language model loss
            if multi_gpu:
                lm_loss = lm_loss.mean()
            if gradient_accumulation > 1:
                lm_loss = lm_loss / gradient_accumulation
            
            if epoch >= 10:
                if multi_gpu:
                    gpt_embeddings = model.module.get_input_embeddings()
                else:
                    gpt_embeddings = model.get_input_embeddings()
                # src_batch_words = gpt_embeddings(src_batch_labels)
                src_batch_words = [gpt_embeddings(src_labels) for src_labels in src_batch_labels]
                tgt_batch_words = [gpt_embeddings(tgt_labels) for tgt_labels in tgt_batch_labels]
                logits = F.softmax(logits / temperature, 2)
                logits = logits[..., :-1, :].contiguous()
                seq_batch_words = torch.matmul(logits, gpt_embeddings.weight)
                '''
                tgt_batch_masks = tgt_batch_masks[..., 1:].contiguous()
                tgt_batch_masks = tgt_batch_masks.unsqueeze(-1).expand(seq_batch_words.size())
                '''
                entity_masks = entity_masks[..., 1:].contiguous()
                entity_masks = entity_masks.unsqueeze(-1).expand(seq_batch_words.size())
                '''
                tgt_batch_words = torch.masked_select(seq_batch_words, tgt_batch_masks).view(seq_batch_words.size(0), -1, seq_batch_words.size(2)).contiguous()
                src_words = F.normalize(src_batch_words, p=2, dim=2, eps=1e-12)
                tgt_words = F.normalize(tgt_batch_words, p=2, dim=2, eps=1e-12)
                # get optimal transport loss
                cosine_cost = 1 - torch.einsum('aij,ajk->aik', src_words, tgt_words.transpose(1,2))
                distance = IPOT_distance2(cosine_cost, device)
                '''
                src_pred_distance = []
                # tgt_pred_distance = []
                for src_words, tgt_words, seq_words, tgt_masks in zip(src_batch_words, tgt_batch_words, seq_batch_words, entity_masks):
                    src_words = src_words.unsqueeze(0)
                    # tgt_words = tgt_words.unsqueeze(0)
                    seq_words = seq_words.unsqueeze(0)
                    tgt_masks = tgt_masks.unsqueeze(0)
                    pred_words = torch.masked_select(seq_words, tgt_masks).view(seq_words.size(0), -1, seq_words.size(2)).contiguous()
                    src_words = F.normalize(src_words, p=2, dim=2, eps=1e-12)
                    # tgt_words = F.normalize(tgt_words, p=2, dim=2, eps=1e-12)
                    pred_words = F.normalize(pred_words, p=2, dim=2, eps=1e-12)
                
                    # get optimal transport loss
                    src_pred_cosine_cost = 1 - torch.einsum('aij,ajk->aik', src_words, pred_words.transpose(1,2))
                    # tgt_pred_cosine_cost = 1 - torch.einsum('aij,ajk->aik', tgt_words, pred_words.transpose(1,2))
                    src_pred_distance.append(IPOT_distance2(src_pred_cosine_cost, device, t_steps=10, beta=0.5, k_steps=3).mean())
                    # tgt_pred_distance.append(IPOT_distance2(tgt_pred_cosine_cost, device, t_steps=10, beta=0.5, k_steps=3).mean())
                src_pred_ot_loss = sum(src_pred_distance) / float(len(src_pred_distance))
                # tgt_pred_ot_loss = sum(tgt_pred_distance) / float(len(tgt_pred_distance))
                # loss = lm_loss + 0.1 * src_pred_ot_loss + 0.1 * tgt_pred_ot_loss
                loss = lm_loss + 0.2 * src_pred_ot_loss
            else:
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
                    output = new_model.generate(batch_input, do_sample=False, max_length=src_lengths + 50, num_beams=5, eos_token_ids=EOS)
                    # output = new_model.generate(batch_input, do_sample=False, max_length=src_lengths + 50, num_beams=5)
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