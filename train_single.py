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
from tqdm import tqdm
import subprocess
import wandb

from gpt2_model import GPT2LMHeadModel

# wandb.init(project="gpt-table")

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
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='Set which graphics cards to use')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='Select Model Parameters')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='Select Thesaurus')
    parser.add_argument('--raw_data_path', default='data/train.json', type=str, required=False, help='Original training corpus')
    parser.add_argument('--tokenized_train_path', default='data/', type=str, required=True,
                        help='Training set tokenized Corpus storage location')
    parser.add_argument('--text_mask_train_path', default='data/', type=str, required=True,
                        help='The training set constructs the text mask the location where the file is stored')
    parser.add_argument('--src_dev', default='data/', type=str, required=False,
                        help='Verify the set input corpus storage location')
    parser.add_argument('--tgt_dev', default='data/', type=str, required=False,
                        help='Verify the set output corpus storage location')
    parser.add_argument('--log_file', default='data/', type=str, required=False,
                        help='log The location where the file is stored')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='Training loop')
    parser.add_argument('--start_epochs', default=0, type=int, required=False, help='The number of rounds to start training')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='training batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='Learning rate')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up Number of steps')
    parser.add_argument('--seed', default=1234, type=int, required=False, help='random seed')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='How many steps to report once loss')
    parser.add_argument('--stride', default=768, type=int, required=False, help='Take the window step size of the training data during training')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='Gradient accumulation')
    parser.add_argument('--fp16', action='store_true', help='Mixed precision')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='How many parts to divide the training corpus')
    parser.add_argument('--start_save_epoch', default=1, type=int, required=False, help='The number of rounds to start saving the model')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='Model output path')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='Model training starting point')
    parser.add_argument('--shuffle', action='store_true', help='Whether in each epoch Shuffle the batch order')
    parser.add_argument('--segment', action='store_true', help='Chinese is in words')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # Set here which graphics cards the program uses
    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    full_tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    full_tokenizer.add_tokens(['<table2text>'])
    # full_tokenizer.add_tokens(['<content_select>'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    tokenized_train_path = args.tokenized_train_path
    text_mask_train_path = args.text_mask_train_path
    src_dev = args.src_dev
    tgt_dev = args.tgt_dev
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  # Do not turn on graphics cards that do not support half precision
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
    with open(text_mask_train_path, 'r') as f:
        text_mask_train_lines = [[int(id) for id in line.strip().split()] for line in f.readlines()]
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
        text_mask_batch = text_mask_train_lines[step * batch_size: (step + 1) * batch_size]
        max_length = 0
        for ids in batch:
            if len(ids) > max_length:
                max_length = len(ids)
        batch_labels = []
        batch_inputs = []
        text_masks = []
        attention_masks = []
        for ids, text_mask_ids in zip(batch, text_mask_batch):
            int_ids_for_labels = [PAD_ID] * max_length
            int_ids_for_inputs = [PAD_ID] * max_length
            text_mask = [0] * max_length
            attention_mask = [0] * max_length
            for x_i, x in enumerate(ids):
                int_ids_for_labels[x_i] = x
                int_ids_for_inputs[x_i] = x
                text_mask[x_i] = text_mask_ids[x_i] 
                attention_mask[x_i] = 1 
            batch_labels.append(int_ids_for_labels)
            batch_inputs.append(int_ids_for_inputs)
            text_masks.append(text_mask)
            attention_masks.append(attention_mask)
        train_batch_data.append([batch_labels, batch_inputs, attention_masks, text_masks])
    
    dev_epoch2bleu = {}
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
            batch_labels = torch.tensor(batch_labels).long().to(device)
            batch_inputs = torch.tensor(batch_inputs).long().to(device)
            attention_masks = torch.tensor(attention_masks).bool().to(device)
            text_masks = torch.tensor(text_masks).bool().to(device)

            #  forward pass
            outputs = model.forward(input_ids=batch_inputs, labels=batch_labels, attention_mask=attention_masks, loss_mask=text_masks)
            loss, logits = outputs[:2]

            #  get loss
            if multi_gpu:
                loss = loss.mean()
            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation

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
                        '''
                        if BOS not in output_ids:
                            input_ids = output_ids[:output_ids.index(EOS)] if EOS in output_ids else output_ids
                            input_ids.append(BOS)
                            if len(input_ids) > MAX_LEN:
                                input_ids = input_ids[:MAX_LEN] + [BOS]
                            src_lengths = len(input_ids)
                            batch_input = torch.tensor([input_ids]).long().to(device)
                            output = new_model.generate(batch_input, do_sample=False, max_length=src_lengths + 50, num_beams=5, eos_token_ids=EOS)
                            output_ids = output.tolist()[0]
                            tgt_ids = output_ids[(output_ids.index(BOS) + 1): output_ids.index(EOS)] if EOS in output_ids else output_ids[(output_ids.index(BOS) + 1):]
                        else:
                        '''
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
        # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
        # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    '''
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')
    '''
    sorted_dev_epoch2bleu = sorted(dev_epoch2bleu.items(), key=lambda item: item[1], reverse=True)
    max_bleu_epoch, max_bleu_score = sorted_dev_epoch2bleu[0]
    log_file.write('epoch%d model has highest bleu score: %.2f'%(max_bleu_epoch, max_bleu_score))
    log_file.close()
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    main()
