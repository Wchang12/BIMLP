from ast import arg
import os
from datetime import datetime

import sys
sys.path.append('/home/b3432/Code/experiment/wangchang/MLP_like') 
import sys
from transformers import BertTokenizer
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from torch.utils.data import Dataset
import os
import scipy.io as sio
import torch
import numpy as np
import pandas as pd
#from wav_to_spectorgram import audio2spectrogram, get_3d_spec, numpy2image
from torchvision import transforms
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
#from generate_feature import wavfile_to_examples
import warnings
warnings.filterwarnings("ignore")
from transformers import BartTokenizer, BartModel
#tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
tokenizer = BertTokenizer.from_pretrained('/home/b3432/Code/experiment/wangchang/models_storage/bert-base-uncased')
class FGM():
    """ 快速梯度对抗训练
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}
 
    def attack(self, epsilon=1., emb_name='TextModel.embeddings.word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
 
    def restore(self, emb_name='TextModel.embeddings.word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
bert_dir='/home/b3432/Code/experiment/wangchang/models_storage/bert-base-uncased'
bert_cache='data/cache'

tokenizer = BertTokenizer.from_pretrained(bert_dir,
                                          use_fast=True,
                                          cache_dir=bert_cache)

    
#torch.multiprocessing.set_start_method('spawn')
class MyDataLoader(Dataset):
    def __init__(self, data):

        # self.path = args["path"]
        # self.data = pd.read_csv(path)
        self.data = data

    def __len__(self) -> int:

        return len(self.data)

    def __getitem__(self, idx):

        # fold5
        
        labels_int = self.process_labels(self.data['emotion'][idx])
        #wav_path = self.data['path_to_wavs'][idx]
        wav_path = self.data['path_to_wavs'][idx]
        index = wav_path.find('/')
        #wav_name = 'path_to_whisper_input/' + wav_path[index + 1:-4] + '.pt'
        wav_name = 'path_to_wavlm_large_input/' + wav_path[index + 1:-4] + '.pt'
        x1 = torch.load('../'+wav_name).squeeze()
        text_ids,text_mask = self.tokenize_text(str('[CLS]'+self.data['asr_text'][idx]+'[SEP]'))
        """
        #fold10
        labels_int = self.process_labels(self.data['emotion'][idx])
        wav_path = self.data['file'][idx]
        index = wav_path.find('/')
        wav_name = 'path_to_wavlm_large_input/' + wav_path[index + 1:-4] + '.pt'
        x1 = torch.load('../'+wav_name).squeeze()
        text_ids,text_mask = self.tokenize_text(str('[CLS]'+self.data['text'][idx]+'[SEP]'))
        """
        return dict(
            #x=input_spe,
            #text_ids = text_ids,
            #text_mask = text_mask,
            #x2 = input_frame,# [batch,10,1,96,64]
            input_feature = x1,
            #ids=ids,
            #mask=mask,
            #lm_labels=lm_labels,
            em_labels=labels_int,
            text_ids = text_ids,
            text_mask = text_mask
            #x2=wav_feature,
            #x3=MFCC_feature
        )
    
    def tokenize_text(self,text) -> tuple:
        encoded_inputs = tokenizer(text,
                                   max_length=128,
                                   padding='max_length',
                                   truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        # token_type=torch.LongTensor(encoded_inputs['token_type_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask
    
    def prepare_text(self,asr_text,target_text):
        source = tokenizer.encode_plus(
            asr_text,
            max_length=128,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = tokenizer.encode_plus(
            target_text,
            max_length=128,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        #target_mask = target["attention_mask"]

        y = target_ids
        y_ids = y[:].contiguous()
        lm_labels = y[:].clone()
        lm_labels[y[:] == tokenizer.pad_token_id] = -100

        ids = source_ids
        mask = source_mask

        return ids,mask,lm_labels

    def process_wav2(self, wav_path):
        img = Image.open(wav_path)
        transform = Compose([
            Resize([224, 224]),
            # CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]), ])
        """
        normMean = [0.72177774, 0.5191023, 0.6619445]
        normStd = [0.31545034, 0.41736808, 0.30021265]
        normMean = [0.7278596, 0.52566695, 0.6671147]
        normStd = [0.3129753, 0.41814464, 0.2992432]
        """
        img = transform(img)
        return img
    def process_labels(self, labels):
        cls_label_map = {"e0": 0, "e1": 1, "e2": 2, "e3": 3}
        return torch.LongTensor([cls_label_map[labels]])
def create_dataloaders(args):
    #data = pd.read_csv(args["path"])
    #size = len(data)
    #val_size = int(len(data) * args["ratio"])

    #dataset = MyDataLoader(data)
    #train_dataset, val_dataset = torch.utils.data.random_split(dataset,[size - val_size, val_size],
    #                                                           generator=torch.Generator().manual_seed(args["seed"]))
    train_dataset = MyDataLoader(pd.read_csv(args["path"]))
    val_dataset = MyDataLoader(pd.read_csv(args["val_path"]))
    if args["num_workers"] > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args["num_workers"],
                                   prefetch_factor=4)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args["batch_size"],
                                        sampler=train_sampler,
                                        drop_last=False)

    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args["val_batch_size"],
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader


from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id for lv2id in predictions]
    lv1_labels = [lv2id for lv2id in labels]

    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv1_f1_macro + lv1_f1_micro) / 2.0

    eval_results = {'Accuracy': accuracy_score(lv1_labels, lv1_preds),
                    'F1_micro': lv1_f1_micro,
                    'F1_macro': lv1_f1_macro,
                    'mean_F1': mean_f1}

    report_dict = classification_report(lv1_labels, lv1_preds,digits=6, output_dict=True)

    WA = report_dict['accuracy'] *100
    UA = report_dict['macro avg']['recall'] * 100
    macro_f1 = report_dict['macro avg']['f1-score'] * 100
    w_f1 = report_dict['weighted avg']['f1-score'] * 100

    eval_results = {'WA':WA,
                    'UA': UA,
                    #'F1_micro': lv1_f1_micro,
                    #'F1_macro': lv1_f1_macro
                    }
    return eval_results
import logging
import os
import time
import torch

def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        #loop=tqdm(val_dataloader,ncols=100)
        #for batch in loop:
        for batch in val_dataloader:
            loss,logits, pred_label_id, label ,accuracy,loss1,loss2= model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
            #loop.set_description(f'[Evluating: ]')
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results
from transformers import AdamW, get_linear_schedule_with_warmup
def build_optimizer(args, model):
    """
    model_lr = {'others': 3e-5,
                'myfc':5e-5,
                'fusion_layer1':5e-5,
                'res_mlp':args['res_mlp'],
                #'encoder':3e-5,
                #'fusion':1e-4,
                #'fc1':1e-4,
                #'fc':1e-4,
                #'features':args["features_lr"],
                #'fc':5e-5
                #'atten':args["att_lr"]
                'TextModel':3e-5,
                #'fc':1e-4,
                #'others':1e-4
               }
    """
    model_lr = {'others': 3e-5,
                'myfc':5e-5,
                'fusion_layer1':5e-5,
                'res_mlp':args['res_mlp'],
               }
    no_decay = ["bias", "LayerNorm.weight"]
    weight_decay = 0.001
    optimizer_grouped_parameters = []
    for layer_name in model_lr:
        lr = model_lr[layer_name]
        if layer_name != 'others':  # 设定了特定 lr 的 layer
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                          and layer_name in n)],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                                          and layer_name in n)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
        else:  # 其他，默认学习率
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                          and not any(
                                name in n for name in model_lr))],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                                          and not any(
                                name in n for name in model_lr))],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args["lr"], eps=args["adam_epsilon"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args["adam_epsilon"],
                                                num_training_steps=args["max_steps"])
    return optimizer, scheduler


import logging
import random
import numpy as np


def setup_device(args):
    args["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_seed(args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    #torch.cuda.manual_seed_all(args["seed"])  #并行gpu
    #torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    #torch.backends.cudnn.benchmark = True 

def setup_logging(args):
    logging.basicConfig(filename=args['log'],
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from functools import partial

import logging
import os
import time
import torch
from tqdm import tqdm

sys.path.append('/home/b3432/Code/experiment/wangchang/iemocap') 

from models.test_flip import whisper_model
import torch.nn.functional as F
from tqdm import tqdm

def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = whisper_model()
    
    fgm = FGM(model)
    optimizer, scheduler = build_optimizer(args, model)
    if args["device"] == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args["device"]))
    best_score = 0.5
    # 3. training
    step = 0
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args["max_epochs"]
    
    res_UA = []
    res_WA = []
    #loop = tqdm(enumerate(train_loader), total =len(train_loader))
    for epoch in range(args["max_epochs"]):
        #loop=tqdm(train_dataloader,ncols=100)
        #for batch in loop:
        for batch in train_dataloader:
            model.train()
            loss,logits, pred_label_id,label,accuracy,loss1,loss2 = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
            step += 1
            if step % args["print_steps"] == 0:
                print(f"Epoch {epoch} step {step}: loss {loss:.4f},loss_ce {loss1:.4f},loss_mse {loss2:.4f}  accuracy { accuracy:.4f}")
            #loop.set_description(f'Epoch [{epoch}/{args["max_epochs"]}]')
            #loop.set_postfix(loss = loss,acc = accuracy )
            if step % args["eval_steps"] == 0:
                loss, results = validate(model, val_dataloader)
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Epoch {epoch} eval_step {step}: loss {loss:.4f}, {results}")
                print(f"Epoch {epoch} eval_step {step}: loss {loss:.4f}, {results}")
                res_UA.append(results['UA'])
                res_WA.append(results['WA'])
    #logging.info(f"res_WA: {res_WA}, res_UA: {res_UA}")
    max_UA = max(res_UA)
    max_UA_index = res_UA.index(max_UA)
    max_WA = res_WA[max_UA_index]
    #logging.info(f"res_WA: {res_WA}, res_UA: {res_UA}")
    return round(max_UA,4),round(max_WA,4)
        #4. validation
        #loss, results = validate(model, val_dataloader)
        
    
        #results = {k: round(v, 4) for k, v in results.items()}
        #logging.info(f"Epoch {epoch} , {results}")
        #print(f"Epoch {epoch} , {results}")
        #"""
        #eval_results = {'WA':WA,
        #            'UA': UA,
        #            'F1_micro': lv1_f1_micro*100,
        #            'F1_macro': lv1_f1_macro*100,
        #            'w_f1': w_f1
        #"""
        # 5. save checkpoint
        #WA = results['WA']
        #UA = results['UA']             
        #if WA>best_score:
        #state_dict = model.module.state_dict() if args["device"] == 'cuda' else model.state_dict()
        #torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'WA': WA,'UA':UA},
        #           f'{args["savedmodel_path"]}/model_epoch_{epoch}_WA_{WA}_UA_{UA}.bin')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(seed):
    args=dict(
        model = 'whisper bert',
        path = '../iemocap/leave_one_sess/',
        val_path = '../iemocap/leave_one_sess/',
        ratio = 0.1,
        seed = 43,
        batch_size = 32,
        res_mlp = 1e-3,
        #res_mlp = 5e-3,
        val_batch_size = 128,
        num_workers = 0,
        features_lr = 3e-5,
        fc_lr = 3e-4,
        att_lr = 3e-4,
        lr = 3e-5,
        weight_decay = 0.001,
        adam_epsilon = 1e-6,
        max_steps = 50000,
        device = 'cuda',
        max_epochs = 10,
        print_steps = 10,
        eval_steps = 50,
        use_attack = 'False',
        use_EMA = 'False',
        savedmodel_path = 'model_checkpoint/',
        Rdrop = 'False',
        k_fold = 5,
        log ='../log/seed_5_fold_test_flip.log'
        )
    args['seed'] = seed
    #args['path'] = '../iemocap/leave_one_sess/Sess1345.csv'
    #args['val_path'] = '../iemocap/leave_one_sess/Sess2.csv'
    k_fold  = args['k_fold'] 
    if k_fold == 5:
        UA = []
        WA = []
        train_data = ['1234.csv','1235.csv','1245.csv','1345.csv','2345.csv']
        test_data = ['5.csv','4.csv','3.csv','2.csv','1.csv']
        for i in range(k_fold):
            args['path'] = '../iemocap/leave_one_sess/Sess'+train_data[i]
            args['val_path'] = '../iemocap/leave_one_sess/Sess'+test_data[i]
            setup_logging(args)
            setup_device(args)
            setup_seed(args)
            os.makedirs(args["savedmodel_path"], exist_ok=True)
            logging.info("Training/evaluation parameters: %s", args)
            train_info = 'Fold'+str(i+1)+':'+'trainsess'+train_data[i]+',testsess'+test_data[i]
            print(train_info)
            logging.info(train_info)
            best_UA , best_WA = train_and_validate(args)
            UAandWAinfo = 'Fold'+str(i+1)+": best_WA: "+str(best_WA)+": best_UA: "+str(best_UA)
            logging.info(UAandWAinfo)
            UA.append(best_UA)
            WA.append(best_WA)

        K_fold_mean_UA = sum(UA) / len(UA)
        K_fold_mean_WA = sum(WA) / len(WA)
        print('K_fold_WA: ',WA,' mean_WA: ',K_fold_mean_WA)
        print('K_fold_UA: ',UA,' mean_UA: ',K_fold_mean_UA)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_path = 'result/test_flip_5_fold_result_seeds.txt'
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'a') as file:
            file.write("Seeds: "+str(seed) + '\n')
            file.write(f'{current_time} - 5_fold_WA: {WA} mean_WA: {K_fold_mean_WA}\n')
            file.write(f'{current_time} - 5_fold_UA: {UA} mean_UA: {K_fold_mean_UA}\n\n')
        res_info1 = 'K_fold_WA: '+str(WA)+' mean_WA: '+str(round(K_fold_mean_WA,4))
        res_info2 = 'K_fold_UA: '+str(UA)+' mean_UA: '+str(round(K_fold_mean_UA,4))
        logging.info(res_info1)
        logging.info(res_info2)
if __name__ == '__main__':
    #for i in range(100):
    #seed = random.randint(1, 100000000)
    seed = 14882089
    main(seed)