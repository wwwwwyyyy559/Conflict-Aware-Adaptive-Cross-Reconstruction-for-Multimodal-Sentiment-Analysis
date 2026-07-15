import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss
from fvcore.nn import FlopCountAnalysis

logger = logging.getLogger('MMSA')

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse
    
def mse_per_sample(pred, real):
    return ((real - pred) ** 2).mean(dim=(0, 2))  # -> [B]

def weighted_mean(values, weights, eps=1e-8):

    return (values * weights).sum() / (weights.sum() + eps)

class CACR():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()           
        self.cosine = nn.CosineEmbeddingLoss()
        ds_for_metric = {'cmumosi':'mosi', 'cmumosei':'mosei'}.get(args.dataset_name, args.dataset_name)
        self.metrics = MetricsTop(args.train_mode).getMetics(ds_for_metric)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()

    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: CACR model 
        params = model[0].parameters() 
        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_CACR = model[0]
        net.append(net_CACR)    
        model = net
                
        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()
              

            train_loss = 0.0 
            left_epochs = self.args.update_epochs 
            with tqdm(dataloader['train']) as td: 
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad() 
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    
            
                    output = model[0](text, audio, vision) 

                    # task loss 
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    loss_task_l_hetero = self.criterion(output['logits_l_hetero'], labels)  
                    loss_task_v_hetero = self.criterion(output['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(output['logits_a_hetero'], labels)
                    loss_task_c = self.criterion(output['logits_c'], labels)
                    
                    # total MSA loss L_msa 
                    if self.args.dataset_name in ['mosi', 'cmumosi']:
                        loss_task = 1* (2 * loss_task_all + 1*loss_task_c  + 3 * loss_task_l_hetero + 1*loss_task_v_hetero + 1*loss_task_a_hetero)
                    elif self.args.dataset_name in ['mosei', 'cmumosei']:
                        loss_task = 1* (1 * loss_task_all + 3*loss_task_c  + 3 * loss_task_l_hetero + 1*loss_task_v_hetero + 1*loss_task_a_hetero)
                    else:
                        loss_task = 1* (3 * loss_task_all + loss_task_c  + 2 * loss_task_l_hetero + 1*loss_task_v_hetero + 1*loss_task_a_hetero)

                    # conflict weighting
                    cl = output['cl'].permute(2,1,0).mean(dim=1)
                    cv = output['cv'].permute(2,1,0).mean(dim=1)
                    ca = output['ca'].permute(2,1,0).mean(dim=1)

                    cos_lv = F.cosine_similarity(cl, cv, dim=1)  # cosine_similarity
                    cos_la = F.cosine_similarity(cl, ca, dim=1)  
                    cos_va = F.cosine_similarity(cv, ca, dim=1)  

                    d_lv = (1.0 - cos_lv).unsqueeze(1)  # [B,1]
                    d_la = (1.0 - cos_la).unsqueeze(1)
                    d_va = (1.0 - cos_va).unsqueeze(1)
                    
                    tau = getattr(self.args, 'rec_tau', 0.5)            

                    S_L = (d_lv + d_la) / 2.0    # [B,1]
                    S_V = (d_lv + d_va) / 2.0
                    S_A = (d_la + d_va) / 2.0
                    w_L = torch.exp(-S_L / tau).clamp(0.0, 1.0).squeeze(-1)  # [B]
                    w_V = torch.exp(-S_V / tau).clamp(0.0, 1.0).squeeze(-1)
                    w_A = torch.exp(-S_A / tau).clamp(0.0, 1.0).squeeze(-1)
                    w_L = torch.clamp(w_L, min=0.10)
                    w_V = torch.clamp(w_V, min=0.10)
                    w_A = torch.clamp(w_A, min=0.10)

                    mix = getattr(self.args, 'rec_mix', 1.0) 
                    one = torch.ones_like(w_L)
                    w_L = (1 - mix) * one + mix * w_L
                    w_V = (1 - mix) * one + mix * w_V
                    w_A = (1 - mix) * one + mix * w_A

                    
                    recon_l_sample = mse_per_sample(output['recon_l'], output['origin_l'])  # [B]
                    recon_v_sample = mse_per_sample(output['recon_v'], output['origin_v'])
                    recon_a_sample = mse_per_sample(output['recon_a'], output['origin_a'])

                    # reconstruction loss L_r  
                    loss_recon = (
                                weighted_mean(recon_l_sample, w_L) +
                                weighted_mean(recon_v_sample, w_V) +
                                weighted_mean(recon_a_sample, w_A)   
                                )

                    # specific loss L_s  
                    loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])
                    loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                    loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla
                    
                    # ort loss L_o
                    if self.args.dataset_name in ['mosi', 'cmumosi'] or self.args.dataset_name == 'sims':
                        num = 50
                    elif self.args.dataset_name in ['mosei', 'cmumosei']:
                        num = 10

                    cosine_similarity_s_c_l = self.cosine(output['s_l'].reshape(-1, num), output['c_l'].reshape(-1, num), torch.tensor([-1]).cuda())
                    cosine_similarity_s_c_v = self.cosine(output['s_v'].reshape(-1, num), output['c_v'].reshape(-1, num), torch.tensor([-1]).cuda())
                    cosine_similarity_s_c_a = self.cosine(output['s_a'].reshape(-1, num), output['c_a'].reshape(-1, num), torch.tensor([-1]).cuda())

                    
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a #+ cosine_similarity_s_c_l_r + cosine_similarity_s_c_v_r + cosine_similarity_s_c_a_r

                    
                    loss_sim=1
                    
                    #overall loss L_CACR
                    combined_loss = loss_task + (loss_s_sr + loss_recon + ( 0*loss_sim + loss_ort ) * 0.18) * 0.1
                    
                    combined_loss.backward()

                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters())  
                                 
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()
                    

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            

            train_loss = train_loss / len(dataloader['train'])

            pred, true = torch.cat(y_pred), torch.cat(y_true)

            train_results = self.metrics(pred, true)

            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN -({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            torch.save(model[0].state_dict(), './pt/' + str(self.args.dataset_name) + '_' + str(epochs) + '.pth')
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                model_save_path = './pt/CACR' + str(self.args.dataset_name)+'.pth'
                torch.save(model[0].state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False,save_conflict_path=None):

        model.eval()
        y_pred, y_true = [], []
        
        #"""
        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
            
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    
                    output = model(text, audio, vision)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
