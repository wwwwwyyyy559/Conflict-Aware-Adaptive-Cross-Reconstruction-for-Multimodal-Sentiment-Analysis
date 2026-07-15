"""
here is the mian backbone for CACR
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 添加镜像源
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder   
from ...singleTask.model.TransformerLength import LengthAdaptiveEncoder
from ...singleTask.model.Transformer import Transformer

class CACR(nn.Module):
    def __init__(self, args):
        super(CACR, self).__init__()
        self.args = args
        if args.dataset_name in ['cmumosi', 'cmumosei']:
            self.use_bert = False
        if args.dataset_name in ['mosi', 'mosei']:
            #if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                            pretrained=args.pretrained)
            self.use_bert = True
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name in ['mosi', 'cmumosi']:
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name in ['mosei', 'cmumosei']:
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500

        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims

        self.num_heads = nheads
        self.r_num_heads = 2  

        self.layers = args.nlevels 

        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

        combined_dim_low = self.d_a    
        combined_dim_high = self.d_a

        combined_dim = (self.d_l + self.d_a + self.d_v ) + self.d_l * 3  
        
        output_dim = 1
        
        # 1. Temporal convolutional layers for initial feature 
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2. Modality-specific encoder 
        self.encoder_s_l = self.get_network(self_type='l', layers = self.layers)       
        self.encoder_s_v = self.get_network(self_type='v', layers = self.layers)
        self.encoder_s_a = self.get_network(self_type='a', layers = self.layers)

        #   Modality-shared encoder 
        self.encoder_c = self.get_network(self_type='l', layers = self.layers)        
        
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)
        
        self.recon_gate_v = nn.Sequential(
            nn.Linear(self.d_v*2, self.d_v),
            nn.Sigmoid())
        self.recon_gate_a = nn.Sequential(
            nn.Linear(self.d_a*2, self.d_a),
            nn.Sigmoid())

        self.ra = self.get_network(self_type='ra')
        self.rv = self.get_network(self_type='rv')
        self.rl = self.get_network(self_type='rl')

        self.decoder_lv = nn.Conv1d(self.d_l + self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_vl = nn.Conv1d(self.d_v + self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        self.decoder_la = nn.Conv1d(self.d_l + self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.decoder_al = nn.Conv1d(self.d_a + self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        
        self.decoder_va = nn.Conv1d(self.d_v + self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.decoder_av = nn.Conv1d(self.d_a + self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        self.linear = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)

        # Modality-specific encoder 
        self.recon_encoder_s_l = self.get_network(self_type='l', layers = self.layers)       
        self.recon_encoder_s_v = self.get_network(self_type='v', layers = self.layers)
        self.recon_encoder_s_a = self.get_network(self_type='a', layers = self.layers)
        # Modality-shared encoder 
        self.recon_encoder_c = self.get_network(self_type='l', layers = self.layers)


        #FSR
        self.text_guided_vis = self.get_network(self_type='lv', layers = self.layers)
        self.text_guided_audio = self.get_network(self_type='la', layers = self.layers)


        self.detect_v = nn.Sequential(
            nn.Conv1d(self.d_v, self.d_v * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=3, padding=1),
            nn.InstanceNorm1d(self.d_v)  
        )
        self.detect_a = nn.Sequential(
            nn.Conv1d(self.d_a, self.d_a * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=3, padding=1),
            nn.InstanceNorm1d(self.d_a)  
        )

        self.text_reconstructor = nn.Sequential(

            nn.Conv1d(self.d_l + self.d_v + self.d_a, self.d_l*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=3, padding=1),
            nn.InstanceNorm1d(self.d_l) 

        )

        self.alignment_gate = nn.Sequential(
            nn.Linear(self.d_l * 2 + self.d_v + self.d_a, self.d_l),
            nn.Sigmoid()
        )


        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)


        # 4 Multimodal Crossmodal Attentions 
        self.trans_l_with_a = self.get_network(self_type='la', layers = self.layers)  
        self.trans_l_with_v = self.get_network(self_type='lv', layers = self.layers) 
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=self.layers)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)


        # 5. fc layers for shared features 
        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        
        # 6. fc layers for specific features 
        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)
        
        # 7. project for fusion
        self.projector_l = nn.Linear(self.d_l, self.d_l)
        self.projector_v = nn.Linear(self.d_v, self.d_v)
        self.projector_a = nn.Linear(self.d_a, self.d_a)
        self.projector_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
        
        # 8. final project
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)


    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, num_heads, attn_dropout = self.d_l,self.num_heads, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, num_heads, attn_dropout = self.d_a,self.num_heads, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, num_heads, attn_dropout = self.d_v,self.num_heads, self.attn_dropout_v        
        elif self_type == 'l_mem':
            embed_dim, num_heads, attn_dropout = self.d_l,self.num_heads, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, num_heads, attn_dropout = self.d_a, self.num_heads,self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, num_heads, attn_dropout = self.d_v, self.num_heads,self.attn_dropout
        elif self_type in ['reconl', 'rl']:
            embed_dim, num_heads, attn_dropout = self.d_l, self.r_num_heads  , self.attn_dropout
        elif self_type in ['rv','rlv']:
            embed_dim, num_heads, attn_dropout = self.d_v, self.r_num_heads , self.attn_dropout_v
        elif self_type in ['ra','rla']:
            embed_dim, num_heads, attn_dropout = self.d_a, self.r_num_heads , self.attn_dropout_a
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    def lengthtrans(self, self_type='l',layers=-1,input_len=35,output_len = 46):
        if self_type=='l':
            embed_dim, attn_dropout, input_len, output_len = self.d_l, self.attn_dropout,input_len, output_len
        elif self_type=='v':
            embed_dim, attn_dropout, input_len, output_len = self.d_v, self.attn_dropout_v,input_len, output_len
        else:
            embed_dim, attn_dropout, input_len, output_len = self.d_a, self.attn_dropout_a,input_len, output_len
        return LengthAdaptiveEncoder(embed_dim=embed_dim,input_len=input_len,
                                        output_len=output_len,num_heads=self.num_heads,
                                        layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video):
        #extraction
        if self.use_bert:
            text = self.text_model(text)
        
        if self.args.need_data_aligned:
            x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training) 
            x_a = audio.transpose(1, 2)
            x_v = video.transpose(1, 2)  
        else: #
            x_l = F.dropout(text, p=self.text_dropout, training=self.training) 
            x_a = audio  
            x_v = video  

        if self.orig_d_l == self.d_l :
            proj_x_l = x_l
            proj_x_l = proj_x_l.permute(2, 0, 1)

        else:
                proj_x_l = self.proj_l(x_l)
                proj_x_l = proj_x_l .permute(2, 0, 1)

        if self.orig_d_a == self.d_a :
            proj_x_a = x_a
            proj_x_a = proj_x_a.permute(2, 0, 1)
        else:

            proj_x_a = self.proj_a(x_a)
            proj_x_a = proj_x_a .permute(2, 0, 1)
        if self.orig_d_v == self.d_v :
            proj_x_v = x_v
            proj_x_v = proj_x_v.permute(2, 0, 1)
      
        else:
            proj_x_v = self.proj_v(x_v)
            proj_x_v = proj_x_v .permute(2, 0, 1)

        #disentanglement 
        s_l = self.encoder_s_l(proj_x_l)   
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)
        

        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)
        

        s_l = s_l.permute(1, 2, 0) 
        s_v = s_v.permute(1, 2, 0)  
        s_a = s_a.permute(1, 2, 0)

        c_l = c_l.permute(1, 2, 0)
        c_v = c_v.permute(1, 2, 0)
        c_a = c_a.permute(1, 2, 0)
        c_list = [c_l, c_v, c_a]

        # cross-reconstruction
        if self.args.dataset_name in ['mosi', 'cmumosi'] or self.args.dataset_name in ['mosei', 'cmumosei']or self.args.dataset_name =='sims':

            s_l = self.rl(s_l.permute(0, 2, 1)).permute(0, 2, 1)
            s_a = self.ra(s_a.permute(0, 2, 1)).permute(0, 2, 1)
            s_v = self.rv(s_v.permute(0, 2, 1)).permute(0, 2, 1)
            c_list[0] = self.rl(c_list[0].permute(0, 2, 1)).permute(0, 2, 1)
            c_list[1] = self.rv(c_list[1].permute(0, 2, 1)).permute(0, 2, 1)
            c_list[2] = self.ra(c_list[2].permute(0, 2, 1)).permute(0, 2, 1)

            recon_vl = self.decoder_vl(torch.cat([s_l, c_list[1]], dim=1))
            recon_al = self.decoder_al(torch.cat([s_l, c_list[2]], dim=1))

            recon_gate_al = self.recon_gate_a(torch.cat([s_l, c_list[2]], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
            recon_gate_vl = self.recon_gate_v(torch.cat([s_l, c_list[1]], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
            recon_l = self.linear(torch.cat([recon_gate_al * recon_al, recon_gate_vl * recon_vl], dim=1))


            recon_la = self.decoder_la(torch.cat([c_list[0], s_a], dim=1))
            recon_va = self.decoder_va(torch.cat([c_list[1], s_a], dim=1))

            recon_lv = self.decoder_lv(torch.cat([c_list[0], s_v], dim=1))
            recon_av = self.decoder_av(torch.cat([s_v, c_list[2]], dim=1))

            recon_v = self.linear(torch.cat([recon_lv, recon_av], dim=1))
            recon_a = self.linear(torch.cat([recon_la, recon_va], dim=1))

        
        recon_l = recon_l.permute(2, 0, 1)  
        recon_v = recon_v.permute(2, 0, 1)   
        recon_a = recon_a.permute(2, 0, 1)


        c_l_r = self.encoder_c(recon_l).permute(1, 2, 0)
        c_v_r = self.encoder_c(recon_v).permute(1, 2, 0)  
        c_a_r = self.encoder_c(recon_a).permute(1, 2, 0)

            

        s_l_r = self.recon_encoder_s_l(recon_l).permute(1, 2, 0)                                                                                             
        s_v_r = self.recon_encoder_s_v(recon_v).permute(1, 2, 0)
        s_a_r = self.recon_encoder_s_a(recon_a).permute(1, 2, 0)

       
        s_l = s_l.permute(2, 0, 1)  
        s_v = s_v.permute(2, 0, 1)   
        s_a = s_a.permute(2, 0, 1)#[48,16,50]

        c_l = c_l_r.permute(2, 0, 1)
        c_v = c_v_r.permute(2, 0, 1)
        c_a = c_a_r.permute(2, 0, 1)

        cl = c_l.permute(2, 0, 1) 
        cv = c_v.permute(2, 0, 1)
        ca = c_a.permute(2, 0, 1)

        

       #enhancement 
        c_l_att = self.self_attentions_c_l(c_l)  
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]

        c_v_att = self.self_attentions_c_v(c_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]

        c_a_att = self.self_attentions_c_a(c_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]

        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)   

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion   
                 
        logits_c = self.out_layer_c(c_proj)     
        
        # FSR

        if self.args.dataset_name in ['mosei', 'cmumosei'] or self.args.dataset_name in ['mosi', 'cmumosi'] :
            s_v_mic = self.detect_v(s_v.permute(1, 2, 0)).permute(2, 0, 1)
            s_a_mic = self.detect_a(s_a.permute(1, 2, 0)).permute(2, 0, 1)

            if self.args.dataset_name in ['mosi', 'cmumosi'] or self.args.dataset_name =='sims' or self.args.dataset_name in ['mosei', 'cmumosei']: 
                s_v_mic = self.text_guided_vis(s_l, s_v_mic, s_v_mic)
                s_a_mic = self.text_guided_audio(s_l, s_a_mic, s_a_mic)
            if self.args.dataset_name in ['mosei', 'cmumosei'] or self.args.dataset_name in ['mosi', 'cmumosi'] :
                x=torch.cat([s_v_mic, s_a_mic, s_l], dim=2)
                s_l_recon = self.text_reconstructor(x.permute(1, 2, 0)).permute(2, 0, 1)
                
                gate = self.alignment_gate(torch.cat([s_l, s_l_recon, s_v_mic, s_a_mic], dim=2))
                enhanced_text = s_l + gate * s_l_recon
                s_l = enhanced_text
            
            s_a = s_a_mic
            s_v = s_v_mic
        
        # L --> L         
        h_ls = s_l                  
        h_ls = self.trans_l_mem(h_ls)  
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]  

        # A --> L 
        h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a) 
        h_as = h_l_with_as
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]  

        # V --> L  
        h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v)  
        h_vs = h_l_with_vs
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]  



        hs_proj_l_high = self.proj2_l_high(
            F.dropout(F.relu(self.proj1_l_high(last_h_l), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_high += last_h_l 
        logits_l_high = self.out_layer_l_high(hs_proj_l_high) 

        hs_proj_v_high = self.proj2_v_high(
            F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_high += last_h_v
        logits_v_high = self.out_layer_v_high(hs_proj_v_high)

        hs_proj_a_high = self.proj2_a_high(
            F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout,
                      training=self.training))
        hs_proj_a_high += last_h_a
        logits_a_high = self.out_layer_a_high(hs_proj_a_high)
        
        #fusion 

        last_h_l = torch.sigmoid(self.projector_l(hs_proj_l_high))   
        last_h_v = torch.sigmoid(self.projector_v(hs_proj_v_high))
        last_h_a = torch.sigmoid(self.projector_a(hs_proj_a_high))
        c_fusion = torch.sigmoid(self.projector_c(c_fusion))


        last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)   

        #prediction
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs                          

        output = self.out_layer(last_hs_proj)


        res = {
            'origin_l': proj_x_l, 
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': s_l,      
            's_v': s_v,
            's_a': s_a,
            'c_l': c_l,   
            'c_v': c_v,
            'c_a': c_a,
            's_l_r': s_l_r,  
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'cl': cl,   
            'cv': cv,
            'ca': ca,

            'recon_l': recon_l,  
            'recon_v': recon_v,
            'recon_a': recon_a,

        
            'logits_l_hetero': logits_l_high,    
            'logits_v_hetero': logits_v_high, 
            'logits_a_hetero': logits_a_high,
            'logits_c': logits_c,
            'output_logit': output  
        }
        return res
