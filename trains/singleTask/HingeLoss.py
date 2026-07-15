import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def compute_cosine(self, x, y):
        # x = self.compute_compact_s(x)
        # y = self.compute_compact_s(y)
        x_norm = torch.sqrt(torch.sum(torch.pow(x, 2), 1)+1e-8)
        x_norm = torch.max(x_norm, 1e-8*torch.ones_like(x_norm))
        y_norm = torch.sqrt(torch.sum(torch.pow(y, 2), 1)+1e-8)
        y_norm = torch.max(y_norm, 1e-8*torch.ones_like(y_norm))
        cosine = torch.sum(x * y, 1) / (x_norm * y_norm)
        return cosine

    def forward(self, ids, feats, margin=0.1):
        B, F = feats.shape
      #用以表示样本对
        # 生成锚点特征矩阵
        s = feats.repeat(1, B).view(-1, F) # B**2 X F
        # 锚点标签矩阵
        s_ids = ids.view(B, 1).repeat(1, B) # B X B
        # 生成目标特征矩阵
        t = feats.repeat(B, 1) # B**2 X F
        # 目标标签矩阵
        t_ids = ids.view(1, B).repeat(B, 1) # B X B 

        cosine = self.compute_cosine(s, t) # B**2
        # 创建一个单位布尔矩阵，标记锚点与目标为同一索引的位置（即自身对比）
        equal_mask = torch.eye(B, dtype=torch.bool) # B X B
        # 使用~equal_mask过滤对角线元素（自身与自身对比）
        s_ids = s_ids[~equal_mask].view(B, B-1) # B X (B-1)
        t_ids = t_ids[~equal_mask].view(B, B-1) # B X (B-1)
        cosine = cosine.view(B, B)[~equal_mask].view(B, B-1) # B X (B-1)
        # 标签正样本对掩码
        sim_mask = (s_ids == t_ids) # B X (B-1)
        # 根据标签差异动态调整边距，标签差异越大，边距越大。
        #  锚点标签=3，目标标签=1 → 差异=2 → 边距=0.15×2=0.3。
        #  锚点标签=0，目标标签=0 → 差异=0 → 边距=0
        margin = 0.15 * abs(s_ids - t_ids)#[~sim_mask].view(B, B - 3)

        loss = 0
        loss_num = 0
        
        for i in range(B):
            sim_num = sum(sim_mask[i]) #当前锚点的正对数
            dif_num = B - 1 - sim_num # 当前锚点的负对数
            if not sim_num or not dif_num:
                continue
            # 提取锚点i的所有正样本对的余弦相似度（余弦相似度已经计算完成）
            sim_cos = cosine[i, sim_mask[i]].reshape(-1, 1).repeat(1, dif_num)
            # 提取锚点i的所有负样本对的余弦相似度
            dif_cos = cosine[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)
            t_margin = margin[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)

            loss_i = torch.max(torch.zeros_like(sim_cos), t_margin - sim_cos + dif_cos).mean()
            loss += loss_i
            loss_num += 1

        if loss_num == 0:
            loss_num = 1

        loss = loss / loss_num
        return loss