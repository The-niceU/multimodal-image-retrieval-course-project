import torch
import torch.nn as nn
import torch.nn.functional as F

class vanilla(nn.Module):
    def __init__(self, clip_model, tokenizer, tau, device):
        super().__init__()
        self.clip = clip_model
        self.tokenizer = tokenizer
        self.tau = tau
        self.device = device

    def extract_img_fea(self, x):
        image_features = self.clip.encode_image(x)
        return image_features

    def extract_text_fea(self, txt):
        txt = self.tokenizer(txt).to(self.device)
        text_features = self.clip.encode_text(txt)
        return text_features

    def extract_query(self, textual_query, visual_query):
        textual_query = F.normalize(self.extract_text_fea(textual_query), p=2, dim=-1)
        visual_query = F.normalize(self.extract_img_fea(visual_query), p=2, dim=-1)
        weight_textual_query = 1.0
        weight_visual_query = 1.0
        query = (weight_textual_query*textual_query + weight_visual_query*visual_query)/(weight_textual_query + weight_visual_query)
        return F.normalize(query, p=2, dim=-1)
    
    def extract_target(self, target_img):
        target_img_fea = self.extract_img_fea(target_img)
        return F.normalize(target_img_fea, p=2, dim=-1)

    def compute_loss(self, visual_query, textual_query, target_img):
        query_feature = self.extract_query(textual_query, visual_query) 
        target_feature = self.extract_target(target_img)  
        scores = torch.mm(query_feature, target_feature.t())
        i2t = - (scores / self.tau).log_softmax(1).diag().mean()
        t2i = - (scores.t() / self.tau).log_softmax(1).diag().mean()

        return i2t + t2i


def create_model_and_optimizer(clip_model, tokenizer, clip_lr, weight_decay, tau, device):
    model = vanilla(clip_model, tokenizer, tau, device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=clip_lr, weight_decay = weight_decay, eps=1e-6)
    return model, optimizer