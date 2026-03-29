"""Provides data for training and testing."""
import os
import PIL
import torch
import json
import torch.utils.data
import pickle

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class yangling(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        self.image_dir = self.path + 'images'
        self.split_dir = self.path + 'image_splits'
        self.caption_dir = self.path + 'captions'
        self.transform = transform

        self.data = []
        with open(os.path.join(self.caption_dir, "train.jsonl"), 'r', encoding="utf-8") as f:
            ref_captions = [json.loads(line.strip()) for line in f]  # 逐行读取并解析 JSON

        for triplets in ref_captions:
            ref_id = triplets['candidate']
            tag_id = triplets['target']
            cap = triplets['captions']
            self.data.append({
                'target': tag_id,
                'candidate': ref_id,
                'captions': cap,
            })   
        with open(os.path.join(self.path, 'data.json'), 'w') as f:
            json.dump(self.data, f, indent=2)

        if not os.path.exists(os.path.join(self.path, 'test_queries.pkl')):
            self.test_queries, self.test_targets = self.get_test_data()
            save_obj(self.test_queries, os.path.join(self.path, 'test_queries.pkl'))
            save_obj(self.test_targets, os.path.join(self.path, 'test_targets.pkl'))
        else:
            with open(os.path.join(self.path, 'data.json'), 'r') as f:
                self.data = json.load(f) 
            self.test_queries = load_obj(os.path.join(self.path, 'test_queries.pkl'))
            self.test_targets = load_obj(os.path.join(self.path, 'test_targets.pkl'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.data[idx]
        mod_str = caption['captions']
        candidate = caption['candidate']
        target = caption['target']

        out = {}
        out['source_img_data'] = self.get_img(candidate, stage=0)
        out['target_img_data'] = self.get_img(target, stage=0)
        out['mod'] = {'str': mod_str}
        return out

    def get_img(self, image_name, stage):
        img_path = os.path.join(self.image_dir, image_name + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        img = self.transform[stage](img)
        return img

    def get_test_data(self): 

        with open(os.path.join(self.split_dir, "split_val.json"), 'r') as f:
            images = json.load(f)
        with open(os.path.join(self.caption_dir, "val.jsonl"), 'r', encoding="utf-8") as f:
            ref_captions = [json.loads(line.strip()) for line in f]  # 逐行解析 JSONL

        test_queries = []
        for idx in range(len(ref_captions)):
            caption = ref_captions[idx]
            mod_str = caption['captions']
            candidate = caption['candidate']
            target = caption['target']
            out = {}
            out['source_img_id'] = images.index(candidate)
            out['source_img_data'] = self.get_img(candidate, stage=1)
            out['target_img_id'] = images.index(target)
            out['target_img_data'] = self.get_img(target, stage=1)
            out['mod'] = {'str': mod_str}

            test_queries.append(out)
        
        test_targets_id = []
        for i in test_queries:
            if i['source_img_id'] not in test_targets_id:
                test_targets_id.append(i['source_img_id'])
            if i['target_img_id'] not in test_targets_id:
                test_targets_id.append(i['target_img_id'])
        test_targets = []
        for i in test_targets_id:
            out = {}
            out['target_img_id'] = i
            out['target_img_data'] = self.get_img(images[i], stage=1)      
            test_targets.append(out)
        return test_queries, test_targets


def load_dataset(dataset_path, preprocess):
    """Loads the input datasets."""
    trainset = yangling(path = dataset_path, transform=preprocess)

    print('trainset size:', len(trainset))

    return trainset