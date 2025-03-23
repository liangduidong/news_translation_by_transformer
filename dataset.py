from torch.nn.utils.rnn import pad_sequence
from data_process import get_vocab, get_spacy_texts, tokenize
from config import *
import spacy
import torch.utils.data as data

# 参数 size 为句子长度
def get_subsequent_mask(size):
    # 1为batch维度
    mask_shape = (1, size, size)
    return 1 - torch.tril(torch.ones(mask_shape)).byte()


def get_padding_mask(x, padding_idx):
    # 扩展Q维度
    return (x == padding_idx).unsqueeze(1).byte()

# 训练数据加载器
class Dataset(data.Dataset):
    def __init__(self, src_data, tgt_data, src_spacy, tgt_spacy, src_vocal, tgt_vocab):
        super().__init__()
        self.src_data = src_data  # 原文
        self.tgt_data = tgt_data
        self.src_spacy = src_spacy
        self.tgt_spacy = tgt_spacy
        self.src_vocal = src_vocal
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, item):
        src_text = self.src_data[item]
        tgt_text = self.tgt_data[item]
        # 先分词
        src_text = tokenize(src_text, self.src_spacy)
        tgt_text = tokenize(tgt_text, self.tgt_spacy)
        # id化
        src_ids = [self.src_vocal.get(v.lower(), unk_id) for v in src_text]
        tgt_ids = [self.tgt_vocab.get(v, unk_id) for v in tgt_text]
        return src_ids, tgt_ids, self.tgt_data[item]

    def collate_fn(self, batch):
        batch_src, batch_tgt, tgt_text = zip(*batch)
        # 获取原输入和掩码
        src_x = pad_sequence([torch.LongTensor([sos_id] + src + [eos_id]) for src in batch_src], True, pad_id)
        src_mask = get_padding_mask(src_x, pad_id)

        # 获取目的输入和其掩码
        tgt_f = pad_sequence([torch.LongTensor([sos_id] + tgt + [eos_id]) for tgt in batch_tgt], True, pad_id)
        tgt_x = tgt_f[:, :-1]  # 解码器输入
        tgt_y = tgt_f[:, 1:]   # 目标
        tgt_pad_mask = get_padding_mask(tgt_x, pad_id)
        tgt_sub_mask = get_subsequent_mask(tgt_x.size(1))
        tgt_mask = tgt_pad_mask | tgt_sub_mask

        return src_x, src_mask, tgt_x, tgt_mask, tgt_y, tgt_text

def get_dataset(path=train_path):
    zh_texts, en_texts = get_spacy_texts(path)

    _, zh_vocab2id = get_vocab('zh')
    _, en_vocab2id = get_vocab('en')

    spacy_zh = spacy.load("zh_core_web_sm")  # 中文分词
    spacy_en = spacy.load("en_core_web_sm")  # 英文分词

    dataset = Dataset(en_texts, zh_texts, spacy_en, spacy_zh, en_vocab2id, zh_vocab2id)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    return loader


