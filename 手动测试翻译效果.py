from config import *
from data_process import get_vocab, tokenize
from transformer import make_model
from torch.nn.utils.rnn import pad_sequence
import spacy
from predict import model_predict
from dataset import get_padding_mask

if __name__ == '__main__':

    en_id2vocab, en_vocab2id = get_vocab('en')
    zh_id2vocab, zh_vocab2id = get_vocab('zh')

    SRC_VOCAB_SIZE = len(en_id2vocab)
    TGT_VOCAB_SIZE = len(zh_id2vocab)

    spacy_zh = spacy.load("zh_core_web_sm")  # 中文分词
    spacy_en = spacy.load("en_core_web_sm")  # 英文分词

    model = make_model(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, d_model, n_head, d_ff, N, dropout)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

#     text = [
#     "Many people in the world have this dream: they hope that some day they will be able to come to china and visit beijing.",
#     "Although there are still many difficulties on our way ahead,we have the confidence to run well all our undertakings and to meet the preset goal.",
# ]
    text = input("原文:")
    while text != "exit":
        src_token = [en_vocab2id.get(v.lower(), unk_id) for v in tokenize(text, spacy_en)]
        batch_src = torch.LongTensor([sos_id]+src_token+[eos_id]).unsqueeze(0)
        src_x = pad_sequence(batch_src, True, pad_id)
        src_mask = get_padding_mask(src_x, pad_id)

        src_x = src_x.to(device)
        src_mask = src_mask.to(device)

        prob_sent = model_predict(model, src_x, src_mask, max_len)
        print("译文:", prob_sent[0])
        text = input("原文:")

    print("已退出手动翻译模式!")
