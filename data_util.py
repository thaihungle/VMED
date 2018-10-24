import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pickle
import sys
import os
import nltk
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
cachedStopWords = stopwords.words("english")
EN_WHITELIST = '.?!0123456789abcdefghijklmnopqrstuvwxyz '  # space is included in whitelist
EN_BLACKLIST = '"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~\''

def load_word2vec(emb_dim=50, dirname = './data/GoogleNews-vectors-negative300.bin',
              str2tok_dir='./data/aclImdb/dump/str2token.pkl', init_zero=False):
    if isinstance(str2tok_dir, str):
        str2tok = pickle.load(open(str2tok_dir, 'rb'))
    else:
        str2tok=str2tok_dir
    with open(dirname, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        print('vocab size {} vector size {}'.format(vocab_size, vector_size))
        binary_len = np.dtype('float32').itemsize * vector_size
        if init_zero:
            initW = np.zeros((len(str2tok), vector_size))
        else:
            initW = np.random.uniform(-0.25, 0.25, (len(str2tok), vector_size))
        embeddings_format = os.path.splitext(dirname)[1][1:]
        not_in=0
        for _ in tqdm(range(vocab_size)):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word).decode('utf-8')
                    break
                if ch != b'\n':
                    word.append(ch)
            if word in str2tok:
                if embeddings_format == 'bin':
                    vector = np.fromstring(f.read(binary_len), dtype='float32')
                elif embeddings_format == 'vec':
                    vector = np.fromstring(f.readline(), sep=' ', dtype='float32')
                else:
                    raise Exception("Unkown format for embeddings: %s " % embeddings_format)
                initW[str2tok[word]] = vector
            else:
                not_in+=1
                if embeddings_format == 'bin':
                    f.read(binary_len)
                elif embeddings_format == 'vec':
                    f.readline()
                else:
                    raise Exception("Unkown format for embeddings: %s " % embeddings_format)

    # PCA Decomposition to reduce word2vec dimensionality
    print('pca...')
    if emb_dim < vector_size:
        U, s, Vt = np.linalg.svd(initW, full_matrices=False)
        S = np.zeros((vector_size, vector_size))
        S[:vector_size, :vector_size] = np.diag(s)
        initW = np.dot(U[:, :emb_dim], S[:emb_dim, :emb_dim])
    print('done with not in {}'.format(len(str2tok)-(vocab_size-not_in)))
    return initW

def loadGloVe(emb_dim=50, dirname = './data/glove.6B',
              str2tok_dir='./data/aclImdb/dump/str2token.pkl'):
    if isinstance(str2tok_dir, str):
        str2tok = pickle.load(open(str2tok_dir, 'rb'))
    else:
        str2tok=str2tok_dir
    filename=dirname+'/{}.{}d.txt'.format(os.path.split(dirname)[-1],emb_dim)
    file = open(filename,'r', encoding='latin')
    dic_emb={}
    print('use globe {}'.format(filename))
    for ii,line in enumerate(file.readlines()):
        if ii%1000==0:
            llprint('\rload {}'.format(ii))
        row = line.strip().split(' ')
        lst=row[1:]
        dic_emb[row[0]]=[float(i) for i in lst]
        # print(dic_emb[row[0]])
        # print(len(dic_emb[row[0]]))
        # print(row[0])
        # raise False
    print('Loaded GloVe! {} vs {}'.format(len(dic_emb),len(str2tok)))
    file.close()
    mat_encoder_emb=np.zeros((len(str2tok),emb_dim))
    not_in=0
    for k,v in str2tok.items():
        if k in dic_emb:
            mat_encoder_emb[v]=dic_emb[k]
        else:
            not_in+=1
            mat_encoder_emb[v]=np.random.uniform(0,1,emb_dim)
    print('Loaded emb! with not in {}'.format(not_in))
    print(mat_encoder_emb.shape)
    return mat_encoder_emb

def load_lines_from_file(fpath, str2tok):
    all_sens=[]
    with open(fpath) as f:
        for line in f:
            sen=[]
            line = ''.join([ch if ch in EN_WHITELIST else ' ' for ch in line.lower()])
            tokens = nltk.word_tokenize(line)
            for tok in tokens:
                if tok in str2tok:
                    sen.append(str2tok[tok])
                else:
                    sen.append(str2tok['<unknown>'])
            all_sens.append([sen,[1]*10])
    # print(all_sens)
    # raise False
    return all_sens

def bleu_score(input_batch, target_batch, predict_batch, token2str, print_prob=0.9995):
    s=[]
    for b in range(target_batch.shape[0]):
        trim_target = []
        trim_predict = []
        str_target = []
        str_predict = []
        str_input=[]
        for t in target_batch[b]:
            if t > 2:
                trim_target.append(t)
                str_target.append(token2str[t])
        for t in predict_batch[b]:
            if t > 2:
                trim_predict.append(t)
                str_predict.append(token2str[t])
        if np.random.rand()>print_prob:
            for t in input_batch[b]:
                if t > 2:
                    str_input.append(token2str[t])
            print('{}-->{} vs {}'.format(str_input, str_target, str_predict))
        try:
            BLEUscore = sentence_bleu([trim_target], trim_predict,smoothing_function=SmoothingFunction().method7)
        except:
            BLEUscore = 0
        s.append(BLEUscore)
    return np.mean(s)

def bleu_score4(input_batch, target_batch, predict_batch, token2str, print_prob=0.9995):
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    for b in range(target_batch.shape[0]):
        trim_target = []
        trim_predict = []
        str_target = []
        str_predict = []
        str_input=[]
        for t in target_batch[b]:
            if t >2:
                trim_target.append(t)
                str_target.append(token2str[t])
        for t in predict_batch[b]:
            if t >2:
                trim_predict.append(t)
                str_predict.append(token2str[t])
        if np.random.rand()>print_prob:
            for t in input_batch[b]:
                if t > 2:
                    str_input.append(token2str[t])
            print('{}-->{} vs {}'.format(str_input, str_target, str_predict))
        try:
            BLEUscore1 = sentence_bleu([trim_target], trim_predict, weights=(1, 0, 0, 0),smoothing_function=SmoothingFunction().method7)

        except:
            BLEUscore1 = 0
        try:
            BLEUscore2 = sentence_bleu([trim_target], trim_predict, weights=(0.5, 0.5, 0, 0),smoothing_function=SmoothingFunction().method7)

        except:
            BLEUscore2 = 0
        try:
            BLEUscore3 = sentence_bleu([trim_target], trim_predict, weights=(0.33, 0.33, 0.33, 0),smoothing_function=SmoothingFunction().method7)

        except:
            BLEUscore3 = 0
        try:
            BLEUscore4 = sentence_bleu([trim_target], trim_predict, weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=SmoothingFunction().method7)
        except:
            BLEUscore4 = 0
        s1.append(BLEUscore1)
        s2.append(BLEUscore2)
        s3.append(BLEUscore3)
        s4.append(BLEUscore4)
    return [np.mean(s1),np.mean(s2),np.mean(s3),np.mean(s4)]


def bow_score(input_batch, target_batch, predict_batch, token2str, mat=None):
    s1=[]
    for b in range(target_batch.shape[0]):
        trim_target = []
        trim_predict = []
        str_target = []
        str_predict = []
        str_input=[]
        if mat is None:
            oh1=np.zeros(len(token2str))
            oh2 = np.zeros(len(token2str))
        else:
            oh1 = np.zeros(mat.shape[1])
            oh2 = np.zeros(mat.shape[1])
        for t in target_batch[b]:
            if t >2 and token2str[t]!='.':
                trim_target.append(t)
                if token2str[t] not in cachedStopWords:
                    if mat is None:
                        oh1+=onehot(t, len(token2str))
                    else:
                        oh1+=mat[t]
                str_target.append(token2str[t])
        for t in predict_batch[b]:
            if t >2 and token2str[t]!='.':
                trim_predict.append(t)
                if token2str[t] not in cachedStopWords:
                    if mat is None:
                        oh2+=onehot(t, len(token2str))
                    else:
                        oh2+=mat[t]
                str_predict.append(token2str[t])

        s1.append(cosine_similarity(np.reshape(oh1,[1,-1]),np.reshape(oh2,[1,-1])))

    return np.mean(s1)

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def load(path):
    return pickle.load(open(path, 'rb'))

def onehot(index, size):
    # print('-----')
    # print(index)
    vec = np.zeros(size, dtype=np.float32)
    vec[int(index)] = 1.0
    return vec

def prepare_sample_batch(diag_list,word_space_size_input,word_space_size_output, bs):
    if isinstance(bs, int):
        indexs = np.random.choice(len(diag_list),bs,replace=True)
    else:
        #print('from {} to {}'.format(bs[0],bs[1]))
        indexs=list(range(bs[0],bs[1]))
        bs = bs[1]-bs[0]
    minlen=0
    moutlne=0

    for index in indexs:
        index2=index
        if index<0:
            index2 = (index+bs) % len(diag_list)
        minlen=max(len(diag_list[index2][0]),minlen)
        moutlne = max(len(diag_list[index2][1]), moutlne)

    input_vecs=[]
    output_vecs=[]
    seq_len = minlen + 1
    decoder_length = moutlne+2
    out_list=[]
    in_list=[]
    masks=[]
    for index in indexs:
        index2 = index
        if index < 0:
            index2 = (index+bs) % len(diag_list)
        # print('\n{}'.format(index))
        ins=diag_list[index2][0]
        in_list.append(ins)
        ose=[1]+diag_list[index2][1]+[2]
        out_list.append(ose)
        input_vec = np.zeros(seq_len)
        output_vec = np.zeros(decoder_length)
        mask=np.zeros(decoder_length, dtype=np.bool)
        for iii, token in enumerate(ins):
            input_vec[minlen-len(ins)+iii] = token
            # if lm_train:
            #     output_vec[minlen - len(ins) + iii+1] = token
            #     mask[minlen - len(ins) + iii+1] = True
        input_vec[minlen] = 2




        for iii, token in enumerate(ose):
            output_vec[iii] = token
            mask[iii]=True

        # print(ins)
        # print(ose)
        # print(input_vec)
        # print(output_vec)
        # print('====')

        output_vec = np.array([onehot(code, word_space_size_output) for code in output_vec])

        input_vec = [onehot(code, word_space_size_input) for code in input_vec]
        input_vecs.append(input_vec)
        output_vecs.append(output_vec)
        masks.append(mask)

    # raise False
    return np.asarray(input_vecs), np.asarray(output_vecs), seq_len, decoder_length, np.asarray(masks), out_list, in_list

