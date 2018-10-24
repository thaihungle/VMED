# VMED
source code for Variational Memory Encoder-Decoder

arXiv version: https://arxiv.org/abs/1807.09950 <br />
NIPS version: https://nips.cc/Conferences/2018/Schedule?showEvent=11166 <br />
repo reference: https://github.com/Mostafa-Samir/DNC-tensorflow https://github.com/Conchylicultor/DeepQA <br />

Please prepare your conversation data as follows: <br />
- A pickle file contains 3 objects: str2tok, tok2str, dialogs <br />
- str2tok and tok2str are dicitonaries mapping from word to index and index to word, respectively <br />
- index 0,1,2 should be spared for special words: _pad_, _go_, _eos_
- dialogs is a list of all conversation pairs. Each of its elements is another list of two lists, corresponding
to the input sequence and output sequence. The sequences contain index of the word in the vocab. Special words will be added later
(e.g, [[269, 230, 54, 94, 532, 23], [90, 64, 269, 125, 35, 94, 532, 9, 61, 1529]]) <br />
- To simulate conversations with multiple pairs, just concatenate all sequences until the response moment as the input sequence,
and the ground truth response as the output sequence<br />
- Please refer https://github.com/Conchylicultor/DeepQA for data preprocessing details<br />

To run the code:<br />
- train VMED example: python qa_task.py --mode=train --num_mog_mode=3 --mem_size=15 --data_dir='path_to_pickle'<br />
- test VMED example: python qa_task.py --mode=test --num_mog_mode=3 --mem_size=15 --data_dir='path_to_pickle'<br />
- VLSTM example: python qa_task.py --mode=train --num_mog_mode=1 --use_mem=False --data_dir='path_to_pickle'<br />
- CVAE example: python qa_task.py --mode=train --num_mog_mode=1 --use_mem=False --single_KL=True --data_dir='path_to_pickle'<br />

Run with word embedding: <br />
- set --use_pretrain_emb value (word2vec or glove)<br />
- hard code to modify path to embedding files<br />

Feel free to modify the hyper-parameters (some are currently hard coded), add beam search and other advanced features <br />



