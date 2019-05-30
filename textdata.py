# Copyright 2015 Conchylicultor. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Loads the dialogue corpus, builds the vocabulary
"""

EN_WHITELIST = '.?!0123456789abcdefghijklmnopqrstuvwxyz '  # space is included in whitelist
EN_BLACKLIST = '"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~\''


import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import re
import random
import string
import collections
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

from corpus.cornelldata import CornellData
from corpus.opensubsdata import OpensubsData
from corpus.scotusdata import ScotusData
from corpus.ubuntudata import UbuntuData
from corpus.lightweightdata import LightweightData

raw_text_path='qa_samples'

class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []


class TextData:
    """Dataset class
    Warning: No vocabulary limit
    """

    availableCorpus = collections.OrderedDict([  # OrderedDict because the first element is the default choice
        ('cornell', CornellData),
        ('opensubs', OpensubsData),
        ('scotus', ScotusData),
        ('ubuntu', UbuntuData),
        ('lightweight', LightweightData),
    ])

    @staticmethod
    def corpusChoices():
        """Return the dataset availables
        Return:
            list<string>: the supported corpus
        """
        return list(TextData.availableCorpus.keys())

    def __init__(self, args):
        """Load all conversations
        Args:
            args: parameters of the model
        """
        # Model parameters
        self.args = args

        # Path variables
        self.corpusDir = os.path.join(self.args.rootDir, self.args.corpus)
        basePath = self._constructBasePath()
        self.fullSamplesPath = basePath + '.pkl'  # Full sentences length/vocab
        self.filteredSamplesPath = basePath + '-length{}-filter{}-vocabSize{}.pkl'.format(
            self.args.maxLength,
            self.args.filterVocab,
            self.args.vocabularySize,
        )  # Sentences/vocab filtered for this model

        self.padToken = -1  # Padding
        self.goToken = -1  # Start of sequence
        self.eosToken = -1  # End of sequence
        self.unknownToken = -1  # Word dropped from vocabulary

        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]

        self.word2id = {}
        self.id2word = {}  # For a rapid conversion (Warning: If replace dict by list, modify the filtering to avoid linear complexity with del)
        self.idCount = {}  # Useful to filters the words (TODO: Could replace dict by list or use collections.Counter)

        self.loadCorpus()

        # Plot some stats:
        self._printStats()

        if self.args.playDataset:
            self.playDataset()

    def _printStats(self):
        print('Loaded {}: {} words, {} QA'.format(self.args.corpus, len(self.word2id), len(self.trainingSamples)))

    def _constructBasePath(self):
        """Return the name of the base prefix of the current dataset
        """
        path = os.path.join(self.args.rootDir, raw_text_path + os.sep)
        if not os.path.isdir(path):
            os.mkdir(path)
        path += 'dataset-{}'.format(self.args.corpus)
        if self.args.datasetTag:
            path += '-' + self.args.datasetTag
        return path

    def makeLighter(self, ratioDataset):
        """Only keep a small fraction of the dataset, given by the ratio
        """
        #if not math.isclose(ratioDataset, 1.0):
        #    self.shuffle()  # Really ?
        #    print('WARNING: Ratio feature not implemented !!!')
        pass

    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.trainingSamples)

    def _createBatch(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args.batchSize !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """
        if args.full_dialog:
            batch = Batch()
            batchSize = len(samples)

            # Create the batch tensor
            for j in range(batchSize):
                # Unpack the sample
                sample2 = samples[j]
                temp1 = []
                temp2 = []
                temp3 = []
                temp4 = []
                for i, sample in enumerate(sample2):
                    if not self.args.test and self.args.watsonMode:  # Watson mode: invert question and answer
                        sample = list(reversed(sample))
                    if not self.args.test and self.args.autoEncode:  # Autoencode: use either the question or answer for both input and output
                        k = random.randint(0, 1)
                        sample = (sample[k], sample[k])
                    # TODO: Why re-processed that at each epoch ? Could precompute that
                    # once and reuse those every time. Is not the bottleneck so won't change
                    # much ? and if preprocessing, should be compatible with autoEncode & cie.

                    temp1.append(list(reversed(sample[0])))  # Reverse inputs (and not outputs), little trick as defined on the original seq2seq paper
                    temp2.append([self.goToken] + sample[1] + [self.eosToken])  # Add the <go> and <eos> tokens
                    temp3.append(temp2[-1][1:])  # Same as decoder, but shifted to the left (ignore the <go>)

                    # Long sentences should have been filtered during the dataset creation
                    assert len(temp1[i]) <= self.args.maxLengthEnco
                    assert len(temp2[i]) <= self.args.maxLengthDeco

                    # TODO: Should use tf batch function to automatically add padding and batch samples
                    # Add padding & define weight
                    temp1[i]   = [self.padToken] * (self.args.maxLengthEnco  - len(temp1[i])) + temp1[i]  # Left padding for the input
                    temp4.append([1.0] * len(temp3[i]) + [0.0] * (self.args.maxLengthDeco - len(temp3[i])))
                    temp2[i] = temp2[i] + [self.padToken] * (self.args.maxLengthDeco - len(temp2[i]))
                    temp3[i]  = temp3[i]  + [self.padToken] * (self.args.maxLengthDeco - len(temp3[i]))
                if len(temp1)>0:
                    batch.encoderSeqs.append(temp1)
                    batch.decoderSeqs.append(temp2)
                    batch.targetSeqs.append(temp3)
                    batch.weights.append(temp4)

            # temp=[]
            # for p in batch.encoderSeqs:
            #     # print(p)
            #     # Simple hack to reshape the batch
            #     encoderSeqsT = []  # Corrected orientation
            #     for i in range(self.args.maxLengthEnco):
            #         encoderSeqT = []
            #         for j in range(batchSize):
            #             encoderSeqT.append(p[j][i])
            #         encoderSeqsT.append(encoderSeqT)
            #     temp.append(encoderSeqsT)
            # batch.encoderSeqs = temp
            #
            # temp2 = []
            # temp3 = []
            # temp4 = []
            # for d,t,w in zip(batch.decoderSeqs, batch.targetSeqs, batch.weights):
            #     decoderSeqsT = []
            #     targetSeqsT = []
            #     weightsT = []
            #     for i in range(self.args.maxLengthDeco):
            #         decoderSeqT = []
            #         targetSeqT = []
            #         weightT = []
            #         for j in range(batchSize):
            #             decoderSeqT.append(d[j][i])
            #             targetSeqT.append(t[j][i])
            #             weightT.append(w[j][i])
            #         decoderSeqsT.append(decoderSeqT)
            #         targetSeqsT.append(targetSeqT)
            #         weightsT.append(weightT)
            #     temp2.append(decoderSeqsT)
            #     temp3.append(targetSeqsT)
            #     temp4.append(weightsT)
            # batch.decoderSeqs = temp2
            # batch.targetSeqs = temp3
            # batch.weights = temp4
        else:
            batch = Batch()
            batchSize = len(samples)

            # Create the batch tensor
            for i in range(batchSize):
                # Unpack the sample
                sample = samples[i]
                if not self.args.test and self.args.watsonMode:  # Watson mode: invert question and answer
                    sample = list(reversed(sample))
                if not self.args.test and self.args.autoEncode:  # Autoencode: use either the question or answer for both input and output
                    k = random.randint(0, 1)
                    sample = (sample[k], sample[k])
                # TODO: Why re-processed that at each epoch ? Could precompute that
                # once and reuse those every time. Is not the bottleneck so won't change
                # much ? and if preprocessing, should be compatible with autoEncode & cie.
                batch.encoderSeqs.append(list(reversed(sample[
                                                           0])))  # Reverse inputs (and not outputs), little trick as defined on the original seq2seq paper
                batch.decoderSeqs.append([self.goToken] + sample[1] + [self.eosToken])  # Add the <go> and <eos> tokens
                batch.targetSeqs.append(
                    batch.decoderSeqs[-1][1:])  # Same as decoder, but shifted to the left (ignore the <go>)

                # Long sentences should have been filtered during the dataset creation
                assert len(batch.encoderSeqs[i]) <= self.args.maxLengthEnco
                assert len(batch.decoderSeqs[i]) <= self.args.maxLengthDeco

                # TODO: Should use tf batch function to automatically add padding and batch samples
                # Add padding & define weight
                batch.encoderSeqs[i] = [self.padToken] * (self.args.maxLengthEnco - len(batch.encoderSeqs[i])) + \
                                       batch.encoderSeqs[i]  # Left padding for the input
                batch.weights.append(
                    [1.0] * len(batch.targetSeqs[i]) + [0.0] * (self.args.maxLengthDeco - len(batch.targetSeqs[i])))
                batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.padToken] * (
                self.args.maxLengthDeco - len(batch.decoderSeqs[i]))
                batch.targetSeqs[i] = batch.targetSeqs[i] + [self.padToken] * (
                self.args.maxLengthDeco - len(batch.targetSeqs[i]))

            # Simple hack to reshape the batch
            encoderSeqsT = []  # Corrected orientation
            for i in range(self.args.maxLengthEnco):
                encoderSeqT = []
                for j in range(batchSize):
                    encoderSeqT.append(batch.encoderSeqs[j][i])
                encoderSeqsT.append(encoderSeqT)
            batch.encoderSeqs = encoderSeqsT

            decoderSeqsT = []
            targetSeqsT = []
            weightsT = []
            for i in range(self.args.maxLengthDeco):
                decoderSeqT = []
                targetSeqT = []
                weightT = []
                for j in range(batchSize):
                    decoderSeqT.append(batch.decoderSeqs[j][i])
                    targetSeqT.append(batch.targetSeqs[j][i])
                    weightT.append(batch.weights[j][i])
                decoderSeqsT.append(decoderSeqT)
                targetSeqsT.append(targetSeqT)
                weightsT.append(weightT)
            batch.decoderSeqs = decoderSeqsT
            batch.targetSeqs = targetSeqsT
            batch.weights = weightsT

        # # Debug
        # self.printBatch(batch)  # Input inverted, padding should be correct
        # print(self.sequence2str(samples[0][0]))
        # print(self.sequence2str(samples[0][1]))  # Check we did not modified the original sample

        return batch

    def getBatches(self):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        self.shuffle()

        batches = []

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, self.getSampleSize(), self.args.batchSize):
                yield self.trainingSamples[i:min(i + self.args.batchSize, self.getSampleSize())]

        # TODO: Should replace that by generator (better: by tf.queue)

        for samples in genNextSamples():
            batch = self._createBatch(samples)
            batches.append(batch)
        return batches

    def getSampleAtIndexRange(self, index_from, index_to):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        return self._createBatch(self.trainingSamples[index_from: index_to])

    def getSampleSize(self):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.trainingSamples)

    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2id)

    def loadCorpus(self):
        """Load/create the conversations data
        """
        datasetExist = os.path.isfile(self.filteredSamplesPath)
        if not datasetExist:  # First time we load the database: creating all files
            print('Training samples not found. Creating dataset...')

            datasetExist = os.path.isfile(self.fullSamplesPath)  # Try to construct the dataset from the preprocessed entry
            if not datasetExist:
                print('Constructing full dataset...')

                optional = ''
                if self.args.corpus == 'lightweight':
                    if not self.args.datasetTag:
                        raise ValueError('Use the --datasetTag to define the lightweight file to use.')
                    optional = os.sep + self.args.datasetTag  # HACK: Forward the filename

                # Corpus creation
                corpusData = TextData.availableCorpus[self.args.corpus](self.corpusDir + optional)
                self.createFullCorpus(corpusData.getConversations())
                self.saveDataset(self.fullSamplesPath)
            else:
                self.loadDataset(self.fullSamplesPath)
            self._printStats()

            print('Filtering words (vocabSize = {} and wordCount > {})...'.format(
                self.args.vocabularySize,
                self.args.filterVocab
            ))
            self.filterFromFull()  # Extract the sub vocabulary for the given maxLength and filterVocab

            # Saving
            print('Saving dataset...')
            self.saveDataset(self.filteredSamplesPath)  # Saving tf samples
        else:
            self.loadDataset(self.filteredSamplesPath)

        assert self.padToken == 0

    def saveDataset(self, filename):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """

        with open(os.path.join(filename), 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'word2id': self.word2id,
                'id2word': self.id2word,
                'idCount': self.idCount,
                'trainingSamples': self.trainingSamples
            }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, filename):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2id = data['word2id']
            self.id2word = data['id2word']
            self.idCount = data.get('idCount', None)
            self.trainingSamples = data['trainingSamples']

            self.padToken = self.word2id['<pad>']
            self.goToken = self.word2id['<go>']
            self.eosToken = self.word2id['<eos>']
            self.unknownToken = self.word2id['<unknown>']  # Restore special words

    def filterFromFull(self):
        """ Load the pre-processed full corpus and filter the vocabulary / sentences
        to match the given model options
        """

        def mergeSentences(sentences, fromEnd=False):
            """Merge the sentences until the max sentence length is reached
            Also decrement id count for unused sentences.
            Args:
                sentences (list<list<int>>): the list of sentences for the current line
                fromEnd (bool): Define the question on the answer
            Return:
                list<int>: the list of the word ids of the sentence
            """
            # We add sentence by sentence until we reach the maximum length
            merged = []

            # If question: we only keep the last sentences
            # If answer: we only keep the first sentences
            if fromEnd:
                sentences = reversed(sentences)

            for sentence in sentences:

                # If the total length is not too big, we still can add one more sentence
                if len(merged) + len(sentence) <= self.args.maxLength:
                    if fromEnd:  # Append the sentence
                        merged = sentence + merged
                    else:
                        merged = merged + sentence
                else:  # If the sentence is not used, neither are the words
                    for w in sentence:
                        self.idCount[w] -= 1
            return merged

        newSamples = []

        # 1st step: Iterate over all words and add filters the sentences
        # according to the sentence lengths
        if not args.full_dialog:
            for inputWords, targetWords in tqdm(self.trainingSamples, desc='Filter sentences:', leave=False):
                inputWords = mergeSentences(inputWords, fromEnd=True)
                targetWords = mergeSentences(targetWords, fromEnd=False)

                newSamples.append([inputWords, targetWords])
        else:
            for p in tqdm(self.trainingSamples, desc='Filter sentences:', leave=False):
                temp=[]
                for inputWords, targetWords in p:
                    inputWords = mergeSentences(inputWords, fromEnd=True)
                    targetWords = mergeSentences(targetWords, fromEnd=False)
                    temp.append([inputWords, targetWords])
                newSamples.append(temp)
        print(len(newSamples))
        words = []

        # WARNING: DO NOT FILTER THE UNKNOWN TOKEN !!! Only word which has count==0 ?

        # 2nd step: filter the unused words and replace them by the unknown token
        # This is also where we update the correnspondance dictionaries
        specialTokens = {  # TODO: bad HACK to filter the special tokens. Error prone if one day add new special tokens
            self.padToken,
            self.goToken,
            self.eosToken,
            self.unknownToken
        }
        newMapping = {}  # Map the full words ids to the new one (TODO: Should be a list)
        newId = 0

        selectedWordIds = collections \
            .Counter(self.idCount) \
            .most_common(self.args.vocabularySize or None)  # Keep all if vocabularySize == 0
        selectedWordIds = {k for k, v in selectedWordIds if v > self.args.filterVocab}
        selectedWordIds |= specialTokens

        for wordId, count in [(i, self.idCount[i]) for i in range(len(self.idCount))]:  # Iterate in order
            if wordId in selectedWordIds:  # Update the word id
                newMapping[wordId] = newId
                word = self.id2word[wordId]  # The new id has changed, update the dictionaries
                del self.id2word[wordId]  # Will be recreated if newId == wordId
                self.word2id[word] = newId
                self.id2word[newId] = word
                newId += 1
            else:  # Cadidate to filtering, map it to unknownToken (Warning: don't filter special token)
                newMapping[wordId] = self.unknownToken
                del self.word2id[self.id2word[wordId]]  # The word isn't used anymore
                del self.id2word[wordId]

        # Last step: replace old ids by new ones and filters empty sentences
        def replace_words(words):
            valid = False  # Filter empty sequences
            for i, w in enumerate(words):
                words[i] = newMapping[w]
                if words[i] != self.unknownToken:  # Also filter if only contains unknown tokens
                    valid = True
            return valid

        self.trainingSamples.clear()


        if not args.full_dialog:
            for inputWords, targetWords in tqdm(newSamples, desc='Replace ids:', leave=False):
                valid = True
                valid &= replace_words(inputWords)
                valid &= replace_words(targetWords)
                valid &= targetWords.count(self.unknownToken) <= args.filterUknown  # Filter target with out-of-vocabulary target words ?
                valid &= len(inputWords)<=args.maxLengthEnco
                targetWords=targetWords[:args.maxLengthDeco]
                if valid:
                    self.trainingSamples.append([inputWords, targetWords])  # TODO: Could replace list by tuple
        else:
            for p in tqdm(newSamples, desc='Replace ids:', leave=False):
                temp=[]
                for inputWords, targetWords in p:
                    valid = True
                    valid &= replace_words(inputWords)
                    valid &= replace_words(targetWords)
                    valid &= targetWords.count(
                        self.unknownToken) <= args.filterUknown  # Filter target with out-of-vocabulary target words ?
                    valid &= len(inputWords) <= args.maxLengthEnco
                    targetWords = targetWords[:args.maxLengthDeco]
                    if valid:
                       temp.append([inputWords, targetWords])
                self.trainingSamples.append(temp)

        self.idCount.clear()  # Not usefull anymore. Free data

    def createFullCorpus(self, conversations):
        """Extract all data from the given vocabulary.
        Save the data on disk. Note that the entire corpus is pre-processed
        without restriction on the sentence length or vocab size.
        """
        # Add standard tokens
        self.padToken = self.getWordId('<pad>')  # Padding (Warning: first things to add > id=0 !!)
        self.goToken = self.getWordId('<go>')  # Start of sequence
        self.eosToken = self.getWordId('<eos>')  # End of sequence
        self.unknownToken = self.getWordId('<unknown>')  # Word dropped from vocabulary

        # Preprocessing data

        for conversation in tqdm(conversations, desc='Extract conversations'):
            self.extractConversation(conversation)

        # The dataset will be saved in the same order it has been extracted

    def extractConversation(self, conversation):
        """Extract the sample lines from the conversations
        Args:
            conversation (Obj): a conversation object containing the lines to extract
        """

        if self.args.skipLines:  # WARNING: The dataset won't be regenerated if the choice evolve (have to use the datasetTag)
            step = 2
        else:
            step = 1

        # Iterate over all the lines of the conversation
        all_pairs=[]
        for i in tqdm_wrap(
            range(0, len(conversation['lines']) - 1, step),  # We ignore the last line (no answer for it)
            desc='Conversation',
            leave=False
        ):
            inputLine  = conversation['lines'][i]
            targetLine = conversation['lines'][i+1]

            inputWords  = self.extractText(inputLine['text'])
            targetWords = self.extractText(targetLine['text'])
            # print('{} vs {}'.format(inputLine, targetLine))
            # print('{} vs {}'.format(inputWords, targetWords))
            if inputWords and targetWords:
                all_pairs.append([inputWords, targetWords])
            if not args.full_dialog:
                if inputWords and targetWords:  # Filter wrong samples (if one of the list is empty)
                    self.trainingSamples.append([inputWords, targetWords])
        if args.full_dialog:
            if all_pairs:
                # print(all_pairs)
                self.trainingSamples.append(all_pairs)
        # raise False

    def extractText(self, line):
        """Extract the words from a sample lines
        Args:
            line (str): a line containing the text to extract
        Return:
            list<list<int>>: the list of sentences of word ids of the sentence
        """
        line = line.lower()
        text = re.sub(r"i'm", "i am", line)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        line =''.join([ch if ch in EN_WHITELIST else ' ' for ch in text])




        # print(line)
        # raise False
        sentences = []  # List[List[str]]

        # Extract sentences
        sentencesToken = nltk.sent_tokenize(line)

        # We add sentence by sentence until we reach the maximum length
        for i in range(len(sentencesToken)):
            tokens = nltk.word_tokenize(sentencesToken[i])

            tempWords = []
            for token in tokens:
                tempWords.append(self.getWordId(token))  # Create the vocabulary and the training sentences

            sentences.append(tempWords)

        return sentences

    def getWordId(self, word, create=True):
        """Get the id of the word (and add it to the dictionary if not existing). If the word does not exist and
        create is set to False, the function will return the unknownToken value
        Args:
            word (str): word to add
            create (Bool): if True and the word does not exist already, the world will be added
        Return:
            int: the id of the word created
        """
        # Should we Keep only words with more than one occurrence ?

        word = word.lower()  # Ignore case

        # At inference, we simply look up for the word
        if not create:
            wordId = self.word2id.get(word, self.unknownToken)
        # Get the id if the word already exist
        elif word in self.word2id:
            wordId = self.word2id[word]
            self.idCount[wordId] += 1
        # If not, we create a new entry
        else:
            wordId = len(self.word2id)
            self.word2id[word] = wordId
            self.id2word[wordId] = word
            self.idCount[wordId] = 1

        return wordId

    def printBatch(self, batch):
        """Print a complete batch, useful for debugging
        Args:
            batch (Batch): a batch object
        """
        print('----- Print batch -----')
        for i in range(len(batch.encoderSeqs[0])):  # Batch size
            print('Encoder: {}'.format(self.batchSeq2str(batch.encoderSeqs, seqId=i)))
            print('Decoder: {}'.format(self.batchSeq2str(batch.decoderSeqs, seqId=i)))
            print('Targets: {}'.format(self.batchSeq2str(batch.targetSeqs, seqId=i)))
            print('Weights: {}'.format(' '.join([str(weight) for weight in [batchWeight[i] for batchWeight in batch.weights]])))

    def sequence2str(self, sequence, clean=False, reverse=False):
        """Convert a list of integer into a human readable string
        Args:
            sequence (list<int>): the sentence to print
            clean (Bool): if set, remove the <go>, <pad> and <eos> tokens
            reverse (Bool): for the input, option to restore the standard order
        Return:
            str: the sentence
        """

        if not sequence:
            return ''

        if not clean:
            return ' '.join([self.id2word[idx] for idx in sequence])

        sentence = []
        for wordId in sequence:
            if wordId == self.eosToken:  # End of generated sentence
                break
            elif wordId != self.padToken and wordId != self.goToken:
                sentence.append(self.id2word[wordId])

        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()

        return self.detokenize(sentence)

    def detokenize(self, tokens):
        """Slightly cleaner version of joining with spaces.
        Args:
            tokens (list<string>): the sentence to print
        Return:
            str: the sentence
        """
        return ''.join([
            ' ' + t if not t.startswith('\'') and
                       t not in string.punctuation
                    else t
            for t in tokens]).strip().capitalize()

    def batchSeq2str(self, batchSeq, seqId=0, **kwargs):
        """Convert a list of integer into a human readable string.
        The difference between the previous function is that on a batch object, the values have been reorganized as
        batch instead of sentence.
        Args:
            batchSeq (list<list<int>>): the sentence(s) to print
            seqId (int): the position of the sequence inside the batch
            kwargs: the formatting options( See sequence2str() )
        Return:
            str: the sentence
        """
        sequence = []
        for i in range(len(batchSeq)):  # Sequence length
            sequence.append(batchSeq[i][seqId])
        return self.sequence2str(sequence, **kwargs)

    def sentence2enco(self, sentence):
        """Encode a sequence and return a batch as an input for the model
        Return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """

        if sentence == '':
            return None

        # First step: Divide the sentence in token
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > self.args.maxLength:
            return None

        # Second step: Convert the token in word ids
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))  # Create the vocabulary and the training sentences

        # Third step: creating the batch (add padding, reverse)
        batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output

        return batch

    def deco2sentence(self, decoderOutputs):
        """Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>):
        """
        sequence = []

        # Choose the words with the highest prediction score
        for out in decoderOutputs:
            sequence.append(np.argmax(out))  # Adding each predicted word ids

        return sequence  # We return the raw sentence. Let the caller do some cleaning eventually

    def playDataset(self):
        """Print a random dialogue from the dataset
        """
        print('Randomly play samples:')
        for i in range(self.args.playDataset):
            idSample = random.randint(0, len(self.trainingSamples) - 1)
            print('Q: {}'.format(self.sequence2str(self.trainingSamples[idSample][0], clean=True)))
            print('A: {}'.format(self.sequence2str(self.trainingSamples[idSample][1], clean=True)))
            print()
        pass


def tqdm_wrap(iterable, *args, **kwargs):
    """Forward an iterable eventually wrapped around a tqdm decorator
    The iterable is only wrapped if the iterable contains enough elements
    Args:
        iterable (list): An iterable object which define the __len__ method
        *args, **kwargs: the tqdm parameters
    Return:
        iter: The iterable eventually decorated
    """
    if len(iterable) > 100:
        return tqdm(iterable, *args, **kwargs)
    return iterable

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test',
                            nargs='?', default=None,
                            help='if present, launch the program try to answer all sentences from data/test/ with'
                                 ' the defined model(s), in interactive mode, the user can wrote his own sentences,'
                                 ' use daemon mode to integrate the chatbot in another program')
    parser.add_argument('--playDataset', type=int, nargs='?', const=10, default=None,
                            help='if set, the program  will randomly play some samples(can be use conjointly with createDataset if this is the only action you want to perform)')
    parser.add_argument('--datasetTag', type=str, default='',
                             help='add a tag to the dataset (file where to load the vocabulary and the precomputed samples, not the original corpus). Useful to manage multiple versions. Also used to define the file used for the lightweight format.')  # The samples are computed from the corpus if it does not exist already. There are saved in \'data/samples/\'
    parser.add_argument('--ratioDataset', type=float, default=1.0,
                             help='ratio of dataset used to avoid using the whole dataset')  # Not implemented, useless ?
    parser.add_argument('--maxLength', type=int, default=20,
                             help='maximum length of the sentence (for input and output), define number of maximum step of the RNN')
    parser.add_argument('--filterVocab', type=int, default=0,
                             help='remove rarelly used words (by default words used only once). 0 to keep all words.')
    parser.add_argument('--filterUknown', type=int, default=1)
    parser.add_argument('--skipLines', action='store_true', default=False,
                             help='Generate training samples by only using even conversation lines as questions (and odd lines as answer). Useful to train the network on a particular person.')
    parser.add_argument('--vocabularySize', type=int, default=40000,
                             help='Limit the number of words in the vocabulary (0 for unlimited)')

    parser.add_argument('--rootDir', type=str, default='./raw_data', help='folder where to look for the models and data')
    parser.add_argument('--watsonMode', action='store_true',
                            help='Inverse the questions and answer when training (the network try to guess the question)')
    parser.add_argument('--autoEncode', action='store_true',
                            help='Randomly pick the question or the answer and use it both as input and output')
    parser.add_argument('--batchSize', type=int, default=1, help='mini-batch size')
    parser.add_argument('--full_dialog', action='store_true', default=False)
    """
    modify the following params:
    """
    parser.add_argument('--corpus', choices=TextData.corpusChoices(), default=TextData.corpusChoices()[0],
                             help='corpus on which extract the dataset.')
    parser.add_argument('--save_file_dir', default="./data/", help="where to put preprocessed data")
    parser.add_argument('--maxLengthEnco', type=int, default=20, help="replace maxLength")
    parser.add_argument('--maxLengthDeco', type=int, default=10,  help="replace maxLength")
    parser.add_argument('--testnum', type=int, default=10, help="num of printed samples")


    args = parser.parse_args()
    if args.full_dialog:
        print("dialog in structural form, not used in VMED paper --> for future use")
        raw_text_path = 'dual_qa_temp'
    else:
        print("dialog in concatenated form,  used in VMED paper")
        raw_text_path = 'single_qa_temp'

    textData= TextData(args)
    test_num= args.testnum
    print(len(textData.word2id))
    print(len(textData.id2word))
    print(textData.word2id['<eos>'])
    print(textData.word2id['<go>'])
    print(textData.word2id['<pad>'])
    print(textData.word2id['<unknown>'])
    print('number of dialogs {}'.format(textData.getSampleSize()))

    if args.full_dialog:

        ntrains=[]
        len_con=[]
        for d in textData.trainingSamples:
            if d:
                ntrains.append(d)
                len_con.append(len(d))
        for c in range(test_num):
            ind = np.random.randint(0, len(ntrains), 1)[0]
            d=ntrains[ind]
            for p1,p2 in d:
                print(p1)
                print(textData.sequence2str(p1, clean=True, reverse=False))
                print(p2)
                print(textData.sequence2str(p2, clean=True, reverse=False))
            print('---')
        print('number of dialogs {}'.format(len(ntrains)))
        print('min len {} max len {} avg len {}'.format(np.min(len_con), np.max(len_con),np.mean(len_con)))
        print('save my own data...')
        save_obj=(textData.word2id, textData.id2word, ntrains)
        pickle.dump(save_obj, open(args.save_file_dir+"/full_{}.pkl".format(args.corpus),'wb'))
    else:
        len_con = []
        ntrains = []
        for d in textData.trainingSamples:
            if d:
                ntrains.append(d)
                len_con.append(len(d[0]))
                len_con.append(len(d[1]))
        for c in range(test_num):
            ind = np.random.randint(0, len(ntrains), 1)[0]
            p1,p2=ntrains[ind]
            print(p1)
            print(textData.sequence2str(p1, clean=True, reverse=False))
            print(p2)
            print(textData.sequence2str(p2, clean=True, reverse=False))
            print('xxxxx')
        print('number of dialogs {}'.format(len(ntrains)))
        print('min len {} max len {} avg len {}'.format(np.min(len_con), np.max(len_con), np.mean(len_con)))
        print('save my own data...')
        save_obj = (textData.word2id, textData.id2word, ntrains)
        pickle.dump(save_obj, open(args.save_file_dir+"/single_{}.pkl".format(args.corpus),'wb'))

        # seq=[]
        # ind= np.random.randint(0, textData.getSampleSize(),1)[0]
        # batch = textData.getSampleAtIndexRange(c, c+1)
        # for b in range(len(batch.encoderSeqs)):
        #     for p in range(len(batch.encoderSeqs[b])):
        #         bs = []
        #         abs = []
        #         for i in range(args.maxLengthEnco):
        #             bs.append(batch.encoderSeqs[b][p][i])
        #         for i in range(args.maxLengthDeco):
        #             abs.append(batch.decoderSeqs[b][p][i])
        #         seq.append(bs)
        #         print(bs)
        #         print(textData.sequence2str(bs, clean=True, reverse=True))
        #         print(abs)
        #         print(textData.sequence2str(abs, clean=True, reverse=False))
        #     print('++')
        # print('----')

        # for i in range(self.args.maxLengthDeco):
        #     feedDict[self.decoderInputs[i]] = batch.decoderSeqs[i]
        #     feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
        #     feedDict[self.decoderWeights[i]] = batch.weights[i]
