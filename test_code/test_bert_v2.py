# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner. Using cluener2020 dataset."""
# 606行——使用crf加快训练速度

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import csv
import os
from bert import modeling
from bert import optimization
from bert import tokenization
from bert import tf_metrics
from test_code import for_test_bert_v2_score as score
import tensorflow as tf
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", '../ner_data_set/cluener_public',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", '../checkpoint/chinese_L-12_H-768_A-12/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", 'BERT-CH-CLUENER', "The name of the task to train.")

flags.DEFINE_string("vocab_file", '../checkpoint/chinese_L-12_H-768_A-12/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", '../output2/epoch-10-15/',
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint",
    # '../checkpoint/chinese_L-12_H-768_A-12/bert_model.ckpt',
    "../output2/epoch-05-10/model.ckpt-1679",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 64,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 5.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        参数：
        InputExample类。
        一个输入样本包含id，text_a，text_b和label四个属性
            text_a和text_b分别表示第一个句子和第二个句子，因此text_b是可选的。

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    PaddingInputExample类。
    定义这个类是因为TPU只支持固定大小的batch，在eval和predict的时候需要对batch做padding。
      如不使用TPU，则无需使用这个类。

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data.
    InputFeatures类
    定义了输入到estimator的model_fn中的feature，
        包括input_ids，
        input_mask，
        segment_ids（即0或1，表明词语属于第一个句子还是第二个句子，在BertModel中被看作token_type_id），
        label_id  在token级任务里是个list哦
        以及is_real_example
    """

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets.
       DataProcessor类以及四个公开数据集对应的子类。（如229行的XnliProcessor(DataProcessor)）
       一个数据集对应一个DataProcessor子类，需要继承四个函数：
          分别从文件目录中获得train，eval和predict样本的三个函数以及一个获取label集合的函数。
      如果需要在自己的数据集上进行finetune，则需要自己实现一个DataProcessor的子类，
          按照自己数据集的格式从目录中获取样本。
      注意！在这一步骤中，对没有label的predict样本，要指定一个label的默认值供统一的model_fn使用。
    """

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    # 验证集
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class ClueNerProcessor(DataProcessor):
    """
    处理InputExample类。
    token级任务。
    *一个输入样本InputExample包含id，text_a，text_b和label四个属性，这里text_b为空
    """

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json"), "train"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json"), "dev"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json"), "test"), "test")

    def get_labels(self):
        # label2id
        label2id = json.loads(open("../ner_data_set/cluener_public/label2id.json").read())
        label_list = []
        for label in label2id:
            label_list.append(label)
        return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            labels = []
            for l in line[-1]:
                labels.append(tokenization.convert_to_unicode(l))
            examples.append(InputExample(guid=guid, text_a=text_a, label=labels))
        print("examples: " + str(len(examples)))
        return examples

    def _read_json(self, input_file, set_type):
        """
        返回一个元组tuple（string, list）
        :param input_file: 文件路径
        :param set_type: 数据集类型
        :return: (text, label_list)
        """
        lines = []
        print(set_type)
        if (set_type == "train") or (set_type == "dev"):
            for line in open(input_file, encoding='utf-8'):
                if not line.strip():
                    continue
                _ = json.loads(line.strip())
                len_ = len(_["text"])
                labels = ["O"] * len_  # 初始值是全O（大写的o），即非实体
                # SBME标记，S表示单个字的词，B表示词的首字，M表示词的中间字，E表示词的结束
                for k, v in _["label"].items():
                    for kk, vv in v.items():
                        for vvv in vv:
                            span = vvv
                            s = span[0]
                            e = span[1] + 1
                            # print(s, e)
                            if e - s == 1:
                                labels[s] = "S_" + k
                            else:
                                labels[s] = "B_" + k
                                for i in range(s + 1, e - 1):
                                    labels[i] = "M_" + k
                                labels[e - 1] = "E_" + k
                text = _["text"]
                lines.append((text, labels))
        elif set_type == "test":
            for line in open(input_file, encoding='utf-8'):
                if not line.strip():
                    continue
                _ = json.loads(line.strip())
                len_ = len(_["text"])
                labels = ["O"] * len_  # 初始值是全O（大写的o），即非实体
                text = _["text"]
                lines.append((text, labels))
        print("lines: " + str(len(lines)))
        return lines


# 2020.10.4
def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`.
    convert_single_example函数。
      可以将一个InputExample转换为InputFeatures，里面调用了tokenizer进行一些句子清洗和预处理工作，
          同时截断了长度超过最大值的句子。

    需要修改的部分：token级任务的example.label是一个list，而不是仅一个label标签
                 直接tokenizer.tokenize会将某些词从一个字分词成 字1 和 #字2 的组合，导致字和label对应不上
    """

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label2id = {}
    for (i, label) in enumerate(label_list):
        label2id[label] = i

    # tokens_a = tokenizer.tokenize(example.text_a)
    # 这里直接分词会将某些词从一个字分词成 字1 和 #字2 的组合，导致字和label对应不上
    tokens_a = []
    tokens_a_labels = []

    for i, word in enumerate(list(example.text_a)):  # example.text_a是一个字符串哦
        token = tokenizer.tokenize(word)
        tokens_a.extend(token)
        word_label = example.label[i]
        for m in range(len(token)):
            if m == 0:
                tokens_a_labels.append(word_label)
            else:
                print("unknown tokens")
                tokens_a_labels.append("O")  # 标记为非实体O

    tokens_b = None
    if example.text_b:  # 这里仅单句，就不进行上述操作了
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
            tokens_a_labels = tokens_a_labels[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    tokens = []  # 分词后的text_a的所有词
    segment_ids = []  # 分割标记
    label_ids = []  # 每个词对应的标签的id值

    # 在序列最前方加上开始标签[CLS]
    tokens.append("[CLS]")  # 开始标签[CLS]
    segment_ids.append(0)  # [CLS]对应的也是0
    # [CLS] [SEP] 可以为 他们构建标签，或者 统一到某个标签，反正他们是不变的，基本不参加训练 即：x-l 永远不变
    # 这里[CLS] [SEP] 被统一至label2id里的第0号标签，即非实体O
    label_ids.append(0)

    # 处理句子序列text_a
    for i, token in enumerate(tokens_a):
        tokens.append(token)
        segment_ids.append(0)  # text_a的所有segment_ids都是0
        label_ids.append(label2id[tokens_a_labels[i]])

    # 在序列最后方加上结束/分割标签[SEP]
    tokens.append("[SEP]")  # 分隔/句尾，cluener中表示句尾，因为没有text_b
    segment_ids.append(0)  # [SEP]对应的也是0
    label_ids.append(0)

    # if tokens_b:
    #   for token in tokens_b:
    #     tokens.append(token)
    #     segment_ids.append(1)
    #   tokens.append("[SEP]")
    #   segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 将每个token转为对应的id（查vocab.txt表）

    # 处理input_mask，为不足max_seq_length的序列补足长度（padding 0）
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        tokens.append("**NULL**")
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # ex_index《5时 输出一些有用信息
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid       : %s" % (example.guid))
        tf.logging.info("tokens     : %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids  : %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask : %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        # tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
        tf.logging.info("label2id   : %s " % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        # label_id=label_id,
        label_id=label_ids,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file.
    file_based_convert_example_to_features函数：
    将一批InputExample转换为InputFeatures，并写入到tfrecord文件中，
      相当于实现了从原始数据集文件到tfrecord文件的转换。
    """

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_id)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator.
    file_based_input_fn_builder函数：
      这个函数根据tfrecord文件，构建estimator的input_fn，
      即先建立一个TFRecordDataset，然后进行shuffle，repeat，decode和batch操作。
    """

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            # if t.dtype == tf.int64:
            #     t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        # 数据类别集中，需要较大的buffer_size，才能有效打乱，或者再 数据处理的过程中进行打乱
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=3000)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model.
    用于构建从input_ids到prediction和loss的计算过程，
      包括建立BertModel，获取BertModel的pooled_output，即句子向量表示，然后构建隐藏层和bias，
      并计算logits和softmax，最终用cross_entropy计算出loss。
      if you want to use the token-level output, use model.get_sequence_output() instead.
    """
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope='bert'
    )

    # In the demo, we are doing a simple classification task on the entire
    # segment. 单分类任务
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead. 如NER任务

    output_layer = model.get_sequence_output()
    hidden_size = output_layer.shape[-1].value
    seq_length = output_layer.shape[-2].value
    print(output_layer.shape)

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    # print("output_weights"+str(output_layer.shape))  # 需要与笔记本对比一下，查看什么情况
    # print("output_bias"+str(output_weight.shape))
    # loss和predict要自己定义
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        # output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        # logits = tf.reshape(logits, [-1, seq_length, num_labels])  # 这里的num_labels对应着总类别数。

        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, shape=(-1, seq_length, num_labels))

        input_m = tf.count_nonzero(input_mask, -1)

        log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits, labels, input_m)
        loss = tf.reduce_mean(-log_likelihood)
        print("transition_matrix"+str(transition_matrix.shape))
        # inference
        # decode_tags, best_score = tf.contrib.crf.crf_decode(logits, transition_matrix, input_m)
        # Args
        # potentials	A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
        # transition_params	A [num_tags, num_tags] matrix of binary potentials.
        # sequence_length	A [batch_size] vector of true sequence lengths.
        # Returns
        # decode_tags	A [batch_size, max_seq_len] matrix, with dtype tf.int32. Contains the highest scoring tag indices.
        # best_score	A [batch_size] vector, containing the score of decode_tags.

        return (loss, logits)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator.
      调用create_model函数，构建estimator的model_fn。
      由于model_fn需要labels输入，为简化代码减少判断，当要进行predict时也要求传入label，
      因此DataProcessor中为每个predict样本生成了一个默认label（其取值并无意义）。
      *这里构建的是TPUEstimator，但没有TPU时，它也可以像普通estimator一样工作。
    """

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, logits) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)  # 最后一个返回的参数变成所有token的predicts

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            # 对于token级任务，eval 的 计算方式metric需要自己定义修改

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                # 评估函数，计算准确率、召回率、F1，假如改类别的话，下方数字需要修改，10是总类别数，1-6是有用的类别。B、I、E，
                # 具体见 tf.metrics里的函数
                precision = tf_metrics.precision(label_ids, predictions, 41, list(range(1, 41)), average="macro")
                recall = tf_metrics.recall(label_ids, predictions, 41, list(range(1, 41)), average="macro")
                f = tf_metrics.f1(label_ids, predictions, 41, list(range(1, 41)), average="macro")

                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f
                }

            eval_metrics = (metric_fn,
                            [total_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn,
                predictions={"probabilities": logits}
            )
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": logits},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def get_result(sentence, label):
    result_words = []
    result_pos = []
    temp_word = []
    temp_pos = ''
    print(len(sentence))
    for i in range(min(len(sentence), len(label))):  # 仅判断句子真实长度
        if label[i].startswith('O'):  # 非实体
            if len(temp_word) > 0:
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            temp_word = []
            temp_pos = ''
        elif label[i].startswith('S_'):  # S表示单个字的词
            if len(temp_word) > 0:
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            result_words.append([i, i])
            result_pos.append(label[i].split('_')[1])
            temp_word = []
            temp_pos = ''
        elif label[i].startswith('B_'):  # B表示词的首字
            if len(temp_word) > 0:
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            temp_word = [i]  # 字的位置
            temp_pos = label[i].split('_')[1]
        elif label[i].startswith('M_'):  # M表示词的中间字
            if len(temp_word) > 0:
                temp_word.append(i)
                if temp_pos == '':
                    temp_pos = label[i].split('_')[1]
        else:  # E表示词的结束字
            if len(temp_word) > 0:
                temp_word.append(i)
                if temp_pos == '':
                    temp_pos = label[i].split('_')[1]
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            temp_word = []
            temp_pos = ''
    return result_words, result_pos


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def main(_):
    """
    首先定义任务名称和processor的对应关系，因此如果定义了自己的processor，需要将其加入到processors字典中。
    其次从FLAGS中，即启动命令中读取相关参数，构建model_fn和estimator，并根据参数中的do_train，do_eval和do_predict的取值决定要进行estimator的哪些操作。
    :param _:
    :return:
    """
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "bert-ch-cluener": ClueNerProcessor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    ############# ↓tpu设置↓ #############
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    ############# ↑tpu设置↑ #############

    ############# ↓train过程↓ #############
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(  # 将一批InputExample转换为InputFeatures，并写入到tfrecord文件中
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(  # 这个函数根据tfrecord文件，构建estimator的input_fn
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    ############# ↑train过程↑ #############

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        predcit_result = estimator.predict(input_fn=eval_input_fn)
        # print(result.keys()

        ############写入验证结果###########
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.json")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Evaluate results *****")
            for (i, (example, prediction)) in enumerate(zip(eval_examples, predcit_result)):
                probabilities = prediction["probabilities"][1:-1, :]
                print(probabilities)
                if i >= num_actual_eval_examples:
                    break
                index = example.guid
                text = example.text_a
                label = [label_list[np.argmax(p)] for p in probabilities]
                print(label)
                print(example.text_a)
                result_words, result_pos = get_result(example.text_a, label)
                # print(result_words)
                # print(result_pos)
                rs = {}
                for w, p in zip(result_words, result_pos):
                    rs[p] = rs.get(p, []) + [w]
                preds = {}
                for p, ws in rs.items():
                    temp = {}
                    for w in ws:
                        word = text[w[0]: w[1] + 1]
                        temp[word] = temp.get(word, []) + [w]
                    preds[p] = temp
                output_line = json.dumps({'id': index, 'label': preds}, ensure_ascii=False) + '\n'
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_eval_examples

        ############求实际的f_score(各标签计算)###########
        gold_file = os.path.join(FLAGS.data_dir, "dev.json")
        print(gold_file)
        f_score, avg = score.get_f1_score(pre_file=output_eval_file, gold_file=gold_file)

        ############输出验证结果###########
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write("%s = %s\n" % ('f_score', str(f_score)))
            writer.write("%s = %s\n" % ('avg', str(avg)))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.json")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, (example, prediction)) in enumerate(zip(predict_examples, result)):
                probabilities = prediction["probabilities"][1:-1, :]
                # print(probabilities)
                if i >= num_actual_predict_examples:
                    break
                index = example.guid
                text = example.text_a
                label = [label_list[np.argmax(p)] for p in probabilities]
                print(label)
                print(example.text_a)
                result_words, result_pos = get_result(example.text_a, label)
                # print(result_words)
                # print(result_pos)
                rs = {}
                for w, p in zip(result_words, result_pos):
                    rs[p] = rs.get(p, []) + [w]
                preds = {}
                for p, ws in rs.items():
                    temp = {}
                    for w in ws:
                        word = text[w[0]: w[1] + 1]
                        temp[word] = temp.get(word, []) + [w]
                    preds[p] = temp
                output_line = json.dumps({'id': index, 'label': preds}, ensure_ascii=False) + '\n'
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
