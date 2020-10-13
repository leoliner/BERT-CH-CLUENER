# issue：keras与tensorflow版本不匹配
# 解决：安装1.14.0的tf
# from keras_bert import get_base_dict
#
# token_dict = get_base_dict()
# print(token_dict)

# issue：测试range函数用法
# print(list(range(1, 40)))
# import json

# issue：测试label2id.json文件的读写和使用
# label2id = json.loads(open("../ner_data_set/cluener_public/label2id.json").read())
# label_list = []
# id_list = []
# for label in label2id:
#     label_list.append(label)
#     id_list.append(label2id[label])
# label_list.append("**NULL**")
# label_list.append("**NULL**")
# id_list.append(0)
# id_list.append(0)
#
# print(label_list)
# print(id_list)
#
# reallen = len(label_list)
# print("labels     : %s " % " ".join([str(x) for x in label_list[:reallen]]))
# print("labels     : %s " % " ".join([str(x) for x in id_list[:reallen]]))

# issue：标签的命名出现问题：应该是_，函数中认为是-，导致报错，已解决
# def get_result(sentence, label):
#     result_words = []
#     result_pos = []
#     temp_word = []
#     temp_pos = ''
#     print(len(sentence))
#     for i in range(min(len(sentence), len(label))):  # 仅判断句子真实长度
#         print(result_words)
#         print(result_pos)
#         print(temp_word)
#         print(temp_pos)
#         if label[i].startswith('O'):  # 非实体
#             print(1)
#             if len(temp_word) > 0:
#                 result_words.append([min(temp_word), max(temp_word)])
#                 result_pos.append(temp_pos)
#             temp_word = []
#             temp_pos = ''
#         elif label[i].startswith('S_'):  # S表示单个字的词
#             print(2)
#             if len(temp_word) > 0:
#                 result_words.append([min(temp_word), max(temp_word)])
#                 result_pos.append(temp_pos)
#             result_words.append([i, i])
#             result_pos.append(label[i].split('_')[1])
#             temp_word = []
#             temp_pos = ''
#         elif label[i].startswith('B_'):  # B表示词的首字
#             print(3)
#             if len(temp_word) > 0:
#                 result_words.append([min(temp_word), max(temp_word)])
#                 result_pos.append(temp_pos)
#             temp_word = [i]  # 字的位置
#             temp_pos = label[i].split('_')[1]
#         elif label[i].startswith('M_'):  # M表示词的中间字
#             print(4)
#             if len(temp_word) > 0:
#                 temp_word.append(i)
#                 if temp_pos == '':
#                     temp_pos = label[i].split('_')[1]
#         else:  # E表示词的结束字
#             if len(temp_word) > 0:
#                 temp_word.append(i)
#                 if temp_pos == '':
#                     temp_pos = label[i].split('_')[1]
#                 result_words.append([min(temp_word), max(temp_word)])
#                 result_pos.append(temp_pos)
#             temp_word = []
#             temp_pos = ''
#     return result_words, result_pos
#
#
# text = "去年11月30日，李先生来到茶店子东街一家银行取钱，准备购买家具。输入密码后，"
# label = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B_address', 'M_address', 'M_address',
#          'M_address', 'E_address', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
#          'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
#          'O', 'B_address', 'B_address', 'M_address', 'M_address']
# result_words, result_pos = get_result(text, label)
# print(result_words)
# print(result_pos)

# issue：for_test_bert_v2_score.py 文件报错：open（）时的gbk字符编码问题
# 解决：open时直接设置encoding='utf-8'即可
import os
import tensorflow as tf

from test_code import for_test_bert_v2_score as score

flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "data_dir", '../ner_data_set/cluener_public',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")
flags.DEFINE_string(
    "output_dir", '../output2/epoch-05-10/',
    "The output directory where the model checkpoints will be written.")

output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.json")
gold_file = os.path.join(FLAGS.data_dir, "dev.json")
print(gold_file)
f_score, avg = score.get_f1_score(pre_file=output_eval_file, gold_file=gold_file)
print(f_score)
print(avg)
