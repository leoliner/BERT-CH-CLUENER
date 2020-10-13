# BERT-CH-CLUENER
*   使用bert在cluener2020细粒度标签数据集上进行NER任务
## 1.数据集
*   [CLUENER 2020](https://github.com/CLUEbenchmark/CLUENER2020)
#### 1.1.标签
`../ner_data_set/cluener_public/label2id.json`

    ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
    
    共十个实体,并使用SBME标记方式命名,加上非实体O,共41个标签值
#### 1.2训练集

`../ner_data_set/cluener_public/train.json`

#### 1.3验证集

`../ner_data_set/cluener_public/eval.json`

#### 1.4测试集

`../ner_data_set/cluener_public/test.json`

    注:未提供真实标签,需要前往CLUE线上提交测试
## 2.项目运行
#### 2.1.创建checkpoint文件夹
下载并存放bert官方提供的预训练的中文模型的参数
*   [BERT-Base, Chinese: Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

#### 2.2.创建bert文件夹
存放官方源码,需要 tensorflow >= 1.11.0
*   `git clone bert`
#### 2.3.修改flags中相关参数
`flags.DEFINE_string(...)`
*   data_dir

    `../ner_data_set/cluener_public`
*   bert_config_file

    bert官方提供的预训练的中文模型`chinese_L-12_H-768_A-12/bert_config.json`
*   vocab_file

    bert官方提供的预训练的中文模型`chinese_L-12_H-768_A-12/vocab.txt`
*   init_checkpoint

    bert官方提供的预训练的中文模型`chinese_L-12_H-768_A-12/bert_model.ckpt`
    * .ckpt实际上是三个文件
*   output_dir

    模型输出checkpoint,eval和predcit结果的文件夹位置
*   do_train,do_eval,do_predict

    `True`

#### 2.4.执行

   `python test_bert_v2.py`

#### 2.5.结果输出将至
    ../
      /output_dir/
                 /model.ckpt
                 /train.tf_record
                 /eval.tf_record
                 /predict.tf_record
                 /eval_result.json
                 /eval_result.txt
                 /test_result.json
                 /...
                   

## 3.结果示例
###### 由于test.json未给出真实标签,仅eval可计算f socre:

    epoch = 10
    f_score = {'address': 0.6351706036745407, 'book': 0.8172757475083058, 'company': 0.8032345013477088, 'game': 0.8440677966101694, 'government': 0.8134920634920635, 'movie': 0.8166089965397924, 'name': 0.8708971553610504, 'organization': 0.7920227920227921, 'position': 0.7929824561403509, 'scene': 0.7073791348600509}
    avg_f_score = 0.7893131247556825
    
###### 模型预测示例
    {"id": "test-0", "label": {"organization": {"四川敦煌学”": [[0, 5]]}, "address": {"丹棱县": [[11, 13]]}, "name": {"胡文和": [[41, 43]]}}}
    {"id": 0, "text": "四川敦煌学”。近年来，丹棱县等地一些不知名的石窟迎来了海内外的游客，他们随身携带着胡文和的著作。"}
    
    {"id": "test-1", "label": {"government": {"尼日利亚海军": [[0, 5]]}, "address": {"阿布贾": [[12, 14]]}, "company": {"尼日利亚通讯社": [[16, 22]]}}}
    {"id": 1, "text": "尼日利亚海军发言人当天在阿布贾向尼日利亚通讯社证实了这一消息。"}
    ...
