# CCKS2022开源多模态军事装备数据的事件要素抽取


多模态军事事件要素抽取，是指从面向军事特定领域多模态数据中抽取出用户所需要的事件信息并以结构化形式呈现的过程，是事件抽取、目标检测、文本识别等技术在军事领域的具体应用。CCKS 2022 组织了本次事件要素抽取评测任务，要求从互联网公开的多模态军事装备数据（文本 + 图像）中抽取相关事件类型及事件要素，提供多模态、结构化、高价值的目标装备信息抽取结果。本文采用文本事件抽取、图像目标检测以及多模态知识融合等技术，实现多模态军事事件要素抽取，以测试集上 F1值 0.53403 的最终成绩，在测评任务中排名第二名。

该文档展示了如何使用与快速复现我们<a href="https://www.biendata.xyz/competition/KYDMTJSTP/">CCKS：开源多模态军事装备数据的事件要素抽取任务</a>的方法。

## 目录结构

以下是本项目主要目录结构及说明：

```python
--CCKS2022_JS
    |--postprocess.py #后处理      
    |--run_ee.sh #主程序批处理脚本
    |--evaluation_metric.py #评价指标
    |--ckpt #模型checkpoint
    |--submit #结果文件
    |--data #数据文件
    |    |--ObjectDect #yolov5模型输入
    |    |    |--train_images
    |    |    |    |--labels
    |    |    |    |--images
    |    |    |--dev_images
    |    |    |    |--labels
    |    |    |    |--images
    |    |--result #结果文件
    |    |--EE1.0 #事件抽取输入
    |    |    |--test.json #测试集
    |    |    |--dev.json #触发词标注后验证集
    |    |    |--roles.xlsx #数据增强替换实体
    |    |    |--trigger/ #触发词抽取输入
    |    |    |--role/ #论元抽取输入
    |    |    |--train.json #触发词标注后测试集
    |    |    |--event_schema.json #事件定义schema
    |    |--raw_data #原始数据集
    |    |    |--test
    |    |    |--train
    |    |    |--val
    |    |--MutiClass #多标签多分类输入
    |    |    |--processed_data #预处理完成模型输入
    |    |    |    |--test.json
    |    |    |    |--dev.json
    |    |    |    |--labels.txt
    |    |    |    |--train.json
    |--Event_extraction #事件抽取
    |    |--data_agumentation.py #事件增强
    |    |--sequence_labeling.py #序列标注
    |    |--test.py 
    |    |--utils.py #工具类
    |    |--duee_1_data_prepare.py #数据预处理
    |    |--model.py #模型代码
    |    |--run_role_labeling.sh #论元抽取脚本
    |    |--sequence_labeling_role.py #论元抽取
    |    |--run_trigger_labeling.sh #触发词抽取脚本
    |    |--sequence_labeling_trigger.py #触发词抽取
    |--bert_multi_classification #多标签分类
    |    |--dataset.py #数据集构建
    |    |--model #模型代码
    |    |--logs #输出日志
    |    |--predict_result.py #预测函数
    |    |--config.py #配置文件
    |    |--train.py #模型训练
    |    |--convert_train_dev_format.py #bert模型数据转换函数
    |    |--convert_test_format.py #bert模型数据转换函数
    |    |--data_preprocess.py #数据预处理
    |    |--utils #工具类
    |    |--models.py #模型函数
    |--yolov5 #目标检测
    |    |--detect.py #目标检测
    |    |--data #数据配置文件
    |    |    |--ccks2022.yaml
    |    |--train.py #模型训练
    |    |--predict.py #模型推理
    |    |--models
    |    |    |--common.py
    |    |    |--yolov5x.yaml
    |    |    |--yolov5l.yaml
    |    |    |--experimental.py
    |    |    |--yolo.py 
    |    |--utils #工具包
    |--role_classification #论元实体判别
    |    |--dataset.py #数据集构建
    |    |--utils.py #工具类
    |    |--data #论元实体类别
    |    |    |--train.xlsx
    |    |--train.py #模型训练
    |    |--predict.py #模型预测

```
## 多模态军事装备事件要素抽取

### 评测方法
本次任务采用事件要素抽取的精确率（Precision, P）、召回率（Recall, R）和F1值（F1-measure, F1）来评估事件要素的识别效果。使用事件要素匹配F1作为最终评价指标，匹配过程不区分大小写，其中F1的计算方式如下：

$$ F1=\frac{2*P*R}{P+R}\ $$

其中，

 - P=预测正确事件要素个数/全部预测事件要素数量.
 - R=预测正确事件要素个数/全部正确标注要素数量。
 - 对于一个预测事件要素，判断其为预测正确的标准是：如果它的事件类型、在文本中的位置、事件要素角色以及在图像中的实体位置与正确标注数据中的一个事件要素匹配，则判定其为预测正确要素。其中，与图像中实体位置匹配的标准为预测的图像实体的位置与正确标注图像实体的位置的交并比大于0.5。若图像中没有与该事件要素对应的图像实体，则输出-1判断为正确。

###  模型数据下载（请严格按照对应路径放置）
<a href="https://pan.baidu.com/s/1aTMXCF-cUaWYjzqz1tTyYA">百度网盘链接</a>   提取码：8b8g

- **将压缩包文件中data.zip解压后所有文件放置于CCKS2022_JS/data中**
- **将压缩包文件中ckpt.zip解压后所有文件放置于CCKS2022_JS/ckpt中**

- **CCKS2022_JS/data/raw_data/train 存放训练用官方数据**
- **CCKS2022_JS/data/raw_data/test 存放测试用官方数据**
- **CCKS2022_JS/data/raw_data/dev 存放验证用官方数据**
- **CCKS2022_JS/data/result 存放各步骤中间结果**
###  项目运行主要环境依赖（建议在conda虚拟环境中安装）
运行系统：
```shell
Linux:
Ubuntu 18.04.6 LTS
GPU:
NVIDIA GeForce RTX 3090
```

---

python:

```shell
python3.8
```

运行依赖:
```shell
pip install -r requirements.txt
```
### 方案复现
下面步骤用于指导我们方案的复现过程。由于方案采用pipeline形式实现，我们将大体流程分为六个过程：
- 1. 数据预处理
- 2. 多标签分类模型实现
- 3. 触发词模型实现
- 4. 论元抽取模型实现
- 5. 目标检测模型实现
- 6. 结果后处理

### 快速开始：
以下指令可通过已经加载模型一键快速生成任务目标结果，详细任务可见CCKS2022_JS/run_ee.sh脚本。同时也可以通过分步执行完成模型训练预测过程。最终结果文件存放于**CCKS2022_JS/submit/result_xxx.txt**中，“xxx”代表时间戳。
``` shell
# 一键执行pipeline任务
sh run_ee.sh pipeline
```
### 分步执行：
以下将详细通过各个步骤生成目标结果。
#### Step1：数据预处理并加载

从比赛官网下载数据集，逐层解压存放于data/raw_data目录下，运行以下脚本将原始数据预处理成序列标注格式数据，并进行数据增强。
处理之后的数据放在data/EE1.0下，其中train.json代表训练集（1400条），dev.json代表验证集（200条），test.json代表测试集（400条），数据增强随机替换样本存放于roles.xlsx。触发词识别数据文件存放在data/EE1.0/role下，论元角色识别数据文件存放在data/EE1.0/trigger下。

``` shell
# 数据预处理
sh run_ee.sh data_prepare

# 数据增强
sh run_ee.sh data_augmentation
```


#### Step2：多标签多分类
多标签分类代码存放于CCKS2022_JS/bert_multi_classification中。首先在data_preprocess.py中,有将数据预处理成bert所需要的格式的相关代码。然后在dataset.py中，将处理好的数据制作成torch所需格式的数据集。在models.py中有多标签分类模型建立的相关代码。最后在train.py中执行训练、验证过程，在predict_result.py中执行预测过程。
```shell
# 多标签多分类模型训练
sh run_ee.sh multi_label_train

# 多标签多分类模型预测
sh run_ee.sh multi_label_predict
```
#### Step3：触发词识别

触发词识别通过部分人工标注的训练集触发词数据进而通过模型进行测试集触发词预测，相关数据文件存放于CCKS2022_JS/data/EE1.0中。标注完成的训练集放置于./trigger/train.tsv，验证集放于./trigger/dev.tsv，如需更换训练数据需按./train.json的格式标注训练集触发词，重新执行step1中脚本完成数据预处理，自动生成项目所需数据集格式。执行以下脚本完成模型训练与预测，中间结果存放于CCKS2022_JS/data/result/trigger/test_predict_trigger.json。


```shell
# 触发词识别模型训练
sh run_ee.sh trigger_train

# 触发词识别模型预测
sh run_ee.sh trigger_predict
```

#### Step4：论元抽取

论元抽取部分采用step2与step3中模型最终投票结果构建模型输入，模型输入的训练集放置于CCKS2022_JS/data/EE1.0/role/train.tsv，验证集放于CCKS2022_JS/data/EE1.0/role/dev.tsv，中间结果存放于CCKS2022_JS/data/result/role/test_predict_role.json。

```shell
# 论元抽取模型训练
sh run_ee.sh role_train

# 论元抽取模型预测
sh run_ee.sh role_predict
```

#### Step5：论元目标检测
论元目标检测采用yolov5模型，相关代码文件见CCKS2022_JS/yolov5

##### 数据

``` 
- 设置数据相关配置文件：./CCKS2022_JS/yolov5/data/ccks2022.yaml
  
  - path: #根目录
  
  - train: #训练数据存放路径
  
  - val: #验证数据存放路径
```


##### 训练
输入以下指令
```shell
# yolov5模型训练
sh run_ee.sh objDec_train
```
会出现以下提示信息：
``` shell
ROOTPATH/yolov5
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: (30 second timeout)
```
此时键盘输入3，点击回车继续执行。

##### 预测

---

```shell
# yolov5模型预测
sh run_ee.sh objDec_predict
```


#### Step6：数据后处理，提交结果

将step1-5中所有结果文件融合，按照比赛预测指定格式构建结果文件。

```shell
# 后处理
sh run_ee.sh pred_2_submit
```

最终结果文件存放于**CCKS2022_JS/submit/result_xxx.txt**中，“xxx”代表时间戳。

