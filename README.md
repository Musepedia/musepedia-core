# MGS-core
### QA-RS-Based-Museum-Guide-System
- 本仓库包含项目的Question Answering和Recommender System算法，不包含业务代码（数据分析模块除外）

### Requirements
- Python >= 3
- Pytorch >= 1.4
- transformers (latest)

### Usage
所有模型都需要离线运行，在QADemo/models/下存放对应的模型文件（每个模型文件包含`pytorch_model.bin`, `config.json`和`vocab.txt`文件）
```shell
cd QADemo
mkdir models
```

jsonHandler.py是基于SQuAD数据库标准生成json文件的函数，若json格式有变则需修改。

使用时，建议在utils/csv/和utils/json/下存放待处理的数据文件。

**新增了生成label和text的csv的功能**

**注意**：生成的json文件并不能直接使用，需要在前加上"data"标签！

f1_evaluator是根据模型得出的结果与人工标注的结果计算得到的评估分数。

### Notices

- 请在每个Python文件的首行加以注释 `# -*- coding: UTF-8 -*-`，部分Pipeline可能会因编码问题报错。
- 请在fork代码之后提交PR，请不要直接向仓库中push代码，尤其是master分支上
- 项目排期和具体的工作安排将在 https://trello.com/ 上显示

### Update Logs
- Demo.py 支持多问题在多文本中搜索答案

### Bug Report
- Demo中的Pipeline可能会截取部分答案内容