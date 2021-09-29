# MGS-core
### QA-RS-Based-Museum-Guide-System
- 本仓库包含项目的Question Answering和Recommender System算法，不包含业务代码（数据分析模块除外）

### Requirements
- Python >= 3
- Pytorch >= 1.4
- transformers (latest)

### Notices
- 请在每个Python文件的首行加以注释 `# -*- coding: UTF-8 -*-`，部分Pipeline可能会因编码问题报错。
- 请在fork代码之后提交PR，请不要直接向仓库中push代码，尤其是master分支上
- 项目排期和具体的工作安排将在 https://trello.com/ 上显示

### Update Logs
- FineTuning.py 支持多问题在单个文本中搜索答案
- Demo.py 支持多问题在多文本中搜索答案