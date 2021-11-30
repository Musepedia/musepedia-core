# MGS-core
### QA-RS-Based-Museum-Guide-System
- 本仓库包含项目的Question Answering和Recommender System算法，不包含业务代码（数据分析模块除外）
- **Version 1.0.4**

### Requirements
- Python >= 3
- PyTorch >= 1.5.1
- Hugging Face transformers >= 4.10.3
- datasets >= 1.14.0

### Usage
根据`requirements.txt`要求配置依赖
```shell
cd MGS-core
sudo pip install -r requirements.txt
```

所有模型都需要离线运行，在src/qa/models/下存放对应的模型文件（每个模型文件包含`pytorch_model.bin`, `config.json`和`vocab.txt`文件）
```shell
cd src/qa
mkdir models
```

启动grpc服务
```shell
cd MGS-core
python server/Server.py
```

### Notices
- 请在每个Python文件的首行加以注释 `# -*- coding: UTF-8 -*-`，部分Pipeline可能会因编码问题报错。
- 在引入非同级目录文件时需要在`import`语句前加入`sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))`
- 如需测试代码请将测试内容包裹在`if __name__ == '__main__':`内
- 请开分支后提交PR，请不要直接向master分支push代码
- 项目排期和具体的工作安排将在 https://trello.com/ 上显示

### Update Logs
- 1.0.x版本将对项目文件重新整理，common文件夹放工程需要的程序，server放grpc相关程序，src放QA和RS算法相关程序，utils放工具类
- 1.0.3 修复了问题回答缓慢的bug，现在模型加载在grpc服务启动前完成
- 1.0.4 增加异常处理

### Bug Report
- 内存泄漏问题可能仍然存在（推测在`server()`中存在未被Python解释器回收的垃圾）