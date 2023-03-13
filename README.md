# MGS-core
- 本仓库包含项目的Question Answering和Recommender System算法，不包含业务代码（数据分析模块除外）
- **Version 1.2.0**

### Usage
根据`requirements.txt`要求配置依赖（请根据本地CUDA版本安装合适的PyTorch v1.5.1，参考https://pytorch.org/get-started/previous-versions/）
```shell
cd musepedia-core
sudo pip install -r requirements.txt
```

使用grpc工具，根据proto文件定义，生成对应的Python文件
```shell
cd src/rpc/proto
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. src/rpc/proto/*.proto 
```

所有模型都需要离线运行，在src/qa/models/下存放对应的模型文件（每个模型文件包含`pytorch_model.bin`, `config.json`和`vocab.txt`文件）
```shell
cd src/qa
mkdir models
```

启动grpc服务，提供6个命令行参数(`--qa`表示启动QA服务，`--no-qa`表示不启动QA服务，`--gpt`, `--no-gpt`, `--spider`和`--no-spider`同理)
```shell
python -m src.rpc.server --qa --gpt
```

运行或测试单个模块
```shell
python -m [module_name]
```

测试多个模块或经grpc调用模块（Pytest框架），在`test_*.py`文件中编写client与测试用例，先启动server服务，再运行测试样例
```shell
python -m src.rpc.server
pytest test/[test_*.py]
```

### Notices
- 请在每个Python文件的首行加以注释 `# -*- coding: UTF-8 -*-`，部分Pipeline可能会因编码问题报错。
- 从v1.1起，项目结构发生变动，现在可以直接引入非同级目录的包或模块，如果需要运行单个模块，使用`python -m [module_name]`
- 如需测试代码请将测试内容包裹在`if __name__ == '__main__':`内
- 请开分支后提交PR，请不要直接向master分支push代码
- 命名规范参考[PEP 8 — the Style Guide for Python Code](https://pep8.org)