import os

# grpc端口
GRPC_PORT = os.getenv('GRPC_PORT', '5555')

# 地图存储文件夹名称及其绝对路径
MAP_FIGURE_DIRECTORY_NAME = os.getenv('MAP_FIGURE_DIRECTORY_NAME', 'map')
MAP_FIGURE_PATH = os.getenv('MAP_FIGURE_PATH',
                            '/data/static/{0}'.format(MAP_FIGURE_DIRECTORY_NAME, MAP_FIGURE_DIRECTORY_NAME))

# 域名，用于定位生成的地图
DOMAIN_NAME = os.getenv('DOMAIN_NAME', 'musepedia.cn')
SUB_DOMAIN_NAME = os.getenv('SUB_DOMAIN_NAME', 'static')

# 定位地图的URL
MAP_URL = os.getenv('MAP_URL',
                    'https://{0}.{1}/{2}/'.format(SUB_DOMAIN_NAME, DOMAIN_NAME, MAP_FIGURE_DIRECTORY_NAME))

# RoBERTa模型的路径
ROBERTA_MODEL_PATH = os.getenv('ROBERTA_MODEL_PATH', 'src/qa/models/roberta-base-chinese-extractive-qa')

# DPR模型（基于Facebook）的路径
FACEBOOK_DPR_MODEL_PATH = os.getenv('FACEBOOK_DPR_MODEL_PATH', 'src/qa/models/dpr-reader-multiset-base')

# DPR模型（基于RocketQA）的路径
ROCKETQA_MODEL_PATH = os.getenv('ROCKETQA_MODEL_PATH', 'src/qa/models/zh_dureader_de/config.json')

# 是否使用GPU (CUDA)运行深度学习模型，False表示使用CPU，True表示在有GPU可用的情况下使用
USE_GPU = os.getenv('USE_GPU', True)

# es端口
ELASTIC_SEARCH_HOST = os.getenv('ELASTIC_SEARCH_HOST', 'http://pt.musepedia.cn:9200')
ELASTIC_SEARCH_USERNAME = os.getenv('ELASTIC_SEARCH_USERNAME', 'elastic')
ELASTIC_SEARCH_PASSWORD = os.getenv('ELASTIC_SEARCH_PASSWORD', 'r2tiq2FqAd5')

# proxy代理
PROXY = os.getenv('PROXY', 'http://34.82.217.181:5555')
