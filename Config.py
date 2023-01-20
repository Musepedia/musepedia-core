# grpc端口
GRPC_PORT = '5555'

# 地图存储文件夹名称及其绝对路径
MAP_FIGURE_DIRECTORY_NAME = 'map'
MAP_FIGURE_PATH = '/data/static/{0}'.format(MAP_FIGURE_DIRECTORY_NAME, MAP_FIGURE_DIRECTORY_NAME)

# 域名，用于定位生成的地图
DOMAIN_NAME = 'musepedia.cn'
SUB_DOMAIN_NAME = 'static'

# 定位地图的URL
MAP_URL = 'https://{0}.{1}/{2}/'.format(SUB_DOMAIN_NAME, DOMAIN_NAME, MAP_FIGURE_DIRECTORY_NAME)

# RoBERTa模型的路径
ROBERTA_MODEL_PATH = 'src/qa/models/roberta-base-chinese-extractive-qa'

# DPR模型（基于Facebook）的路径
FACEBOOK_DPR_MODEL_PATH = 'src/qa/models/dpr-reader-multiset-base'

# DPR模型（基于RocketQA）的路径
ROCKETQA_MODEL_PATH = 'src/qa/models/zh_dureader_de/config.json'

# 是否使用GPU (CUDA)运行深度学习模型，False表示使用CPU，True表示在有GPU可用的情况下使用
USE_GPU = True
