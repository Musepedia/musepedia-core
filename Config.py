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
