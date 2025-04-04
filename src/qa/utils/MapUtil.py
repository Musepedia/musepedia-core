# -*- coding: UTF-8 -*-
import jieba
import hashlib

from pyecharts.charts import Map
from pyecharts import options as opts
from pyecharts.render import make_snapshot
from snapshot_phantomjs import snapshot

from src.common.exception.ExceptionHandler import catch
from Config import MAP_FIGURE_PATH, MAP_URL


class MapUtil:
    """
    地图生成工具，用于根据问题对应的答案，生成包含国家分布的世界地图，或包含省份分布的中国地图
    后续版本中，将重构至src.utils.NLPUtil的NLPUtil类下
    """

    THEME_COLOR = '#FF6839'

    def __init__(self):
        self._country_dict, self._provinces = self.preload()
        self._country_list = self._country_dict.values()

    @staticmethod
    def preload():
        """
        加载分词所用的国家或地区中英对照词典，以及中国省份列表

        :return: 国家或地区中英对照词典，中国省份列表构成的元组
        """
        country_dict = {
            "Somalia": "索马里",
            "Liechtenstein": "列支敦士登",
            "Morocco": "摩洛哥",
            "W. Sahara": "西撒哈拉",
            "Serbia": "塞尔维亚",
            "Afghanistan": "阿富汗",
            "Angola": "安哥拉",
            "Albania": "阿尔巴尼亚",
            "Andorra": "安道尔共和国",
            "United Arab Emirates": "阿拉伯联合酋长国",
            "Argentina": "阿根廷",
            "Armenia": "亚美尼亚",
            "Australia": "澳大利亚",
            "Austria": "奥地利",
            "Azerbaijan": "阿塞拜疆",
            "Burundi": "布隆迪",
            "Belgium": "比利时",
            "Benin": "贝宁",
            "Burkina Faso": "布基纳法索",
            "Bangladesh": "孟加拉国",
            "Bulgaria": "保加利亚",
            "Bahrain": "巴林",
            "Bahamas": "巴哈马",
            "Bosnia and Herz.": "波斯尼亚和黑塞哥维那",
            "Belarus": "白俄罗斯",
            "Belize": "伯利兹",
            "Bermuda": "百慕大",
            "Bolivia": "玻利维亚",
            "Brazil": "巴西",
            "Barbados": "巴巴多斯",
            "Brunei": "文莱",
            "Bhutan": "不丹",
            "Botswana": "博茨瓦纳",
            "Central African Rep.": "中非",
            "Canada": "加拿大",
            "Switzerland": "瑞士",
            "Chile": "智利",
            "China": "中国",
            "Côte d'Ivoire": "科特迪瓦",
            "Cameroon": "喀麦隆",
            "Dem. Rep. Congo": "刚果民主共和国",
            "Congo": "刚果",
            "Colombia": "哥伦比亚",
            "Cape Verde": "佛得角",
            "Costa Rica": "哥斯达黎加",
            "Cuba": "古巴",
            "N. Cyprus": "北塞浦路斯",
            "Cyprus": "塞浦路斯",
            "Czech Rep.": "捷克",
            "Germany": "德国",
            "Djibouti": "吉布提",
            "Denmark": "丹麦",
            "Dominican Rep.": "多米尼加",
            "Algeria": "阿尔及利亚",
            "Ecuador": "厄瓜多尔",
            "Egypt": "埃及",
            "Eritrea": "厄立特里亚",
            "Spain": "西班牙",
            "Estonia": "爱沙尼亚",
            "Ethiopia": "埃塞俄比亚",
            "Finland": "芬兰",
            "Fiji": "斐济",
            "France": "法国",
            "Gabon": "加蓬",
            "United Kingdom": "英国",
            "Georgia": "格鲁吉亚",
            "Ghana": "加纳",
            "Guinea": "几内亚",
            "Gambia": "冈比亚",
            "Guinea-Bissau": "几内亚比绍",
            "Eq. Guinea": "赤道几内亚",
            "Greece": "希腊",
            "Grenada": "格林纳达",
            "Greenland": "格陵兰",
            "Guatemala": "危地马拉",
            "Guam": "关岛",
            "Guyana": "圭亚那",
            "Honduras": "洪都拉斯",
            "Croatia": "克罗地亚",
            "Haiti": "海地",
            "Hungary": "匈牙利",
            "Indonesia": "印度尼西亚",
            "India": "印度",
            "Br. Indian Ocean Ter.": "英属印度洋领土",
            "Ireland": "爱尔兰",
            "Iran": "伊朗",
            "Iraq": "伊拉克",
            "Iceland": "冰岛",
            "Israel": "以色列",
            "Italy": "意大利",
            "Jamaica": "牙买加",
            "Jordan": "约旦",
            "Japan": "日本",
            "Siachen Glacier": "锡亚琴冰川",
            "Kazakhstan": "哈萨克斯坦",
            "Kenya": "肯尼亚",
            "Kyrgyzstan": "吉尔吉斯坦",
            "Cambodia": "柬埔寨",
            "Korea": "韩国",
            "Kuwait": "科威特",
            "Lao PDR": "老挝",
            "Lebanon": "黎巴嫩",
            "Liberia": "利比里亚",
            "Libya": "利比亚",
            "Sri Lanka": "斯里兰卡",
            "Lesotho": "莱索托",
            "Lithuania": "立陶宛",
            "Luxembourg": "卢森堡",
            "Latvia": "拉脱维亚",
            "Moldova": "摩尔多瓦",
            "Madagascar": "马达加斯加",
            "Mexico": "墨西哥",
            "Macedonia": "马其顿",
            "Mali": "马里",
            "Malta": "马耳他",
            "Myanmar": "缅甸",
            "Montenegro": "黑山",
            "Mongolia": "蒙古",
            "Mozambique": "莫桑比克",
            "Mauritania": "毛里塔尼亚",
            "Mauritius": "毛里求斯",
            "Malawi": "马拉维",
            "Malaysia": "马来西亚",
            "Namibia": "纳米比亚",
            "New Caledonia": "新喀里多尼亚",
            "Niger": "尼日尔",
            "Nigeria": "尼日利亚",
            "Nicaragua": "尼加拉瓜",
            "Netherlands": "荷兰",
            "Norway": "挪威",
            "Nepal": "尼泊尔",
            "New Zealand": "新西兰",
            "Oman": "阿曼",
            "Pakistan": "巴基斯坦",
            "Panama": "巴拿马",
            "Peru": "秘鲁",
            "Philippines": "菲律宾",
            "Papua New Guinea": "巴布亚新几内亚",
            "Poland": "波兰",
            "Puerto Rico": "波多黎各",
            "Dem. Rep. Korea": "朝鲜",
            "Portugal": "葡萄牙",
            "Paraguay": "巴拉圭",
            "Palestine": "巴勒斯坦",
            "Qatar": "卡塔尔",
            "Romania": "罗马尼亚",
            "Russia": "俄罗斯",
            "Rwanda": "卢旺达",
            "Saudi Arabia": "沙特阿拉伯",
            "Sudan": "苏丹",
            "S. Sudan": "南苏丹",
            "Senegal": "塞内加尔",
            "Singapore": "新加坡",
            "Solomon Is.": "所罗门群岛",
            "Sierra Leone": "塞拉利昂",
            "El Salvador": "萨尔瓦多",
            "Suriname": "苏里南",
            "Slovakia": "斯洛伐克",
            "Slovenia": "斯洛文尼亚",
            "Sweden": "瑞典",
            "Swaziland": "斯威士兰",
            "Seychelles": "塞舌尔",
            "Syria": "叙利亚",
            "Chad": "乍得",
            "Togo": "多哥",
            "Thailand": "泰国",
            "Tajikistan": "塔吉克斯坦",
            "Turkmenistan": "土库曼斯坦",
            "Timor-Leste": "东帝汶",
            "Tonga": "汤加",
            "Trinidad and Tobago": "特立尼达和多巴哥",
            "Tunisia": "突尼斯",
            "Turkey": "土耳其",
            "Tanzania": "坦桑尼亚",
            "Uganda": "乌干达",
            "Ukraine": "乌克兰",
            "Uruguay": "乌拉圭",
            "United States": "美国",
            "Uzbekistan": "乌兹别克斯坦",
            "Venezuela": "委内瑞拉",
            "Vietnam": "越南",
            "Vanuatu": "瓦努阿图",
            "Yemen": "也门",
            "South Africa": "南非",
            "Zambia": "赞比亚",
            "Zimbabwe": "津巴布韦",
            "Aland": "奥兰群岛",
            "American Samoa": "美属萨摩亚",
            "Fr. S. Antarctic Lands": "南极洲",
            "Antigua and Barb.": "安提瓜和巴布达",
            "Comoros": "科摩罗",
            "Curaçao": "库拉索岛",
            "Cayman Is.": "开曼群岛",
            "Dominica": "多米尼加",
            "Falkland Is.": "马尔维纳斯群岛（福克兰）",
            "Faeroe Is.": "法罗群岛",
            "Micronesia": "密克罗尼西亚",
            "Heard I. and McDonald Is.": "赫德岛和麦克唐纳群岛",
            "Isle of Man": "曼岛",
            "Jersey": "泽西岛",
            "Kiribati": "基里巴斯",
            "Saint Lucia": "圣卢西亚",
            "N. Mariana Is.": "北马里亚纳群岛",
            "Montserrat": "蒙特塞拉特",
            "Niue": "纽埃",
            "Palau": "帕劳",
            "Fr. Polynesia": "法属波利尼西亚",
            "S. Geo. and S. Sandw. Is.": "南乔治亚岛和南桑威奇群岛",
            "Saint Helena": "圣赫勒拿",
            "St. Pierre and Miquelon": "圣皮埃尔和密克隆群岛",
            "São Tomé and Principe": "圣多美和普林西比",
            "Turks and Caicos Is.": "特克斯和凯科斯群岛",
            "St. Vin. and Gren.": "圣文森特和格林纳丁斯",
            "U.S. Virgin Is.": "美属维尔京群岛",
            "Samoa": "萨摩亚"
        }
        provinces = ['北京', '上海', '重庆', '天津', '内蒙古', '广西', '宁夏', '新疆', '西藏', '香港', '澳门', '黑龙江', '吉林', '辽宁',
                     '河北', '河南', '山西', '江苏', '浙江', '安徽', '福建', '江西', '山东', '湖北', '湖南', '广东', '海南', '四川', '贵州',
                     '云南', '陕西', '甘肃', '青海', '台湾']

        return country_dict, provinces

    @catch(Exception)
    def render_map(self, answer: str) -> str:
        """
        在答案中找出可能包含的国家或地区，生成地区分布图，
        如果包含多个国家或地区名称，则生成世界地图；如果包含多个省份名称，则生成中国地图。
        按照URL规则，返回生成地图的URL

        :param answer: 根据问题抽取出的答案
        :return: 生成地图的URL
        """

        for word in self._provinces:
            jieba.add_word(word)
        for word in self._country_list:
            jieba.add_word(word)

        words = jieba.lcut(answer)
        # print(words)
        country = []
        province = []
        flag = 'world'
        for word in words:
            if word in self._provinces:
                flag = 'china'
                province.append((word, 1))
            if word in self._country_list:
                country.append((word, 1))
        # print(data)

        if flag == 'china':
            data = province
        else:
            data = country

        # print(data)
        map = (
            Map()
                .add("", data, flag, is_map_symbol_show=False, name_map=self._country_dict)
                .set_global_opts(
                visualmap_opts=opts.VisualMapOpts(is_show=False, max_=2, range_color=[self.THEME_COLOR]),
            )
                .set_series_opts(
                label_opts=opts.LabelOpts(is_show=False)
            )
        )

        filename = str(hashlib.md5(answer.encode('utf-8')).hexdigest()) + '.png'
        filepath = MAP_FIGURE_PATH + filename
        make_snapshot(snapshot, map.render(), filepath)
        url = MAP_URL + filename
        return url


if __name__ == "__main__":
    data = "在中国、俄罗斯、日本、美国、德国等地均有分布.在我国福建、山东、上海、内蒙古等地均有分布"
    map_util = MapUtil()
    url = map_util.render_map(data)
    print(url)
