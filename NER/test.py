from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline('rex-uninlu', model='/mnt/workspace/nlp_deberta_rex-uninlu_chinese-base', model_revision='v1.2.1')

# # 命名实体识别 {实体类型: None}
# output = semantic_cls(
#     input='1944年毕业于北大的名古屋铁道会长谷口清太郎等人在日本积极筹资，共筹款2.7亿日元，参加捐款的日本企业有69家。', 
#     schema={
#         '人物': None,
#         '地理位置': None,
#         '组织机构': None
#     }
# ) 
# print(output)



# 命名实体识别 {实体类型: None}
output = semantic_cls(
    input='大唐华银(湖南)新能源有限公司将业务系统划分为生产控制大区和管理信息大区，生产控制大区进一步划分为安全区Ⅰ和安全区Ⅱ，湖南新能源集控系统部署在生产控制大区安全区Ⅰ；安全区Ⅰ与安全区Ⅱ之间无数据交互物理隔离，安全区Ⅰ与安全区Ⅲ之间通过部署南瑞 Syskeeper-2000 正向隔离装置进行强逻辑隔离，安全区Ⅰ与安全接入区之间通过部署南瑞 Syskeeper-2000 正、反向隔离装置进行强逻辑隔离；纵向与电力专网、电信专网网络边界处部署有南瑞 Netkeeper-2000 电力行业专用的纵向加密认证装置，实现远方数据传输的数据加密、身份认证和访问控制。', 
    schema={
        '系统名称': None,
        '公司名称': None,
        '证书名称': None,
        '备案单号': None,
        '地址': None
    }
) 
print(output)

# 命名实体识别 {实体类型: None}
output = semantic_cls(
    input='湖南新能源集控系统采用冗余的网络结构设计，Ⅰ区核心交换机等关键网络设备以及应用服务器、数据库服务器等关键服务器为双机冗余部署，保证了系统的稳定性和高可用性；系统设备均使用国产品牌，确保了系统的可靠性和安全性。系统通过本地方式对网络设备、安全设备进行管理，运维人员仅能在机房内对网络设备、安全设备进行本地登录管理；系统通过远程方式对主机设备进行管理，运维人员通过 SSH 协议对主机设备进行远程登录管理，运维人员每月进行一次增量备份，备份数据存储在存储介质中；湖南新能源集控系统未建立数据异地备份中心，不能利用通信网络将业务数据的进行异地实时备份。', 
    schema={
        '系统名称': None,
        '公司名称': None,
        '证书名称': None,
        '备案单号': None,
        '地址': None
    }
) 
print(output)

# 命名实体识别 {实体类型: None}
output = semantic_cls(
    input='系统生产控制大区部署了锐捷 RG-IDP-2000E IPS 装置、深信服 NTA-100 B620 APT 威胁检测系统，能够对关键网络节点的入侵行为进行实时监测和限制；部署了深信服 SIP-Logger 日志审计系统、网络安全监测装置，能够收集系统内部分设备日志并进行综合分析。', 
    schema={
        '系统名称': None,
        '公司名称': None,
        '证书名称': None,
        '备案单号': None,
        '地址': None
    }
) 
print(output)
