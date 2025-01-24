# 查询对应文档的指标
select j.* from db_standard_tree e,
(select c.id from db_standard_tree c,
(select a.id from db_standard_tree a, 
(select id from db_standard_tree
where `name` like '安全通信网络%'
and id in (2,3,11,12,874,14,15,16,17,880)) b
where a.pid = b.id
and a.`name` like '通用%') d
where d.id = c.pid) f,
db_standard_details j
where f.id = e.pid
and e.`name` = '3'
and j.nid = e.id


#查询1000条数据
SELECT * FROM `report_db2_content_results_info`
where security_class = '安全物理环境'
and detail_id in (808,809,811,812,813,814,817,818,829,830,831,832,833,834,838,839,840,841,842,843,847,848)
order by created_time DESC
limit 1000


#查询50条不符合的数据
SELECT * FROM `report_db2_content_results_info`
where security_class = '安全物理环境'
and result_score = '不符合'
and detail_id in (808,809,811,812,813,814,817,818,829,830,831,832,833,834,838,839,840,841,842,843,847,848)
order by created_time DESC
limit 50


SELECT * FROM `report_db2_content_results_info`
where security_class = '安全运维管理'
and detail_id in (1073,1074,1075,1076,1077,1078,1079,1080,1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126)
order by created_time DESC
limit 10000


SELECT * FROM `report_db2_content_results_info`
where security_class = '安全通信网络'
and detail_id in (858,859,860,861,862,867,868,870)
order by created_time DESC
limit 5000


SELECT * FROM `report_db2_content_results_info`
where security_class = '安全区域边界'
and detail_id in (874,875,876,877,885,886,887,888,889,896,897,898,899,905,906,910,911,912,913,918)
order by created_time DESC
limit 5000


SELECT * FROM `report_db2_content_results_info`
where security_class = '安全建设管理'
and detail_id in (991,992,993,994,1011,1012,1013,1014,1015,1016,1023,1024,1025,1026,1027,1028,1029,1036,1037,1038,1039,1040,1041,1046,1047,1049,1050,1051,1052,1053,1054,1066,1067,1068)
order by created_time DESC
limit 5000


SELECT * FROM `report_db2_content_results_info`
where security_class = '安全管理中心'
and detail_id in (927,928,929,930,934,935,948,949,950,951,952,953)
order by created_time DESC
limit 10000


SELECT * FROM `report_db2_content_results_info`
where security_class = '安全管理制度'
and detail_id in (954,955,956,957,958,959,960)
order by created_time DESC
limit 5000