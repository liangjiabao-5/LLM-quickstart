import os
import yaml

# 定义了一个名为tool_config_from_file的函数，它接受两个参数：tool_name（工具名称，用于匹配文件名），directory（目录路径，默认为"Tool/"）。
def tool_config_from_file(tool_name, directory="Tool/"):
    """搜索工具的YAML配置文件，并以JSON格式返回。"""
    for filename in os.listdir(directory):
        if filename.endswith('.yaml') and tool_name in filename:
            file_path = os.path.join(directory, filename)
            with open(file_path, encoding='utf-8') as f:
                # 使用yaml.safe_load()函数从打开的文件中读取YAML内容，并将其解析为Python的数据结构（如字典或列表）。然后返回这个数据结构。
                return yaml.safe_load(f)
    return None
