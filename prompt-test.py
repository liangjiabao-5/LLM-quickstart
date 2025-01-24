import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ['TRANSFORMERS_OFFLINE'] = '1' # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
MODEL_PATH = "/mnt/workspace/glm4/model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

messages = [
    {
        "role": "system",
        "content": f"""
        按照以下预期结果的几点规则，进行充分的思考和推理，判断user输入的content是否符合:
        请注意,以下几个注意点优先级比预期结果优先级更高：
        1.在评估时对具体对象进行比对，识别和理解它们所属的类别或具有的属性一致则为符合。
        2.预期结果中包含互斥情况。当user输入的content明确支持预期结果中的某一情况时，模型应专注于评估该情况，而无需对其他互斥情况进行判断。
        3.只要语义相同即可，进行模糊匹配。
        4.对预期结果的指令可以不进行描述与判断。
        5.当用户通过描述，无需进行某项指标判断时，则该项预期结果为符合。
        6.服务器、终端开启审计功能就算符合。

        预期结果：
        1）执行命令"systemctl status rsyslog"或“service syslog status”，核查系统日志是否正常运行；
             执行命令“systemctl status auditd”或“service auditd status”，核查日志审计功能是否正常运行；
             执行命令“systemctl list-unit-files|grep enabled|grep rsyslog”和“systemctl list-unit-files|grep enabled|grep auditd”，核查rsyslog和auditd是否加入开启启动项；
             执行命令“uptime” ，核查系统时钟是否正确。
        2）核查安全审计功能是否覆盖到系统的所有用户；
        3）执行命令“more /etc/audit/audit.rules”或“auditctl -l”，核查是否对重要用户行为和重要安全事件进行审计。
        或通过访谈和核查，是否采用第三方审计系统，对所有用户行为进行审计，登录第三方审计系统，审计内容是否覆盖重要的用户行为和重要安全事件。
        4）LINUX主机和数据库日志和审计是分开的，如果只开启了日志，未开启审计，算部分符合，中风险。
        5）日志记录不全、有审计数据但无法直观展示等情况，判中风险。
        6）关键设备（资产重要程度为非常重要的设备）无任何审计措施，或未开启任何审计功能，且未采用堡垒机、审计设备等措施，判不符合，高风险。

        最后判断，如果每条预期结果都符合则输出"结果：符合"，部分预期结果符合输出"结果：部分符合"，每条预期结果都不符合则输出"结果：不符合"。
        """,
    },
    {
        "role": "user",
        "content": '''1）经核查，操作系统开启了安全审计功能。
2）经核查，安全审计功能能够记录用户的登录/登出，配置变更，用户添加，权限配置等，实现对重要的用户行为和重要安全事件进行审计。'''
    }
]

inputs = tokenizer.apply_chat_template(messages,
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))