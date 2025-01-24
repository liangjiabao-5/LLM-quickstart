# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import json
import torch
from typing import Union, Dict, Any

from transformers import AutoTokenizer, AutoModel, AutoConfig, set_seed

from modelscope.models.base import TorchModel
from modelscope.preprocessors.base import Preprocessor
from modelscope.pipelines.base import Model, Pipeline
from modelscope.utils.config import Config
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.models.builder import MODELS

from rex.data_utils import data_loader, token_config
from rex.arguments import get_args, DataArguments, UIEArguments
from rex.model.model import RexModel
from rex.Trainer.trainer import RexModelTrainer
from rex.Trainer.utils import compute_metrics

@PIPELINES.register_module('rex-uninlu', module_name='nlp_deberta_rex-uninlu_chinese-base-pipe')
class RexUniNLUPipeline(Pipeline):

    def __init__(self, model, preprocessor=None, **kwargs):
        super().__init__(model=model, auto_collate=False)
        self.model_dir = model
        self.model, self.trainer = self.init_model(**kwargs)
    
    def init_model(self, **kwargs):
        data_args, training_args, model_args = get_args()
        training_args.bert_model_dir = self.model_dir
        training_args.load_checkpoint = self.model_dir
        # training_args.fp16 = False
        training_args.no_cuda = True
        tokenizer = AutoTokenizer.from_pretrained(training_args.bert_model_dir)
        tokenizer.add_special_tokens({
            "additional_special_tokens": [token_config.PREFIX_TOKEN, token_config.TYPE_TOKEN, token_config.CLASSIFY_TOKEN, token_config.MULTI_CLASSIFY_TOKEN]
        })
        config = AutoConfig.from_pretrained(training_args.bert_model_dir)

        model = RexModel(config, training_args, model_args)
        if training_args.no_cuda:
            model.load_state_dict(torch.load(os.path.join(training_args.load_checkpoint, 'pytorch_model.bin'), map_location=torch.device('cpu')), strict=False)
        else:
            model.load_state_dict(torch.load(os.path.join(training_args.load_checkpoint, 'pytorch_model.bin')), strict=False)


        uie_token_data_loader = data_loader.UIEDataLoader(
            data_args, 
            tokenizer, 
            data_args.data_path, 
            training_args.local_rank, 
            training_args.world_size,
            training_args.no_cuda)
        
        trainer = RexModelTrainer(model, training_args, uie_token_data_loader.get_collate_fn(),
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        trainer.rex_dl = uie_token_data_loader
        trainer.data_args = data_args
        return model, trainer

    def forward(self, input, **forward_params):
        """ Provide default implementation using self.model and user can reimplement it
        """
        print(input)
        text = input
        print(forward_params)
        schema = forward_params.pop('schema')
        if type(schema) == str:
            schema = json.loads(schema)

        input_dict = {
            'text': text,
            'schema': schema
        }
        pred_info_list = self.trainer.prediction_step(self.model, input_dict, prediction_loss_only=False, do_pred=True)
        return {'output': pred_info_list}

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs

    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input

    def _sanitize_parameters(self, **pipeline_parameters):
        return {},pipeline_parameters,{}


# Tips: usr_config_path is the temporary save configuration location， after upload modelscope hub, it is the model_id
# usr_config_path = '/mnt/workspace/learn/companyName/nlp_deberta_rex-uninlu_chinese-base'
# config = Config({
#     "framework": 'pytorch',
#     "task": 'rex-uninlu',
#     "pipeline": {"type": "nlp_deberta_rex-uninlu_chinese-base-pipe"},
#     "allow_remote": True
# })
# config.dump('/mnt/workspace/learn/companyName/nlp_deberta_rex-uninlu_chinese-base' + 'configuration.json')

if __name__ == "__main__":
    from modelscope.models import Model
    from modelscope.pipelines import pipeline
    # model = Model.from_pretrained(usr_config_path)
    text = "北大西洋议会春季会议26日在西班牙巴塞罗那闭幕。"
    inference = pipeline('rex-uninlu', model='/mnt/workspace/nlp_deberta_rex-uninlu_chinese-base')
    # output = inference(text, schema={"人物": None, "地理位置": None, "组织机构": None})
    # print(output)
    
    output = inference(
    input='大唐华银(湖南)新能源有限公司将业务系统划分为生产控制大区和管理信息大区，生产控制大区进一步划分为安全区Ⅰ和安全区Ⅱ，', 
    schema={
        '系统名称': None,
        '公司名称': None,
        '地址': None
    }
) 
    print(output)
    
    print("---------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------")

# 命名实体识别 {实体类型: None}
#     output = inference(
#     input='湖南新能源集控系统采用冗余的网络结构设计，Ⅰ区核心交换机等关键网络设备以及应用服务器、数据库服务器等关键服务器为双机冗余部署，保证了系统的稳定性和高可用性；系统设备均使用国产品牌，确保了系统的可靠性和安全性。系统通过本地方式对网络设备、安全设备进行管理，运维人员仅能在机房内对网络设备、安全设备进行本地登录管理；系统通过远程方式对主机设备进行管理，运维人员通过 SSH 协议对主机设备进行远程登录管理，运维人员每月进行一次增量备份，备份数据存储在存储介质中；湖南新能源集控系统未建立数据异地备份中心，不能利用通信网络将业务数据的进行异地实时备份。', 
#     schema={
#         '系统名称': None,
#         '公司名称': None,
#         '地址': None
#     }
# ) 
#     print(output)
    
#     print("---------------------------------------------------------------------------------")
#     print("---------------------------------------------------------------------------------")

# # 命名实体识别 {实体类型: None}
#     output = inference(
#     input='系统生产控制大区部署了锐捷 RG-IDP-2000E IPS 装置、深信服 NTA-100 B620 APT 威胁检测系统，能够对关键网络节点的入侵行为进行实时监测和限制；部署了深信服 SIP-Logger 日志审计系统、网络安全监测装置，能够收集系统内部分设备日志并进行综合分析。', 
#     schema={
#         '系统名称': None,
#         '公司名称': None,
#         '地址': None
#     }
# ) 
#     print(output)
