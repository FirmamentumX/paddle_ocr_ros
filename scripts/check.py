
# 临时诊断脚本
import paddle.inference as paddle_infer
import os
model_dir = "../models/en_PP-OCRv4_rec_infer"
config = paddle_infer.Config(
    os.path.join(model_dir, "inference.pdmodel"),
    os.path.join(model_dir, "inference.pdiparams")
)
predictor = paddle_infer.create_predictor(config)
print("模型输入名称:", predictor.get_input_names())
# 获取输出层信息
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
print("输出层维度:", output_handle.shape())