# NLP009_GPT-2-LoRA
NLP009: gpt-2-lora using pytorch

## 使用说明
### 要求
> Python == 3.8 \
> PyTorch == 2.3.1  
### 预训练模型
```shell script
cd pretrained_checkpoints
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin
```
### 数据编码
```shell script
python data_encode.py
```
### 训练
```shell script
python train.py
```
### 已训练模型
[model.26289.pt(提取码8888)](https://pan.baidu.com/s/1UbOvEVw6g7McBHDQ76k8oQ)
### 测试
```shell script
python predict.py  
```
```
Input:
name : Alimentum | area : city centre | family friendly : no | near : Burger King

Output:
The Alimentum is not family - friendly but located in the city centre . It 's near Burger King .
```
## 参考
https://github.com/samylee/NLP008_GPT-2   
https://github.com/microsoft/lora  
https://blog.csdn.net/samylee  
