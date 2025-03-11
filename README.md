





# Code Implement Instrcution 



# Acknowledgment

# 1 Code follow  N-Lora  paper

The related code and paper, please check  the link 

【COLING 2025】[Is Parameter Collision Hindering Continual Learning in LLMs?](https://arxiv.org/abs/2410.10179) 【COLING 2025】参数冲突会阻碍 LLM 的持续学习吗？ 

[![arXiv](https://camo.githubusercontent.com/6d6b1abf032c5f63c770a6679a3b4cd8435673c43428ce3fc877d32645672469/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f41727869762d50617065722d6233316231622e7376673f6c6f676f3d6172586976)](https://arxiv.org/abs/2410.10179) [![License](https://camo.githubusercontent.com/f062369f074348024e3a42e994b9295e0f3c9feb1f3c25eb910a68f8827e1607/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c6963656e73652d417061636865253230322e302d79656c6c6f77)](https://github.com/PKU-YuanGroup/N-LoRA/blob/main/LICENSE) [![Hits](https://camo.githubusercontent.com/29dd6ffb729fecb85323f7532cede80806918cb9c24991038e25d23a89e23431/68747470733a2f2f686974732e736565796f756661726d2e636f6d2f6170692f636f756e742f696e63722f62616467652e7376673f75726c3d68747470732533412532462532466769746875622e636f6d253246504b552d5975616e47726f75702532464e2d4c6f524126636f756e745f62673d253233373943383344267469746c655f62673d2532333535353535352669636f6e3d2669636f6e5f636f6c6f723d253233453745374537267469746c653d56697369746f7226656467655f666c61743d66616c7365)](https://hits.seeyoufarm.com/) [![GitHub issues](https://camo.githubusercontent.com/6642ca99d6afd7cc26aecb2236828d31529703e866168177f99115654ba47f97/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6973737565732f504b552d5975616e47726f75702f4e2d4c6f52413f636f6c6f723d637269746963616c266c6162656c3d497373756573)](https://github.com/PKU-YuanGroup/N-LoRA/issues?q=is%3Aopen+is%3Aissue)



```bibtex
@article{yang2024parameter,
  title={Is Parameter Collision Hindering Continual Learning in LLMs?},
  author={Yang, Shuo and Ning, Kun-Peng and Liu, Yu-Yang and Yao, Jia-Yu and Tian, Yong-Hong and Song, Yi-Bing and Yuan, Li},
  journal={arXiv preprint arXiv:2410.10179},
  year={2024}
}
```





#  Some Analyse abut the N-LoRA Code 

1  N-Lora  LoRA Part code using the peft 0.3.0:[peft · PyPI](https://pypi.org/project/peft/0.3.0/#files) .

 1.1 It change the code  src/peft/tuners/lora.py , [[N-LoRA/src/peft/tuners/lora.py at main · PKU-YuanGroup/N-LoRA](https://github.com/PKU-YuanGroup/N-LoRA/blob/main/src/peft/tuners/lora.py)], please check the code with note  # modified.

1.2 It change the code  src/peft/utils/save_and_load.py , [[N-LoRA/src/peft/utils/save_and_load.py at main · PKU-YuanGroup/N-LoRA](https://github.com/PKU-YuanGroup/N-LoRA/blob/main/src/peft/utils/save_and_load.py)] , please check the code with note  # modified.



2  I saved these changed and rename these file with _modifiedNlora, and I also try to  compare the two different vison , the compare result are saved with compare-analyse. 

I add the origin LoRA code. They are src/peft/tuners/lora_originLoRA030.py and src/peft/utils/save_and_load_originLoRA030.py

Pelese Note , If you want to 

- Run the origin lora code,  please copy src/peft/tuners/lora_originLoRA030.py into src/peft/tuners/lora.py , copy src/peft/utils/save_and_load_originLoRA030.py into src/peft/utils/save_and_load.py
- Run the Nlora code,  please copy src/peft/tuners/lora_modifiedNlora.py into src/peft/tuners/lora.py , copy src/peft/utils/save_and_load_modifiedNlora.py into src/peft/utils/save_and_load.py
- Check the difference between origin LoRA 0.3.0 version and  modified NLoRA , please

read the code src/peft/tuners/lora_compare-analyse.py.



# Understand main loop for run the code

## 1  use  bash to run the code

```
bash scripts/order_1.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &
```

It tun the scripts/order_1.sh , 

the log file are  logs_and_outputs/order_1/logs/train_and_infer.log

## 2 check the  scripts/order_1.sh

```python
CUDA_VISIBLE_DEVICES=1 deepspeed --master_port $port src/run_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path initial_model/t5-large \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_T5large-lora/order_1/outputs/1-dbpedia \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 256 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order1_round1 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.4
```

它来运行 src/run_lora.py 代码，这是程序的主入口， 并且传递了很多参数

注意，和我们之前讲到的，

- 如果想要运行原始lora代码，请复制 src/peft/tuners/lora_modifiedNlora.py 到 src/peft/tuners/lora.py ，并且为了正确保存，请复制 copy src/peft/utils/save_and_load_modifiedNlora.py 到 src/peft/utils/save_and_load.py
- 如果想要运行Nlora代码，请复制 src/peft/tuners/lora_modifiedNlora.py 到 src/peft/tuners/lora.py , copy src/peft/utils/save_and_load_modifiedNlora.py into src/peft/utils/save_and_load.py
- 如果想要查看这两个部分代码的区别，请查看 src/peft/tuners/lora_compare-analyse.py.



代码参数的设置

如果模型过大，请减小 --per_device_train_batch_size ， --per_device_eval_batch_size

请注意训练结束后保存的文件存储在   --output_dir 目录下，具体的文件名在  src/run_Nlora.py 下代码, 这里保存的名字要和.sh下次加载的路径要一致。

```python
# 修改名称
        peft_model_id = training_args.output_dir + "/adapter_T5small_Nlora_full"
        
```

## 3 check the run_Nlora.py 

主要的代码逻辑在 def main(): 函数中，

- 加载数据

-  加载预训练模型或者加载之前训练保存好的模型
- 数据按着训练集，测试集划分
- 设置trainer
- 训练模型并保存 if training_args.do_train:
- 评估模型效果 if training_args.do_predict:
- 我添加的代码 评估模型的 flat minal 



### 请注意 在第一次训练时，往预训练模型添加 LoRA  的逻辑

是在 

0  src/run_N_lora/ def main: 

```python
else: 
    model = get_peft_model(model, peft_config)
```

1 peft/mapping.py   get_peft_model(model, peft_config):

2 peft/peft_model.py / class PeftModelForSeq2SeqLM(PeftModel): / def \_init\_ 

3 peft/tuner/lora.py / class LoraModel: / def \_init\_ 

   def \_init\_ 调用了 self.add_adapter()  函数 , 

   self.add_adapter()  函数 调用了 self._find_and_replace(adapter_name) 函数

   self._find_and_replace(adapter_name) 函数 调用了  new_module = Linear(adapter_name, in_features, out_features, bias=bias, r_sum=lora_config.r_sum, **kwargs) 

这里生成了新的包含 lora部分的模块

4  peft/tuner/lora.py / class LoraModel: / def \_init\_  其中的  self.update_layer NLoRA进行了改动

这也是为什么调用Lora /Nlora 需要修改 lora.py 文件的原因



### 保存训练好的 lora 模型的逻辑 

src/peft/peft_model.py  

```python
class PeftModel(PushToHubMixin, torch.nn.Module):

	def save_pretrained(self, save_directory, **kwargs):
```



### 第二次运行加载训练好的Lora模型的代码的逻辑

0  src/run_N_lora  def main:

```python
if 'adapter' in model_args.model_name_or_path:
    config = PeftConfig.from_pretrained(model_args.model_name_or_path)
```

1  src/peft/utils/config.py     def from_pretrained(cls, pretrained_model_name_or_path, 

```python
@classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, subfolder=None, **kwargs):
        model.load_adapter(model_id, adapter_name, **kwargs)
        return model
```

2   src/peft/peft_model.py  class PeftModel(PushToHubMixin, torch.nn.Module): 

def load_adapter(self, model_id, adapter_name

```python
def load_adapter(self, model_id, adapter_name
# load the weights into the model
        set_peft_model_state_dict(self, adapters_weights, 
```

3 src/peft/utils/save_and_load.py.py

```python
def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="defaul
                              
```

这里作者也进行了改动，所以这也是为什么调用Lora /Nlora 需要修改 save_and_load.py 文件的原因



# Requirements.txt

If the requireents.txt file fails on your device ,you can try requirements_liying.txt.





# Download Initial Model 

To download initial_model, please run  download_Model.py.





# Eval the flat minal

To evaluate the model's flat minal, I add two methods on uie_trainer_lora.py.

1 compute_loss_landscape ,: To calcualte the Loss land and save in the .h5 file.

2 compute_hessian_version1: Use block_lanczos to calculate the Hessien eigvalue.



After train a model, just use 

trainer.compute_loss_landscape , trainer.compute_hessian_version1 to eval the model.



# Some log Information 

For clarify how the code work, I add some log information to check the middle information.



#  The modified I made

为了运行 flat minal 的分析代码，可以直接运行训练之后直接进行分析，这样只需要在,

最后调用 即可。

```python

```

但是为了区分到底当前运行的是lora 代码还是 Nlora的代码，我们在 

src/run_Nlora.py 中添加了  flag 用以区分

```python
@dataclass
class UIETrainingArguments(Seq2SeqTrainingArguments):
	 # 尝试添加自定义参数 ，do_flatminal 用来指示是否执行flatminal的评估
    do_flatminal: bool = field(default=False, metadata={"help": "Whether to do flatminal."})

    # 尝试添加自定义参数 ，flag_originLoRA 用来指示 当前方法是否为Lora 方法
    flag_originLoRA: bool = field(default=False, metadata={"help": "Whether to do flatminal."})
    
    # 尝试添加自定义参数 ，flag_modifiedNLoRA 用来指示 当前方法为NLora方法
    flag_modifiedNLoRA: bool = field(default=False, metadata={"help": "Whether to do flatminal."})
    
    # 尝试添加自定义参数 ，flag_modified_fullLoRA 用来指示 当前方法为NLora方法,是否对所有的lora部分进行评估
    flag_modifiedNLoRA_fullLoRA: bool = field(default=False, metadata={"help": "Whether to do flatminal."})
    
    # 尝试添加自定义参数 ，flag_modified_taskLoRA 用来指示 当前方法为NLora方法,是否只对与任务有关的LoRA进行评估
    flag_modifiedNLoRA_fullLoRA: bool = field(default=False, metadata={"help": "Whether to do flatminal."})
    
```

由于我们添加了新的参数，因此也需要相应地修改数据库中用来 进行 flat_minal 的参数，用来设定进行评估 flat_minal 的数据设定

```python
@dataclass
class DataTrainingArguments:
	# 添加参数，用来设置 flatminal 的最大数量
    max_flatminal_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )

    
# 修改代码 ，设置用来评估模型的参数
    if training_args.do_flatminal:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        flatminal_dataset = raw_datasets["test"]
        if data_args.max_flatminal_samples is not None:
            flatminal_dataset = flatminal_dataset.select(range(data_args.max_flatminal_samples))

```







注意为了节省空间，可以直接加载保存好的模型，然后再进行评估，而不是再训练完就进行评估，此时调用的 .sh 需要加载 保存好的模型，这里就需要重新设置保存结果的路径





