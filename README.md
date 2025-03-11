





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

 1.1 It change the code  src/peft/tuners/lora.py , [[N-LoRA/src/peft/tuners/lora.py at main · PKU-YuanGroup/N-LoRA](https://github.com/PKU-YuanGroup/N-LoRA/blob/main/src/peft/tuners/lora.py)](https://github.com/zuti666/Continual-Flat-LoRA/blob/main/src/peft/tuners/lora.py), please check the code with note  # modified.

1.2 It change the code  src/peft/utils/save_and_load.py , [[N-LoRA/src/peft/utils/save_and_load.py at main · PKU-YuanGroup/N-LoRA](https://github.com/PKU-YuanGroup/N-LoRA/blob/main/src/peft/utils/save_and_load.py)](https://github.com/zuti666/Continual-Flat-LoRA/blob/main/src/peft/utils/save_and_load.py) , please check the code with note  # modified.



2  I saved these changed and rename these file with _Nola, and I also try to  compare the two different vison , the compare result are saved with _compare. 

I add the origin LoRA code. They are src/peft/tuners/lora_originLoRA030.py and src/peft/utils/save_and_load_originLoRA030.py

Pelese Note , If you want to 

- Run the origin lora code,  please copy src/peft/tuners/lora_originLoRA030.py into src/peft/tuners/lora.py , copy src/peft/utils/save_and_load_originLoRA030.py into src/peft/utils/save_and_load.py
- Run the Nlora code,  please copy src/peft/tuners/lora_modifiedNlora.py into src/peft/tuners/lora.py , copy src/peft/utils/save_and_load_modifiedNlora.py into src/peft/utils/save_and_load.py
- Check the difference between origin LoRA 0.3.0 version and  modified NLoRA , please

read the code src/peft/tuners/lora_compare-analyse.py.

3 
