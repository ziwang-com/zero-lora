# zero-lora零训练llm调参算法

工程案例参见：全球首个StableVicuna中文优化版。

https://github.com/ziwang-com/chinese-StableVicuna

整个项目，仅用半天时间，其中大部分时间花在格式转换方面，与zero-lora相关的环节，不到20%。

zero-loro零训练llm调参算法，属于zw团队在llm一线工程中，总结的实战算法，相关理论，正在摸索当中，欢迎llm领域的专家学者，共同探讨。

![st-vic-qcod](https://user-images.githubusercontent.com/11691791/235562989-601c9ead-7732-4c56-b380-324f0866536e.png)

所谓zero-lora调参，就是无需任何调参，直接采用，已经调整过的各种lora，与相关的llm模型，进行叠加，合成即可。

这个特别适合原版模型的汉化处理，可以把汉化包视为一个lora补丁。

虽然zw-sd-vicuna可能是llm第一个zero-lora工程案例，不过在AIGC绘图模型领域，已经有大量的第三方stable difusion绘图模型，是基于lora叠加模式。

### 优点：
* 算力成本只有传统lora调参的万分之一。
* 无需GPU计算，平台cpu即可，唯一要求就ram尽量大一点，推荐：i9，64G
* 已经有成功的工程案例：zw-sd-vicuna
* 可以使用线性数学，快速处理lora数据。
* 优化chatgpt训练模型矢量权重数据。
* 微调的低成本替代方案。
* 便于可视化分析，参见后文

目前，关于zero-lora调参，网络上面其实也已经有不少第三方的独立探讨，只不过角度不同，名称不同，这其中的观点有：

* 论文《转向矢量与GPT优化》，已经上传到本项目网站。
https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector

* 《[研究] 转向矢量》
    * https://github.com/ggerganov/llama.cpp/issues/1460 
    * https://github.com/ggerganov/llama.cpp/pull/1472

* 《10 个 loras 在一个 ggml 文件中》
    * https://github.com/ggerganov/llama.cpp/discussions/1481

###【可视化分析】

参见：
https://github.com/ggerganov/llama.cpp/pull/1472
代码#1，生成token数据
./bin/main --model ../models/llama-7b-q4_0.bin -n 32 \
   --seed 123 \
  --prompt "I want to kill you because you're such a" \
  --steering-add "I love you so much" \
  --steering-sub "I hate you so much" \
  --steering-source 1 \
  --steering-layer 20 \
  --steering-mul 2
代码#2，可视化分析

import numpy as np
from matplotlib import pyplot as plt
​
steer = np.fromfile("~/src/llama.cpp/build/steering.bin", dtype=np.float32).reshape((512, -1))
​
fig, ax = plt.subplots(3)
for i in range(0, len(ax)):
    ax[i].imshow(steer[3+i, :].reshape((32, -1)))

![xlora03](https://github.com/ziwang-com/zero-lora/assets/11691791/be38138a-d4aa-48b0-a8a0-525c16623665)


传统llm模型的lora调参训练，通常只有loss曲线，无法对具体的token，进行深度的关联分析 ，情感分析。

以上图片，只是最简单的love和hate，两个不同情感单词，token语句的可视化分析。

下图，是论文《转向矢量与GPT优化》当中，关于“婚礼”一词不同layer的权重对比分析。

![Uploading xlora05.png…]()


理论上，这方面有无限延展的空间，甚至可以衍生出一个完整的可视化lora优化架构和理论体系。

##TODO：

本文，可能首次正式提出zero-lora调参这个概念，这只是个开始。

关于zero-lora调参架构，还有相关的理论体系，还有大量的工作需要大家补充完善。

其中，根据AI一线工程经验，比较急切的有：

各种不同模型架构的lora的归一化。sd绘图模型lora叠加之所以成为主流优化架构之一，其中最主要的原因就是，base-mode只有stable difusion一种，各种第三方优化模型权重，shape，size等参数统一，见过简单处理，即可以直接叠加。

zero-lora架构，各个相关环节的梳理，优化。

可量化的评测指标，便于不同lora体系的整合。

简单完善zero-lora可视化模块。

zero-lora相关理论体系研究。

基于时间（不同训练周期检查点）、空间（不同token权重对比）、z深度（tok在不同模型的权重映射）等多种维度的lora权重优化体系。

多模态lora权重优化体系。

。。。。。。





