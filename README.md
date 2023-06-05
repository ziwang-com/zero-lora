# zero-lora零训练llm调参算法

工程案例参见：全球首个StableVicuna中文优化版。

https://github.com/ziwang-com/chinese-StableVicuna

整个项目，仅用半天时间，其中大部分时间花在格式转换方面，与zero-lora相关的环节，不到20%。

zero-loro零训练llm调参算法，属于zw团队在llm一线工程中，总结的实战算法，相关理论，正在摸索当中，欢迎llm领域的专家学者，共同探讨。


![st-vic-qcod](https://user-images.githubusercontent.com/11691791/235562989-601c9ead-7732-4c56-b380-324f0866536e.png)

近日很多网友咨询zw-Stable-Vicuna，模型仍在测试，目前只对合作伙伴提供。
合作伙伴，请提供相关文字资料：团队核心成员简介，研究课题，合作方向，以及相关PPT资料。
联系方式： 
微信：zwpython，或扫描二维码。QQ：357811718（zw字王） 
联系信息注明：sv模型合作。 

![qcod-zw200x](https://user-images.githubusercontent.com/11691791/236652627-76351e5d-68e8-43a5-98e6-5b837ae0a3f1.png)

所谓zero-lora调参，就是无需任何调参，直接采用，已经调整过的各种lora，与相关的llm模型，进行叠加，合成即可。

这个特别适合原版模型的汉化处理，可以把汉化包视为一个lora补丁。

虽然zw-sd-vicuna可能是llm第一个zero-lora工程案例，不过在AIGC绘图模型领域，已经有大量的第三方stable difusion绘图模型，是基于lora叠加模式。

### 优点：
* 算力成本只有传统lora调参的万分之一。
* 无需GPU计算，cpu即可，ram尽量大一点，推荐：i9，64G以上
* 已经有成功的工程案例：zw-sd-vicuna
* 可以使用线性数学，快速处理lora数据。
* 优化chatgpt训练模型矢量权重数据。
* 微调的低成本替代方案。
* 便于可视化分析，参见后文
* ......

关于zero-lora调参，目前也已经有不少第三方在研究，只不过角度不同，名称不同，其中的观点有：
* （相关资料稳定，已经上传到本项目网站）
* 论文《转向矢量与GPT优化》。
https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector

* 《[研究] 转向矢量》
    * https://github.com/ggerganov/llama.cpp/issues/1460 
    * https://github.com/ggerganov/llama.cpp/pull/1472

* 《10 个 loras 在一个 ggml 文件中》
    * https://github.com/ggerganov/llama.cpp/discussions/1481
    *
* 《Lora能在不同的大语言模型间交叉融合使用吗？》
    * https://github.com/StarRing2022/ChatGPTX-Uni

## 【可视化分析】

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

steer = np.fromfile("~/src/llama.cpp/build/steering.bin", dtype=np.float32).reshape((512, -1))

fig, ax = plt.subplots(3)
for i in range(0, len(ax)):
    ax[i].imshow(steer[3+i, :].reshape((32, -1)))

![xlora03](https://github.com/ziwang-com/zero-lora/assets/11691791/be38138a-d4aa-48b0-a8a0-525c16623665)


传统llm模型的lora调参训练，通常只有loss曲线，无法对具体token，进行深度的关联分析 ，情感分析。

以上图片，只是最简单的love和hate，两个不同情感单词，token语句的可视化分析。

下图，是论文《转向矢量与GPT优化》当中，关于“婚礼”一词不同layer的权重对比分析。

<img width="632" alt="xlora05" src="https://github.com/ziwang-com/zero-lora/assets/11691791/8ff8054a-7cde-418d-b4f3-55381ad3fd78">

理论上，这方面有无限延展的空间，甚至可以衍生出一个完整的：可视化lora优化架构和理论体系。

## TODO：

本文，可能是llm领域，首次正式提出zero-lora调参，这一概念，这也许，只是个开始。

关于zero-lora调参架构，还有相关的理论体系，有大量的工作，需要大家补充完善。
想刷高分paper，以及在GPT时代，寻找市场机会的llm创业团队，尽管放马过来。

根据AI一线工程经验，其中，比较急切的问题有：

* 各种不同模型架构的lora的归一化。sd绘图模型lora叠加之所以成为主流优化架构之一，其中最主要的原因就是，base-mode只有stable difusion一种，各种第三方优化模型权重，shape，size等参数统一，只需简单处理，即可以直接叠加。
* zero-lora架构，各个相关环节的梳理，优化。
* 可量化的评测指标，便于不同lora体系的整合。
* 完善zero-lora可视化模块。
* zero-lora相关理论体系研究。
* 基于时间（不同训练周期检查点）、空间（不同token权重对比）、深度（不同模型的tok权重映射）等多种维度的lora权重优化体系。
* 多模态lora权重优化体系。
* 集成TOT思维树等广义版本的zero-lora零训练llm优化技术，通过优化模型的逻辑流程，提高llm的实际推理能力。类似我们的logNET逻辑神经网络，如果能够集成专业知识库，更加理想。
* 。。。。。。



近期，zw团队正在升级m-f.vip元字库网站，尽快部署行业领先的：中文GPT和国际版GPT。
为方便国内广大个人用户，中小企业，创业团队，尽快体验GPT这一最新科技成果。
m-f.vip元字库网站，近期，将率先在国内推出低成本的GPT”百元包月“服务，不限流量，任意使用。

![zwgpt-pub_20230517161849](https://github.com/ziwang-com/zero-lora/assets/11691791/c692636d-139a-4da7-b9e9-d5bef98904d4)


更多参见：
http://ziwang.com 
http://metafont.vip 
短域名:http://m-f.vip 


![zwagi- (4)](https://github.com/ziwang-com/zero-lora/assets/11691791/628f48a9-d5b4-4a87-9985-0008b1f70f82)
