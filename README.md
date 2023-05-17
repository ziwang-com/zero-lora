# zero-lora
zero零训练llm调参
工程案例参见：全球首个StableVicuna中文优化版。

https://github.com/ziwang-com/chinese-StableVicuna

整个项目，仅用半天时间，其中大部分时间花在格式转换方面，与zero-lora相关的环节，不到20%。

所谓zero-lora调参，就是无需如何调参，直接采用，已经调整过的各种lora，与相关的llm模型，进行叠加，合成即可。
## 优点：
算力成本只有传统lora调参的万分之一。
无需GPU计算，平台cpu即可，唯一要求就ram尽量大一点，推荐：i9，64G
已经有成功的工程案例：zw-sd-vicuna
虽然zw-sd-vicuna可能是llm第一个zero-lora案例，不过在AIGC
绘图模型领域，已经有大量的第三方stable difusion绘图模型，是基于lora叠加模式。


# TODO：

### TODO：
