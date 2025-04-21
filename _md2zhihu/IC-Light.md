## 在复现IC-Light中游玩Stable-Diffusion-v1.5的记录

花了小一个月在这事上，因为老板项目没谈成所以似乎告一段落了。总而言之还是写一个记录，下次用SD的时候可能也会拿来看看。

> IC-Light项目主页：https://github.com/lllyasviel/IC-Light


### 对Diffusion原理的困惑

我们都知道，Diffusion是一个不断加噪后不断去噪的Markov过程，Stable-Diffusion(SD)就是在隐空间上，用Unet架构加入Condition的Diffsuion。
![alt text](https://cdn.jsdelivr.net/gh/XllDife/MarkdownRepo@_md2zhihu/asset/IC-Light/image.png)
在生成过程中，我们从一个纯噪声在Condition的控制下，一步步进行去噪，最终得到生成出的图片。那么，在训练过程中，我们是不是对这每一步的去噪都要学习对应的关系，也就是需要得到一组不断加噪的图片，然后将相邻两张输入Unet进行训练？

是也不是，我们确实需要学习每一步去噪的对应关系。但是通过学单独的每个加噪阶段来的。具体来说，我们对一个随机的噪声$\epsilon$，通过 Scheduler 得到 t 时间上的加噪图像 $z_t$，将 $z_t$和 $t$ 作为Unet的输入进行训练。
<img src="https://www.zhihu.com/equation?tex=z_t%3D%5Csqrt%7B%5Cbar%7B%5Calpha%7D_t%7Dz_0%2B%5Csqrt%7B1-%5Cbar%7B%5Calpha%7D_t%7D%5Cepsilon" alt="z_t=\sqrt{\bar{\alpha}_t}z_0+\sqrt{1-\bar{\alpha}_t}\epsilon" class="ee_img tr_noresize" eeimg="1">
训练中，我们一般想要预测原始噪声（约等于预测原图），也就是说我们对于给定的加噪图像$z_t$和时间步 $t$，就能得到我们加的原始噪声$\epsilon$。而有了这个噪声，在生成过程中我们就能进行一步去噪，得到$z_{t-1}$从而完成生成过程。
![alt text](https://cdn.jsdelivr.net/gh/XllDife/MarkdownRepo@_md2zhihu/asset/IC-Light/image-1.png)
那么实际上，$z_{t-1}$与$z_t$之间的相关关系蕴含于时间步$t$中。不同的$t$决定了Unet对图像加噪程度的预期，从而更好的生成不同尺度的特征。

### 在写代码中遇到的困难

代码上我使用Vscode + Copilot（Claude3.7）作为主要工具。所以下述问题主要都是通过Copilot解决。

#### 多次反向传播

> 一致性训练出错: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.


IC-Light训练由两部分组成: Vanilla和Consistency。我为他们各写了一个训练函数，但同时我在他们训练函数末尾都加了一个backward()。这是错误，且不必要的。反向传播可以在两个训练模块都完成后进行。

最后我加了一个训练的总函数，调用两个训练模块，最后在末尾进行诸如反向传播之类的工作。

#### 多卡训练

用torchrun框架构造。需要重新包装一下优化器和模型，并加入一个setup_ddp()的世界初始化。没什么细节的，照抄Copilot。

#### 测试数据集

因为数据集由其他同学负责处理，我本来并没有去跑过我的代码。经师兄提醒，生成了一份随机的测试数据集用来debug。

经验必活用于下次。

#### 多卡运行中的VScode调试(?)

好像要多进程在VScode有点麻烦。我现在也不太搞得懂，难道debug的时候应该开单进程？但这样那些多进程的包装之类的操作不会报错么？

#### device

也太烦人了，训练前写一段检查device的语句吧。

#### 构造数据集

终于等到了可用的数据集，虽然只有6个。构造继承于torch.utils.data.Dataset的RelightDataset类，其中保存查找各个样本用的样本id，和各属性贴图的地址即可。具体数据的赋予，放在类中的__getitem__函数中处理。（get item，从中读取数据）

#### 数据集类中使用模型

在数据处理其中本来我会用CLIP去处理条件文本变为tensor，但这样会报device不一致的错误。Copilot的建议是让我尽量不要在dataset类中使用gpu模型：

> 解决方案

方案 1: 在 dataset 中避免使用模型
最佳做法是在数据集类中避免使用模型，特别是 GPU 模型。让我们修改 dataset.py 来返回 token IDs 而不是编码：`然后在`train_step` 中处理这些 token IDs：
方案 2: 设置 num_workers=0
如果您不想修改 dataset 代码，另一个选择是禁用多进程加载：
这会使用主进程加载数据，避免模型在工作进程中使用的问题，但会降低数据加载性能。
方案 3: 在 dataset 中明确处理设备

似乎是多进程加载的问题。我的理解是读数据主要是cpu在发力，尽量不使用gpu。

#### 梯度累积

> 梯度累积是一种有效的训练技术，可以模拟更大批量的训练效果而不增加显存占用。这对于大型模型和复杂架构尤其有用，它允许您使用较小的批次大小，但通过累积多个批次的梯度来实现与大批次训练相似的效果。


梯度累积（gradient_accumulation）顾名思义，就是过几个batch再更新梯度，假装我的batch_size很大。这对显存严重不足只能跑size=1的我来说很有用。

> 缺点：

1.训练速度略有下降：由于需要多个步骤才进行一次参数更新，总体训练速度会略有下降。
2.批归一化层影响：梯度累积可能会影响批归一化层的表现，因为它们使用小批次的统计数据。如果发现训练不稳定，可以考虑使用组归一化等替代方案。
3.调整学习率：使用梯度累积时，可能需要调整学习率，通常是将学习率乘以累积步数的平方根。

### 调用SD上遇到的问题

#### 哪个才是SD1.5？

我以为SD1.5在stability.ai的huggingface主页里应该能找到，但现在（2025.4）主页中只有那些新SD模型了。

大家现在用的最多的SD1.5模型应该是

> sd15_name = 'stablediffusionapi/realistic-vision-v51'


#### 怎么用？

找到SD1.5以后，从里面读出他的各个模块

```python
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder").to(device=device, dtype=dtype)
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae").to(device=device, dtype=dtype)
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet").to(device=device, dtype=dtype)
```

然后，你就可以手写一个训练过程了（随机时间步，加噪，训练，预测噪声）

在IC-Light的Demo里，它使用的是SD的Pipeline做为生成过程。

```python
t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)
```

那微调是不是也得自己写？

#### 训练

跑起来了很好，但是又不是很好。随便跑了一个结果出来放demo那一用，我了个豆，一点用没有。讨论了一下原来是bug。

bug改完了再试试，我了个豆，还是一点用没有。-原来是学习率太低了。

学习率也调了，epoch也多了，现在呢：

![alt text](https://cdn.jsdelivr.net/gh/XllDife/MarkdownRepo@_md2zhihu/asset/IC-Light/11931ff235b8c39f6ddd4c31ec2305d4.png)

散了散了

#### Wandb上的可视化图表

![alt text](https://cdn.jsdelivr.net/gh/XllDife/MarkdownRepo@_md2zhihu/asset/IC-Light/dcdfd56c2f5a743f9e2166c58c6ee178.png)
好酷的弹跳啊。但SD的训练好像都是震荡中下降？

### 总结

这算我第一次成功(?)的代码复现，在一个已有的预训练模型上修改成一个新模型，也是对跑模型这事认知更深了，希望下次面对那些大项目代码能更游刃有余吧。

对SD这事来说，使用SD很简单，所以当年AI绘图才这么普及，更难的可能是数据的收集与处理，训练过程只要简单修改即可。



Reference:

