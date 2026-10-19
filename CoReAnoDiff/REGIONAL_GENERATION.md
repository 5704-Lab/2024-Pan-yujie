# 类别适配与区域组合：使用说明

当前主入口位于 `dual-interrelated_diff`。每个专家是完整双分支LoRA与SAIM适配器，不是独立的普通单分支LoRA。

## 数据

使用LabelMe多边形标注真实可见的病害区域，类别可用 `honeycomb`、`groove`、`mud`。同图不同类别分开标注，同类多块可使用多个同名多边形。边界需要专业判定；未标注不等于正常。

准备图片与同名JSON到 `data/train_annotations/`，将人工确认正常的图放到 `data/normal/`。应先按芯样/孔号/采集来源分组划分集合，不要让同图裁剪或连拍跨训练、验证、测试。以下仅导出已选定的训练部分：

```bash
cd dual-interrelated_diff
python export_core_annotations.py ../data/train_annotations ../data/train_prepared \
  --classes honeycomb groove mud --normal-images ../data/normal --reference-images
```

输出原图分辨率的各类0/255单通道PNG、单类JSONL、共存清单和统计。`--reference-images` 以相对路径引用原图，不复制大量图像；移动时保留相对关系。正常图来自明确传入的正常目录，并记录来源，不由空JSON推断。

独立专家第一版只读取单类记录，共存图保留在 `mixed.jsonl`；它不能简单复制两份并删去另一类标签后当作纯单类。所有清单必须保留实际类别集合，以便校验。

单类清单格式（路径相对于清单目录）：

```json
{"image":"images/example.png","mask":"masks/example_honeycomb.png","class":"honeycomb","labels":["honeycomb"],"specimen_id":"example-core"}
```

正常记录使用 `class=normal, labels=[]`，不要求缺陷掩码。单类判断只说明标注类别集合，不能代替漏标检查。

整景照片中缺陷在512尺寸下可能只剩几像素。应先裁剪单根芯样或合适局部并同步转换坐标，保留来源和上下文。`prepare_generation_pilot.py` 是流程测试裁剪工具，依赖与原图哈希一致的YOLO芯样多边形，当前固定使用class2作为芯样轮廓，使用前核对类别定义。它不是通用自动缺陷标注器，也不能自动确认正常。

## 基础权重与训练

基础模型目录需包含SD1.5的tokenizer、text_encoder、vae、unet、scheduler配置及权重；公开来源为 `stable-diffusion-v1-5/stable-diffusion-v1-5`。示例目录相对于项目根目录。

```bash
python manifest_dataset.py ../data/train_prepared/honeycomb.jsonl --category honeycomb
python manifest_dataset.py ../data/train_prepared/normal.jsonl --category normal --normal

python train_regional_lora.py \
  --base-model ../models/stable-diffusion-v1-5 \
  --manifest ../data/train_prepared/normal.jsonl --category normal --role anchor \
  --output regional_runs/normal --steps 20 --save-every 20

python train_regional_lora.py \
  --base-model ../models/stable-diffusion-v1-5 \
  --manifest ../data/train_prepared/honeycomb.jsonl --category honeycomb --role expert \
  --anchor regional_runs/normal/adapter_bank.pt \
  --output regional_runs/honeycomb --steps 20 --save-every 20

python train_regional_lora.py \
  --base-model ../models/stable-diffusion-v1-5 \
  --manifest ../data/train_prepared/groove.jsonl --category groove --role expert \
  --anchor regional_runs/normal/adapter_bank.pt \
  --output regional_runs/groove --steps 20 --save-every 20
```

20步仅用于跑通流程，并未生成有效缺陷。正式训练步数应在数据核验后通过验证集确定。默认512、batch1、累积4、rank4、fp16；显存不足可尝试 `--gradient-checkpointing`。

所有专家必须从同一正常基准初始化，文本编码器冻结。专家rank继承基准。输出 `adapter_bank.pt` 完整保存两个分支LoRA与 `temporal_*` 参数，checkpoint还有优化器、scaler及随机状态。`--resume` 仅加载自己可信的 `training_state.pt`，恢复时保留原训练参数。不要在专家训练后覆盖其基准文件；推理检查SHA256。

## 区域组合推理

编辑 `regional.example.json`：权重、适配器、布局和输出路径相对于配置文件目录。每个类别提供一张与输出尺寸一致的单通道布局PNG，像素0～255；这些是请求布局，不是生成结果的真实标注。

```bash
python inference_regional.py regional.example.json
```

在各自注意力和交叉注意力处理器输出处计算：

\[
Y=Y_a+\sum_k\alpha_k\odot(Y_k-Y_a),\quad
\alpha_k=\frac{w_k\widetilde M_k}{\max(1,\sum_jw_j\widetilde M_j)}.
\]

每个专家内部先完成原双分支交互，再融合输出差值。差值包含LoRA与SAIM的影响，不能把收益全部归因于LoRA。特征高宽来自真实Transformer2D输入，支持矩形；同布局作用于双分支与CFG。单次生成一对整图/局部图，多张通过seeds列表生成。

硬布局使用 `interpolation=nearest, feather=0`；软边界可用bilinear和feather。重叠处总权重归一化，强度不会无限放大残差。`shared_saim=true` 是推理替换SAIM的消融；专家训练的 `--freeze-saim` 是另一个不同实验。

`mode=edit` 时指定reference和edit_strength，运行参考潜变量背景约束。`image_raw.png` 是采样输出，`image.png` 还经过布局alpha与对齐原图的像素合成，须分开评估。安装区域处理器后，勿再调用会替换处理器的LoRA加载、xformers或attention slicing。

当前不支持BCM、SDXL和ControlNet。局部输出路由不保证像素隔离；注意力、卷积及后续交互会传播影响。生成掩码需核验实际边界后才能用于下游训练。

## 验证

```bash
python -m unittest discover -s tests -v
```

24项CPU测试覆盖退化关系、空间路由、重叠归一化、双分支/CFG、自/交叉注意力、参数保存、梯度、数据导出及裁剪。注意力单元测试直接提取原源码类执行以隔离可选pipeline依赖；采样编排测试使用微型替身。实际GPU流程也已执行，但生成质量不合格，见 [流程记录](GENERATION_SMOKE.md)。

正式研究需对比原DualAnoDiff、全局专家混合、区域路由、拼贴和顺序编辑，并消融SAIM及边界处理；这些自动对比实验尚未实现。独立训练不等于表征纯粹解耦，当前也没有证明区域命中、减少串扰或下游增益。
