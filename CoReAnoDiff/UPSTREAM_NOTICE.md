# 原项目来源

本研究原型基于 [DualAnoDiff](https://github.com/yinyjin/DualAnoDiff) 的普通双分支实现，源版本为 `0ae11b06ec10eaf38447c549b8188ed8880a56a4`。原说明见 [UPSTREAM_README.md](docs/UPSTREAM_README.md)。

原论文：Jin et al., *DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation*, arXiv:2408.13509。原论文的作者、发表记录及实验结果不代表CoReAnoDiff的作者信息或结果。

`dual-interrelated_diff/diffusers/` 是原仓库所携带的修改版diffusers，当前双分支适配器依赖其中自定义处理器，不能直接换成官方pip包。保留完整包是为了维持它的导入依赖，而不是表示本项目修改了其中每个模型。

原代码中的版权与许可头均保留。本审核包没有另行声明覆盖所有源码的新许可证；相关文件沿用其原有声明。本项目新增内容主要为类别独立适配器管理、区域输出残差路由、独立训练/推理入口、数据整理工具和测试。
