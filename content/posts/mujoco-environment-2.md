---
title: mujoco 记录视频问题
date: 2024-01-08T14:19:27+08:00
tags: [bug修复]
categories: [技术笔记]
---

## env.render() 出错

之前想记录机器人运行的视频, 然后发现无法渲染(也许是用这个词吧!)的问题:
1. 渲染无图 (当时google了一下, 好像是 `unset` 了一个什么东西, 虽然能够 `render` 但是图像全黑...)
2. `# Failed to load OpenGL` (如果不 `unset` 那个玩意儿, 就会出现这个问题)

找到了一个issue里面提到[解决办法](https://github.com/openai/mujoco-py/issues/665#issuecomment-1049503083)

> There have been multiple issues which relate to the same error ([#598](https://github.com/openai/mujoco-py/issues/598), [#187](https://github.com/openai/mujoco-py/issues/187), [#390](https://github.com/openai/mujoco-py/issues/390)) but unfortunately none of them worked for me. What worked was to change [this line](https://github.com/openai/gym/blob/c8321e68bbd7e452ef65fc237525669979bb45af/gym/envs/mujoco/mujoco_env.py#L193) in `<site_packages>/gym/envs/mujoco/mujoco_env.py` to:
> ```python
> self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, None, -1)
> ```
> You can find the location of `<site_packages>` in your system by using `pip show gym`.

如果是 gymnasium 的话, 就在 `<site_packages>` 里面找到 `gymnasium/envs/mujoco/mujoco_env.py` 然后修改 `self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, None, -1)` 这一行.

一些版本信息:
```bash
gym                       0.26.2                   pypi_0    pypi
gym-notices               0.0.8                    pypi_0    pypi
gymnasium                 0.26.3                   pypi_0    pypi
gymnasium-notices         0.0.1                    pypi_0    pypi
mujoco                    3.1.1                    pypi_0    pypi
mujoco-py                 2.1.2.14                 pypi_0    pypi
```