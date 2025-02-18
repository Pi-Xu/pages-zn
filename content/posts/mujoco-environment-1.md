---
title: mujoco-py环境配置
date: 2024-01-03T14:19:27+08:00
tags: [环境配置]
categories: [技术笔记]
---

## 写在前面

有时候会需要配置环境, 但是发现自己每次都得重新开始配置环境, 还是记录一下比较好... 当然记录可能也可能会出错! 之后有啥问题再 Google!

## mujoco 环境快速安装

- Step 1: 下载到 `/root/` 文件夹下(Terminal的 `~/` 即为)
```bash
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
```

- Step 2: 解压至 `.mujoco/`
```bash
mkdir ~/.mujoco # 创建
tar -zxvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
```

- Step 3: 追加环境信息
```bash
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin" >> ~/.bashrc 
source .bashrc
```

- Step 4: 在对应的 conda 环境下安装相关的 python 库
```bash
pip install mujoco
pip install mujoco-py
```

---
换源: Ubuntu 的软件源配置文件是 `/etc/apt/sources.list`。将系统自带的该文件做个备份，将该文件替换为下面内容，即可使用选择的软件源镜像。(虽然 PC 上不需要, 但是服务器上可能需要)

```bash
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse

deb http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse
deb-src http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
```

可以通过 vim 命令完成:
```bash
vim /etc/apt/sources.list
### --- ###
# 在 vim 视图中
# 清空源
# `gg` 跳转到文件开头 d 删除 G直到文件末尾
ggdG
# 然后将上述代码粘贴即可
```

依赖安装
```bash
apt-get update
apt install libosmesa6-dev libgl1-mesa-glx libglfw3
apt-get install patchelf
```

注：可能还需要一些其他的安装, 直接参考 mujoco-py 的 [github](https://github.com/openai/mujoco-py) 库
