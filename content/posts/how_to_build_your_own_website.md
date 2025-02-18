---
title: 如何搭建你的个人主页
date: 2023-11-16T14:57:27+08:00
tags: [个人主页相关]
categories: [技术笔记]
---

# 如何搭建你的个人主页

## 安装 Hugo

在安装 Hugo 之前, 你需要先安装 git, 可以参见 [git 官网](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

安装 Hugo 可以参见 [Hugo 官网](https://gohugo.io/getting-started/installing/). 我是在 Linux 下安装的, 使用的是 snap 安装的, 命令如下:

```bash
sudo snap install hugo
```

在完成安装后, 可以使用 `hugo version` 命令查看 hugo 的版本.

## 创建一个网站

使用 `hugo new site <网站名>` 命令创建一个网站, 网站名可以自己定义, 我这里使用的是 `my_website`.

```bash
hugo new site my_website
cd my_website
```

通过 `ls` 指令可以看到:

```bash
archetypes  assets  content  data  hugo.toml  i18n  layouts  static  themes

```

## 选择一个主题

在主题库中选择一个主题, 可以参见 [Hugo Themes](https://themes.gohugo.io/).

我这里选择的是 [LoveIt](https://github.com/dillonzq/LoveIt). 可以直接把这个主题克隆到 themes 目录:

```bash
git clone https://github.com/dillonzq/LoveIt.git themes/LoveIt
```

或者, 初始化你的项目目录为 git 仓库, 并且把主题仓库作为你的网站目录的子模块:

```bash
git init
git submodule add https://github.com/dillonzq/LoveIt.git themes/LoveIt
```

## 初始化主题

将下述代码复制到 `hugo.toml` 文件中, 并且根据你的需要进行修改. 在之后的文章中, 我将继续解释 `hugo.toml` 文件中的配置. (挖个坑)

```toml
baseURL = "http://example.org/"

# 更改使用 Hugo 构建网站时使用的默认主题
theme = "LoveIt"

# 网站标题
title = "我的全新 Hugo 网站"

# 网站语言, 仅在这里 CN 大写 ["en", "zh-CN", "fr", "pl", ...]
languageCode = "zh-CN"
# 语言名称 ["English", "简体中文", "Français", "Polski", ...]
languageName = "简体中文"
# 是否包括中日韩文字
hasCJKLanguage = true

# 作者配置
[author]
  name = "xxxx"
  email = ""
  link = ""

# 菜单配置
[menu]
  [[menu.main]]
    weight = 1
    identifier = "posts"
    # 你可以在名称 (允许 HTML 格式) 之前添加其他信息, 例如图标
    pre = ""
    # 你可以在名称 (允许 HTML 格式) 之后添加其他信息, 例如图标
    post = ""
    name = "文章"
    url = "/posts/"
    # 当你将鼠标悬停在此菜单链接上时, 将显示的标题
    title = ""
  [[menu.main]]
    weight = 2
    identifier = "tags"
    pre = ""
    post = ""
    name = "标签"
    url = "/tags/"
    title = ""
  [[menu.main]]
    weight = 3
    identifier = "categories"
    pre = ""
    post = ""
    name = "分类"
    url = "/categories/"
    title = ""

# Hugo 解析文档的配置
[markup]
  # 语法高亮设置 (https://gohugo.io/content-management/syntax-highlighting)
  [markup.highlight]
    # false 是必要的设置 (https://github.com/dillonzq/LoveIt/issues/158)
    noClasses = false
```

## 创建第一篇文章

此时, 你可以使用 `hugo new posts/my-first-post.md` 命令创建你的第一篇文章, 并且编辑它.

```bash
hugo new posts/my-first-post.md
```

此时, 你可以在 `content/posts` 目录下看到你的文章.

```markdown
+++                                     
title = 'First_post'
date = 2023-11-16T14:00:41+08:00
draft = true
+++
```

此时需要将 `draft` 改为 `false`, 并且编辑你的文章.

```markdown
+++
title = 'First_post'
date = 2023-11-16T14:00:41+08:00
draft = false
+++
# Hello World
```

在本地启动你的 hugo 服务, 并且在浏览器中打开 `http://localhost:1313/`:

```bash
hugo server
```

## 与 GitHub Pages 集成

那么在完成上述步骤后, 你的网站已经可以在本地运行了, 但是你可能想要将你的网站部署到 GitHub Pages 上. 目前一些做法可能会将github的个人主页仓库公开, 但是那样可能会导致自己的数据不安全, 我们可以将自己的仓库设置为私有的, 再通过github action来完成个人主页的推送, 那么你需要做的是:

- 创建一个私有仓库
- 将本地仓库 push 到 github 上面
- 在该 github 仓库中, 从主菜单选择 `Settings > Pages`. 在页面中会看到 `Build and deployment` 下面有 `Source`
- 将 `Source` 中 `Deploy from a branch` 更改为 `GitHub Actions`. 更改是即时的, 无需按保存按钮
- 建立一个空的文件:
```bash
.github/workflows/hugo.yaml
```
- 将下面的 YAML 复制并粘贴到创建的文件中. 根据需要更改分支名称和 Hugo 版本

```yaml
# Sample workflow for building and deploying a Hugo site to GitHub Pages
name: Deploy Hugo site to Pages

on:
  # Runs on pushes targeting the default branch
  push:
    branches:
      - main

  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:

# Sets permissions of the GITHUB_TOKEN to allow deployment to GitHub Pages
permissions:
  contents: read
  pages: write
  id-token: write

# Allow only one concurrent deployment, skipping runs queued between the run in-progress and latest queued.
# However, do NOT cancel in-progress runs as we want to allow these production deployments to complete.
concurrency:
  group: "pages"
  cancel-in-progress: false

# Default to bash
defaults:
  run:
    shell: bash

jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    env:
      HUGO_VERSION: 0.120.2
    steps:
      - name: Install Hugo CLI
        run: |
          wget -O ${{ runner.temp }}/hugo.deb https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.deb \
          && sudo dpkg -i ${{ runner.temp }}/hugo.deb          
      - name: Install Dart Sass
        run: sudo snap install dart-sass
      - name: Checkout
        uses: actions/checkout@v4
        with:
          submodules: recursive
          fetch-depth: 0
      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v3
      - name: Install Node.js dependencies
        run: "[[ -f package-lock.json || -f npm-shrinkwrap.json ]] && npm ci || true"
      - name: Build with Hugo
        env:
          # For maximum backward compatibility with Hugo modules
          HUGO_ENVIRONMENT: production
          HUGO_ENV: production
        run: |
          hugo \
            --gc \
            --minify \
            --baseURL "${{ steps.pages.outputs.base_url }}/"          
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v2
        with:
          path: ./public

  # Deployment job
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v2
```
- 将更改提交到本地版本库, 并附上类似 "Add workflow"的提交信息, 然后推送到 GitHub
- 回到 GitHub 页面, 你会看到一个进度条, 该进度条显示了 GitHub Actions 的工作流程的进度. 当工作流程完成时, 你会看到一个绿色的复选标记, 表示工作流程已成功完成

这个时候你就能通过 `https://<username>.github.io/<repository-name>/` 来访问你的个人主页啦～
