---
layout: '@/templates/BasePost.astro'
title: 如何使用docker在windows上愉悦运行ROS/ROS2
description: docker、ros
pubDate: 2024-11-21T00:00:00Z
imgSrc: '/assets/images/ros.jpeg'
imgAlt: 'Image post 5'
---

友情链接[this page](../sixth-post/).
## 1 安装docker，vscode 和VcXsrv
请自己去网上找吧（
Docker: Accelerated, Containerized Application Development
Visual Studio Code – Code Editing. Redefined
https://github.com/ArcticaProject/vcxsrv

## 2 一些docker的概念和命令
## 2.1 Image和Containers
Image是一个打包好的包含系统，环境。用户和别的一堆东西的镜像。非常好的是这个玩意在构建好之后就是只读的，
而镜像在运行之后就能在其上方生成一个容器，镜像和容器就如同面向对象的类和实例。对于此例而言，我们要一个ubuntu18系统装好ros的镜像来部署。
## 2.2 下载镜像，生成容器
首先我们可以在Explore Docker’s Container Image Repository | Docker Hub上面找到各种各样的镜像文件，然后就能部署了。

此例中我们使用这个大佬的镜像。

    docker run -dit --name=ros_melodic -v d:/home/d  -e DISPLAY=host.docker.internal:0.0 fishros2/ros:melodic-desktop-full

使用run 命令来下载镜像并且直接运行容器

当然也可以pull命令下下来再使用run

    docker pull fishros2/ros:melodic-desktop-full
    docker run ros_melodic

## 3 愉快使用
先愉快启动容器

    docker run ros_melodic
    /bin/bash

然后你就能用roscore啦
进入已经启动的容器我们使用 ps -a

    PS C:\Users\m1573> docker ps -a
    CONTAINER ID   IMAGE                               COMMAND                  CREATED       STATUS                   PORTS     NAMES
    be7c44aec2b9   fishros2/ros:melodic-desktop-full   "/bin/bash"              5 hours ago   Exited (0) 5 hours ago             pensive_sinoussi
    9a392f29a964   fishros2/ros:melodic-desktop-full   "/bin/bash"              5 hours ago   Up 51 minutes
    ros_melodic
    3545d3e880fb   fishros2/ros:humble-desktop         "/bin/bash /scripts/…"   6 hours ago   Up 51 minutes
    d2lros2humble

记住container id，使用attach指令来进入容器

    PS C:\Users\m1573> docker attach 9a392f29a964
    root@9a392f29a964:/#

gui通过VcXsrv来实现
打开xlaunch一路下一步直到这里选择最后的选项
再打开vscode安装Dev Containers这个插件lol
再在左侧远程里面就能看到你的容器了

