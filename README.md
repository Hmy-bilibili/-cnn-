前段时间AlphaGo战胜了棋圣李世石，不知道大家是否了解，背后的关键技术是Google旗下DeepMind团队开发的深度强化学习算法。在本图文中，我将带领大家用一种深度强化学习算法，开发一个能够玩Pong这个游戏的AI。
![Image text](https://s3.jpg.cm/2021/09/05/It0OEC.gif)

这个游戏很简单，控制我们的板子不断接住对方的球，并让对手接不到球。
一、环境配置
我们使用Python3配合Pytorch框架和OpenAI的Gym环境。（pytorch相对tensorflow更简洁，Gym则是非常好用的测试环境，其中提供了很多种游戏可以用来测试我们的强化学习算法的效果。）考虑到绝大多数酷友使用windows系统，所以以下步骤都是在win10下执行的。

1)安装anaconda：在官网下载anaconda安装包并安装。（可以参照网上的教程，有很多。）记住安装目录，后面需要用哦。

2)安装Python IDE，我这里用的是Pycharm社区版，百度搜索后进入官网并下载。

3)创建虚拟环境按键盘win+R，在弹出的框中输入cmd打开命令行界面。如果第一步中的anaconda安装正确，我们可以在命令行中使用conda命令创建虚拟环境。
在命令行中输入：conda create -n ai python=3.6并按回车键，在提示输入y/n的时候输入y并按回车，等待命令执行完毕（需要联网）

4)安装需要的python包。
1.在命令行中输入activate ai激活虚拟环境
2.输入以下命令安装pytorch（CPU版本，本次实验未使用卷积层，GPU加速意义不大。），速度可能很慢。。有条件的同学可以考虑用飞机场。。。

[pip install](https://www.coolapk.com/link?url=https%3A%2F%2Fdownload.pytorch.org%2Fwhl%2Fcpu%2Ftorch-1.0.1-cp36-cp36m-win_amd64.whl
)

输入以下命令安装openai-gym:
[pip install gym -i](https://pypi.tuna.tsinghua.edu.cn/simple/)

输入以下命令安装cloudpickle:
[pip install cloudpickle](https://pypi.tuna.tsinghua.edu.cn/simple/)

输入以下命令安装atari-py:
[pip install --no-index -f atari_py](https://github.com/Kojoley/atari-py/releases)

二、创建项目
打开Pycharm后，我们选择Create New Projec来新建一个项目。在弹出的窗口中可以选择项目存储的位置，这个大家随意选择。在Location下面一行，有一行Project Interpreter，单击左边的三角展开选项，在弹出的选项中，选择Existing interpreter。然后单击右边的齿轮，选择Add local。在弹出的窗口中，左边选择Conda Environment，然后在右面单击那个…的图标，然后选择你的anaconda安装目录下的envs文件夹下的ai文件夹下的python.exe。然后一路点击OK。

点击create后进入coding界面。右击project栏的ai文件夹，选择New->Python file，然后命名为reinforce，点击OK。 
![Image text](https://s3.jpg.cm/2021/09/05/It2P0u.jpg)

三、算法原理
这篇教程中主要使用了强化学习算法。强化学习算法大体可分为两类：value-based和policy-based。这里我们使用了一种最简单的policy-based算法，名叫Reinforce（中文为强化）。使用这个算法原因是我认为这个算法和人类学习的过程很相似。关于这个算法的具体推导比较复杂，涉及到一些微积分、概率和随机过程的知识，这里就不做详细说明了，只是大体讲一下这个算法的直观理解。先考虑一下人类玩游戏的过程，当我们观察到游戏画面时，我们的大脑会对当前的画面做出反应（即执行一个动作，比如遇到怪物时选择躲避或者攻击），如果我们选择这个动作取得了一个好的结果（比如过关了），那么下次遇到类似的场景时，我们还会选择这个动作；反之，如果我们执行一个动作后，游戏失败了，那下次我们就不会在执行这个动作了。这是一个试错的过程。假设我们最后学会了如何通关一款游戏，我们在每个场景下，都会执行一个我们认为比较和的动作，这就相当于我们学会了一个映射，一个从场景到一个动作的映射，也可以认为是一个特殊的“函数”。我们可以把这个函数称为一个策略（Policy）。Reinforce算法就是想要得到这样一个策略函数，给这个策略输入一张游戏图像，策略输出一个对应当前图像的最好的动作。那这个函数是一个很复杂的函数，怎么用表达式来表达呢？这时就需要借助强大的“人工神经网络”，通过不断的试错，“学习”一个最优的策略。
关于神经网络的知识，可以参考:[查看链接](https://zybuluo.com/hanbingtao/note/433855)。
关于深度学习算法的知识，可以参考：[查看链接](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)

四、敲代码
这里就直接上代码啦。这里我们对每部分代码做一些粗略的讲解。
关于pWrapper这个类，是对原始的gym环境进行了包装，将观察到的游戏图像做了一些预处理。
关于make_envs这个函数，是利用了一个工具类SubProcVecEnv，在多个子进程中并行的开启多个环境进行交互，以提高算法的效率，同时还有利于算法探索到更好的策略（类似于A2C的做法）。开启的环境个数可以根据自己机器的配置来设置，个人电脑推荐4-8个。
关于Policy类，这个类继承了pytorch.nn.Module，定义了一个由多个全连接层组成的神经网络，即“策略”函数。
Select_action函数给我们的策略传入一张游戏图片，策略则返回在该状态下应该执行的动作。
Finish_episode函数则负责对玩完的一局游戏进行总结。判断执行过的动作的好坏，并根据游戏结果，让策略函数朝着更好的方向前进。
Main函数控制程序总流程，利用我们的策略不断的玩游戏，每玩完一局就进行经验总结，来得到一个更好的策略，并不断重复这个过程。
我们的程序还会将策略保存下来。
建议大家直接用我分享给大家的代码，是我调试好的，能够在windows下直接运行的。下面的代码可能会遇到些问题。

[图片1](https://s3.jpg.cm/2021/09/05/It2Q6p.jpg)

[图片2](https://s3.jpg.cm/2021/09/05/It2noX.jpg)

[图片3](https://s3.jpg.cm/2021/09/05/It25xD.jpg)


将这两个文件直接拷贝到我们建立的pycharm项目中，并在左侧双击reinforce.py，然后在代码编辑栏右击，选择run reinforce，就可以运行程序了。

