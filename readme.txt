1.0.1
1.解决了init_modal不生效的问题
2.添加了dqtl部分代码,并能够正常输出实验结果,包括权重以及gan生成图像

1.0.2
1.添加了输出config的功能,并通过openpyxl重写expo_result()
2.增加make_scheduler
3.修改dqtl模型,增加sa模块测试,增加klloss多个loss权重选项

1.0.3
1.gan训练时nohup显示有问题,已修正
2.添加最佳epoch显示
3.添加平衡loss
4.修改色彩表
5.减少了对xlrd, xlwt, xlutils的包依赖
6.加入对train.pretrained的支持
7.网络加入预训练编码解码器,加入load_modal
8.添加make_scheduler里对多个起始学习率低的学习率函数的支持,并且修正之前不起效果的bug
9.加入最新进度保存
10.网络结构更改
麻了,第10条出现巨多问题,这版就这样吧,截止了。

1.0.4
1.更新solver和两阶段solver一致
2.增加了双模swin_t的baseline
3.增加自步学习模块