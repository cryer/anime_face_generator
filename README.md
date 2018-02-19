# anime_face_generator

  Use DCGAN to generate anime faces.If you want to know more about GAN and DCGAN,goto my another repo
[some kinds of GANs](https://github.com/cryer/GAN).

This repo's model set is almost a copy of DCGAN.py in upper url.Just because this kind of model set is 
provided in author's paper and it's definitly useful.You can also transfer it to other tasks.

## Datasets

You can use spider to download animefaces online by yourself,but I got it from 
[baidu cloud disk](https://pan.baidu.com/s/1eSifHcA) with extracted code `g5qa` created by [何之源](https://zhuanlan.zhihu.com/p/24767059)

## Note

It is a simple DCGAN implementation to generate much more complicated images than mnist or cifar10,
so results may not satisfy you.However,many superme algorithm have already shown great success in this
anime face generating field,even people can not distinguish them.

For example,this [paper](https://makegirlsmoe.github.io/assets/pdf/technical_report.pdf) is very nice.
They also provide [Online generate](http://make.girls.moe/#/),you can choose your favorite anime attributes
,they will generate anime girls as you wish.

## Code

I tried my best to make my code simpler.Main code only a few lines.I believe it will make sense.

## Train

```
pyhon train.py
```

It really takes some time to run the code,because datasets is large and each image is also not small.
By the way,I have ran out of memory several times T T

## Results

At beginning:

![](https://github.com/cryer/anime_face_generator/raw/master/image/2_200.png)

In the end:

![](https://github.com/cryer/anime_face_generator/raw/master/image/fake_mnist1.png)

![](https://github.com/cryer/anime_face_generator/raw/master/image/fake_mnist2.png)

![](https://github.com/cryer/anime_face_generator/raw/master/image/fake_mnist3.png)

![](https://github.com/cryer/anime_face_generator/raw/master/image/fake_mnist4.png)
