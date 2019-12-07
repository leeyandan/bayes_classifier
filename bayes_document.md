# 贝叶斯分类器推导过程
> 参考自我在课堂上学何老师的知识
--------
## 1. 贝叶斯公式
+ 贝叶斯公式的一般形式

$$
P(A|B)=\frac{P(A,B)}{P(B)}\Leftrightarrow P(A,B)=P(B)P(A|B)=P(A)P(B|A) \tag{1.1}
$$
&emsp;&emsp; 将$P(A,B)$ 用 $P(A)P(B|A)$ 替代得

$$
P(A|B)=\frac{P(A)P(B|A)}{P(B)} \tag{1.2}
$$
&emsp;&emsp; 这个时候你可能还是一头雾水，这个和我们分类有什么问题呢？你只用记得上面是对的就可以了。<br>
&emsp;&emsp; 接下来会“套用”到分类问题上。
+ 对于分类问题的贝叶斯公式<br>
假设有$k$个类 $\{w_1, w_2, \cdots, w_K\}$，属性值有$D$维$\vec{x}=\{x^{(1)},x^{(1)},\cdots,x^{(D)}\}$，在已知属性$\vec{x}$ 下属于某类的概率为：

$$
P(w_k|\ \vec{x}) = \frac{P(w_k)P(\vec{x}|w_k)}{p(\vec{x})} \tag{1.3}
$$
&emsp;&emsp;&emsp;其中，$P(w_k)$称为先验概率，就是一个类占比多少的概率。$P(\vec{x}|w_k)$ 称为类条件概率，<br>
&emsp;&emsp;&emsp;是在已知某一类下有该属性的概率。<br>
&emsp;&emsp;&emsp;所以，我们基于最大后验概率原则可以预测样本是属于哪一类的:

$$
predict\_class = argmax\ P(w_k|\ \vec{x})  \qquad k=\{1,2,\cdots,K\}  \tag{1.4}
$$
&emsp;&emsp;&emsp;在类间对比概率的时候，由于大家分母都有相同的 $p(\vec{x})$ 可以消去。

$$
P(w_k|\ \vec{x}) = \frac{P(w_k)P(\vec{x}|w_k)}{p(\vec{x})}\propto P(w_k)P(\vec{x}|w_k) \tag{1.5}
$$
&emsp;&emsp;&emsp; **所以目前的公式的重要在于 $P(\vec{x}|w_k)$ 如何计算的问题：**

$$
P(\vec{x}|w_k) = P(x_1,x_2,\cdots,x_D\ |\ w_k) \tag{1.6}
$$
## 2. 朴素贝叶斯分类器
+ 朴素贝叶斯朴素就朴素在它**假设了属性之间是独立**的，它们之间是概率独立的。如下：

$$
P(\vec{x}|w_k) = P(x_1,x_2,\cdots,x_D\ |\ w_k) = P(x_1|w_k)P(x_2|w_k)\cdots P(x_D|w_k) \tag{2.1}
$$
+ 概率 $P(x_i|w_k)$ 计算问题
    + 离散型随机变量<br>
      计算$w_k$的总次数N，在$w_k$中$x_i$出现的次数n。则$P(x_i|w_k)=\frac{n}{N}$
    + 连续型随机变量<br>
    假设服从正态分布
    
    $$
     P(x_i|w_k) = \frac{1}{\sqrt{2\pi}\delta _{ki}}e^{-\frac{1}{2}\left ( \frac{x_i-\mu _{ki}}{\delta _{ki}} \right )^2} \tag{2.2}
    $$
    &emsp;&emsp;&emsp;其中，$\delta _{ki}$ 是第k类第i维的标准差，$\mu _{ki}$ 是第k类第i维的均值。

## 3. 灵活贝叶斯分类器