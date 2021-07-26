---
layout: post
title: mmdetectionのFocalLossの演算が合っているのか気になったので、確認してみた。
created: 2021-07-27 02:51:57
updated: 2021-07-27 02:51:55
published: 
category: paper
tags:
- object detection
- region proposal
- CVPR
---

最近お世話になっているmmdetectionさんの[`FocalLoss`](https://github.com/open-mmlab/mmdetection/blob/31b3a58036de3a095837e21e810051451155821b/mmdet/models/losses/focal_loss.py#L106)クラスが気になったので、計算過程を確認してみたという内容です。
FocalLossクラスの中では、PyTorchベースとCUDAベースで演算するのかで演算方法が分岐するんですが、今回はPyTorch実装で確認していきます。  
具体的には、[`py_sigmoid_focal_loss()`](https://github.com/open-mmlab/mmdetection/blob/31b3a58036de3a095837e21e810051451155821b/mmdet/models/losses/focal_loss.py#L11)を確認していきます。

---

# Focal Lossってなに？

Focal Lossのmmdetection実装を追う前に、Focal Lossを軽くさらっていきます。

こんなこと今更いらないかもしれませんが、PositiveとNegativeサンプルの不均衡問題に対処したRetinaNetで利用されている損失関数です。ちなみに、筆者はRetinaNetの論文をちゃんと読んでいません。テキトー解説です。いつか読みます。  

Focal Lossは、ソフトなバギングを行っているイメージです。バギングは、認識が難しいサンプル（Hard Example）を取り出して集中的に学習していくハードなサンプリング方法ですが、Focal Lossはソフトなサンプリングを行います。  
具体的には、サンプルを取る or 取らないではなく、Hard Exampleには強い重みを、Easy Exampleには軽い重みをつけて損失関数を作ることで、Hard Exampleの勾配を強くすることを行います。

Focal Lossの大元はバイナリクロスエントロピーで、その正例・負例に対してConfに基づいた重みがつけられます。

バイナリクロスエントロピーは、次式で表されます。\\( p \\)はConfの確率を表しており、0~1です。バイナリクロスエントロピーを使うのは、正例が負例と判定されたConf値を損失に加えるような意図があると思われます。  

$$
\mathrm{CE}(p, y)= \begin{cases}-\log (p) & \text { if } y=1 \\ -\log (1-p) & \text { otherwise }\end{cases}
$$

Focal Lossは、これに対して、Conf \\( p \\)が大きければ重みを下げるような係数をかけます。\\( \alpha \\)は各クラスのスケーリング係数です。そんなに影響ないみたいですが、一応これで調整すると少し性能が改善するらしいです。  

$$
\mathrm{FL}=\begin{cases} - \alpha \left(1-p\right)^{\gamma} \log \left(p\right) & \text { if } y = 1 \\ 
-(1 - \alpha )p^{\gamma} \log \left(1 - p \right) & \text {otherwise}
\end{cases}
$$

\\( p \\)は、ロジットそのままでも、まあいいっちゃいいかもしれないんですが、安定化のためにSigmoid関数を一旦挟むのが良いと論文中で述べられており、mmdetectionでもTensorFlowでもSigmoidを一旦挟むようです。

# mmdetectionのFocal Loss実装(PyTorch)

さて、mmdetectionのFocal Loss実装を見ていきましょう。
全体は以下の構造になっています。各クラスの所属確率とGTのターゲットの２つのテンソルを受け取ります。

```python
# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss
```

筆者は、次の部分が最初よくわかりませんでした。

```python
pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
```

\\( p_t \\)というのは、Focal Lossの論文中で使われている定義です。筆者はこの表現は好きではないので、上記の定義では省略せずにFocal Lossを定義しました。ただ、論文中では次式なんですが...

$$
p_{\mathrm{t}}= \begin{cases}p & \text { if } y=1 \\ 1-p & \text { otherwise }\end{cases}
$$

コードを見ると逆になっています。ここで混乱。

$$
p_{\mathrm{t}}= \begin{cases} 1-p & \text { if } y=1 \\ p & \text { otherwise }\end{cases}
$$

まあいいやと思い、とりあえず、コードの定義に従って行きましょう。
次に、`focal_weight`のところですが、これは展開すると次式で表せます。

$$
\text{focal\_weight}=\begin{cases} \alpha \left(1-p\right)^{\gamma} & \text { if } y = 1 \\ 
(1 - \alpha )p^{\gamma} & \text {otherwise}
\end{cases}
$$

最後に、`loss`のところですが、`F.binary_cross_entropy_with_logits(pred, target, reduction='none')`を使って、\\( \log \left( p \right) , \log \left(1 - p \right) \\)の部分を作っており、それらを\\( y = 1 \\)同士、\\( \text {otherwise} \\)同士で掛け合わせると、Focal Lossが出来上がります。（-を忘れないでネ）

一応、ちゃんとFocal Lossができているようです。  

# もっと分かりやすくならない？

なります。こんな感じで\\( y = 1 \\)と\\( \text {otherwise} \\)で別々に実装できます。

```python
    # <省略>
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)

    pos_loss = - alpha * ((1 - pred_sigmoid) ** gamma) * torch.log(pred_sigmoid) * target
    neg_loss = - (1 - alpha) * (pred_sigmoid ** gamma) * torch.log(1 - pred_sigmoid) * (1 - target)

    loss = pos_less + neg_less

    # <省略>
```

これでやっと自分のなかで腑に落ちた感じがします。ふぅ。



