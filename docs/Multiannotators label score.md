## Multiannotators scenario
[original diff](https://github.com/verandah/jiayi/commit/5bb0329352eec5c03268f9adac235bcb35141739) \
[diff](https://github.com/verandah/jiayi/blob/master/cleanlab/rank.py#L485)<-Several updates was added, the newest version starts from line 483 in rank.py
###
For any given input labels matrix and predicted probabilities matrix,  the [`get_multiannotator_label_quality_scores`](https://github.com/verandah/jiayi/blob/master/cleanlab/rank.py#L681) does the following:
1. call `vote2score_2d`, which will call`get_quality_score` to compute the **label quality score** for each label.
2. call `vote2freq_2d` to compute **label agreements** for each label.
3. call `get_label_weighted_score` to aggregate **label agreements** and **label quality score** to get **label weighted score**  for each label.
4. Choose the label with highset **label weighted score** as **chosen label**.
5. Choose the highest agreement as **sample agreement**.
6. call `get_overall_score_2d` to compute the overall score.
7. combine **label quality score**, **sample agreement** and **overall score** to generate report (in pd frame format) as required.

### label agreement
`vote2freq_2d`

The text-book formula to compute agreement
$$\text{agr}(label=k) = \frac{\text{number}(\text{label}=k)}{\text{number}\text{(all labels)}}$$

### sample agreement
sample agreement is the maximum label agreement over all the annotators' labels.
$$\max_k(\text{agr}(\text{label}=k))$$
It reflects to what extent the annotators reach agreement on a sample. I don't concern about the tie since it doesn't affect the level of agreement (Tie labels does affect the overall score).

#### Further discussion on agreement
We can put an extra weight for each annotator, which reflects how much we trust his/her in the annotation process. We might trust experts more than random online annotators. Then the agreement will be:
$$\text{agr}(\text{label}=k) = \frac{\sum_{\text{label}=k} w_i}{\sum_{i \in M} w_i}$$

### label weighted score for each label in a sample
[`get_label_weighted_score`](https://github.com/verandah/jiayi/commit/5bb0329352eec5c03268f9adac235bcb35141739#diff-44f75f931f626f8a5e428e4604c79c3f6a7732c958522b32e24ee67130b8e47fR666)

The label quality scores and label agreements come from model and human respectively. That is to say, for each sample, we have ratings from two independent systems. I suggest that we use a common way to aggregate it. One option is to take **the weighted arithmetic mean**
$$(1-\alpha) \cdot \text{LQS} + \alpha \cdot \text{agr} $$
Another option is to take **the weighted harmonic mean**
$$\frac{(1+\beta^2)(\text{LQS}\cdot\text{agr})}{\beta^2 \cdot \text{LQS}+ \text{agr}}$$
The $\alpha$ (or $\beta$) gives flexibility to balance the rating from model and human. $\alpha$ indicates how much weight we rely on the annotators.
+ $\alpha=1$ means that we only trust the annotator.
+ $\alpha = 0$ implies that we only trust the model prediction.
+ By default we set $\alpha=0.5$ (equal weight).

Similar to $\alpha$, the larger the $\beta$, the more we trust on annotators. By default we set $\beta = 1$ (equal weight)

### chosen label
Chosen label is the label has the highest label weighted score.
$$\text{chosen label} = argmax_{label \in [K]}(\text{label\_ weighted\_scores(label)}) $$
If we set $\alpha$ as $1$ and take the weighted arithmetic mean method, the chosen lable is the label which has the highest label quality score. On the other hand, if we set $\alpha=0$ this label is the consensus label with the majority votes (i.e. chosen label could be different from consensus label)

### overall score for a sample
[`get_overall_score_2d`](https://github.com/verandah/jiayi/commit/5bb0329352eec5c03268f9adac235bcb35141739#diff-44f75f931f626f8a5e428e4604c79c3f6a7732c958522b32e24ee67130b8e47fR629)

For one sample, consider the label weighted scores for all potential labels. This score is an aggregation of rating from model and huamn. Intuitively, a sample is well-labeled if it 'outstands' in the annotators' labels. 'Outstanding' can be interpreted as a high rating score (the close to 1 the better) and a clear gap from the other labels. In short, a well-labelled sample has a label with high weighted score and low ambiguity. Define outstanding ratio be
$$oR = \frac{\text{highest score} - \text{2nd highest score}}{\text{1 - highest score}} \in [0, \infty]$$
(Note: the score refers to the label weighted score)
+ When the chosen label is perfect (i.e. it has highest score = 1, it has largest gap from 2nd one), the outstanding ratio is $\frac{1-0}{1-1} =\infty$
+ When the chosen label is ambiguous (i.e. its weighted score has a very small gap from 2nd highest one), the outstanding ratio is small ($\approx 0$)
+ When a mixed situation happens, for example, $\text{1 - highest score} \approx \text{highest score} - \text{2nd highest score}$. The outstanding ratio approximates $1$.

To scale it to $[0,1]$, we apply transformation $f(x) = \frac{2}{\pi} \arctan (wx)$, $w \in [1, +\infty]$ is a parameter to give flexibility to the scaling. By default, we can take $w=1$. The outstanidng score (overall score) is defined as:
$$\text{overall\_score}_w = \frac{2}{\pi} \arctan (w*oR)$$

