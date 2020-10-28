![frontespizio](frontespizio.jpg)

------



**Table of contents**

[TOC]

# Introduction

> *“It is the time you have wasted for your rose that makes your rose so important.”*  - Antoine de Saint-Exupery

Imbalanced learning refers to a classification problems where the distribution of the target variable is extremely skewed: some classes are more frequent than others. Common examples of such problems are churn, fraud and anomaly detection and clinical data, when one of the classes is rare because problematic, costly,  unethical, dangerous to produce, or unexpected. Class unbalancing, specified as the proportion in the number of samples in different classes, can reach values in the orders of $10^2\div 10^4 : 1$ and up to $10^5:1$ [^Provost, 2001]

![image-20201013124846482](image-20201013124846482.png)

![image-20201013124941427](image-20201013124941427.png)

*Fig. 1 Examples of synthetic balanced and imbalanced datasets.*

Most real datasets does not have exactly equal number of instances in each class, and while a small difference seldom matters, a heavy imbalance can quickly become a bottleneck in model training. Most learning methods have been conceived to identify the classification rules that better fits the data by some global accuracy criteria. Their target is to minimize the global error, that may not be influenced enough from the minority class. Some methods, like the broadly used logistic regression, is more vulnerable to this effect, but even non-parametric methods, like trees, and association rules, are not immune from this effect. For example tree generated from imbalanced datasets will have an high accuracy on the prevalent class and a very low precision in identifying the rare event. It appears evident how things become worse when the minority class is the event of interest, like a positive diagnosis of cancer in a patient.

A brief description of the sections of this work:

**More data, less data.** The most heard sentence in machine learning community is "You need more data!". Still, a large dataset might indeed expose a different, and perhaps more balanced perspective on the classes: more minority examples can indeed be useful. Other strategies may include considering more than once one or more minority samples. Chapter 1 will review the bibliography about solution for this  problem, giving a view over cost-sensitive learning and different oversampling and undersampling methods, their advantages and disadvantage.

**Random Over Sampling Examples** In chapter 2 we will focus on one of these techniques, henceforth named just by its acronym ROSE, that propose a smart, albeit simple, way to generate new data from existing ones.

**The Accuracy Paradox**. To assess the performance of a solution a metric is needed. When a class represent almost the totality of a dataset, a learning algorithm can achieve a good accuracy by classifying every test sample as belonging to the majority class. To avoid this problem, different metrics has been developed to assess the real model utility and assessing capabilities. Chapter 3 will review available metrics that can be used to effectively evaluate performance of resampling methods.

**A method is as good as it is available.** Rose is already available since 2014 with the R package ROSE, and it proved to be successful in many situations[^Menardi, 2014]. To make Rose available to a larger community of data scientists is one of the main goals of this work, and it involved incorporating it in the most used Python machine learning package, `scikit-learn` , and in particular in its sub-project `imbalanced-learn`. We will overview the development methods, CI/CD, software testing and documentation, in chapter 4.

**But is it good?** In chapter 5 will set up a wide testing framework for evaluating performance of resampling methods over 27 different famous standard datasets commonly used for classification problems. Different supervised models has been trained and tested on imbalanced and balanced data, and their performance reported. But toy datasets are usually easier to balance. In chapter 6 we used ROSE to dramatically improve classification capabilities of some models for the analysis of a real-world dataset, with the aim of forecasting the economic outcome of small firms.

**How can I use it?** In appendix 1 we will show code snippets, use cases and links to repositories that will facilitate user's experience with ROSE, and guarantee repeatability of all experiments included in the present world.

# Imbalanced learning

## Imbalanced dataset problem

Despite the fact that in literature most imbalanced learning problems are traditionally referred to binary datasets, real world datasets can often be multiclass, as in microarray research[^Yu, 2013], protein classification [^Zhao, 2008], medical diagnostics[^Cerf, 2013], activity recognition[^Gao, 2014], target detection[^Razakarivony, 2016], and video mining[^Gao2, 2014]. Extending imbalanced classification to multi-class scenarios is a natural path, then. As the number of classes increases, so does the challenge of representing the whole problem space accurately, and the need for taking into account the presence of multi-minority and multi-majority classes [^Wang, 2010]. 

In many problem, imbalancing is intrinsically tied to the nature of the data, and not due to lack of sampling, bias, or other sampling errors. In other cases no enough samples of a specific class exists at all.

Most learning methods' loss functions are supposed to be minimized globally, under the assumptions that all class have the same weight. When data are imbalanced, the learning process  often achieves this objective by focusing on majority class, leading to bad performance [^Ganganwar, 2012], with higher errors on minority classes. 

The lack of model effectiveness in prediction of rare classes has been deeply discussed in literature. Both parametric and non parametric methods appear to be sensitive to imbalancing. As an example in logistic regression, one of the most used for binary classification, this effect depends from an underestimation of conditional probability of the rare class [^King, 2001],[^Menardi, 2014]. 

Even the more flexible non-parametric methods, like classification trees and association rules are immune from the effect of asymmetric class distribution. Trees, for example, are being grown finding the recursive divisions of the parameter space that maximize the impurity reduction. The imbalance found in the dataset will be often mirrored in the imbalance of the accuracy over different classes [^Menardi, 2014], [^Chawla, 2003]. Even association rules, being selected by their supports, tend to underperform[^Gue, 2003],[^Ndour, 2012].

## Treating imbalanced datasets

Many solution has been advanced to treat imbalanced data problems. Most fall in one of the following two approaches: using cost-sensitive learning models, and resampling the data.

### Cost-sensitive learning

Cost sensitive learning is an umbrella term for algorithms in whose objective function it is possible to assign a different cost to misclassification of different classes. An intuitive example of this approach can be imagined when talking about a binary clinical cancer test: a false positive will lead to some extra exam, while a false negative will probably cost a life. The most logical decision is to estimate the relationship between these costs, and assign a larger (*hopefully, much larger*) cost to a false negative.

For multiclass data, a cost matrix $\mathbf C$ is computed, where $\mathbf C_{i,j}$ will be the cost of misclassifying a sample belonging to the class $j$ as it were belonging to the class $i$ [^Liu, 2006], [^Zhou, 2010]. Note that introducing a different loss function to deal with different costs in some cases implies modifying the original algorithm.

### Resampling

A different, alternative approach against imbalancing can be tried by preprocessing the data, instead of modifying the learning rules, using sampling methods. This approach has consistently proven effective, with different degrees [^He, 2009],[^Weiss, 2003]. Different resampling methods has been proposed, falling in two categories:

* undersampling methods, where majority class samples are being randomly discarded to remove imbalancing, at the price of sample size, in a non-heuristic way;
* oversampling methods, where different techniques can be used to generate new minority samples from the existing ones. The following sub-chapters (1.1.3 and 1.1.4) gives an overview of these methods.

Oversampling and undersampling presents different pro and cons, leading to the need of an empirical comparison between different methodologies.

| Methods       | Pros                                            | Cons                                                |
| ------------- | ----------------------------------------------- | --------------------------------------------------- |
| undersampling | faster learning                                 | loss of sample size                                 |
| oversampling  | slower learning<br />higher computational costs | introduction of artifacts<br />possible overfitting |
|               |                                                 |                                                     |

Despite those problems, resampling is more commonly used than cost-sensitive learning, that is not supported for all learning methods.

### Undersampling strategies

Undersampling reduces the size of majority class to avoid imbalancing. In this paragraph we will provide an overview over commonly available undersampling strategies.

* **Random UnderSampler (RUS)**: it works by simply choosing random samples from the majority class.
* **Condensed NN**: [^Hart, 1968] uses a 1-nearest neighbor rule to iteratively decide if a sample should be removed or not. It is sensitive to noise and will generate noisy samples. 
* **One Sided Selection** [^Kubat, 1997]and **Tomek Links** instead tends to remove noisy samples. <img src="sphx_glr_plot_illustration_tomek_links_0011.png" alt="" style="zoom:70%;" />
  *Fig 2: Tomek link strategy for undersampling. Tomek links nodes, classified as noise, can be removed.*

* **Edited NN** and **Repeated Edited NN** [^Wilson, 1972]  apply (respectively one of more times) a nearest-neighbors algorithm and “edit” the dataset by removing samples which do not agree “enough” with their neighborhood. For each sample in the class to be under-sampled, the nearest-neighbors are computed and if the selection criterion is not fulfilled, the sample is removed. The criterium can be based on majority, or totality of nearest neighbors belonging to the same class of the inspected sample to be kept in the dataset.
* **All KNN** is another iterative process that does the same of the latter, but incrementing at each iteration the number of  considered neighbors.
* **Near Miss** [^Mani, 2003] is a collection of three different algorithms that respectively:
  * selects the majority samples for which the average distance to the $k$ *nearest* neighbors of the minority class is the *smallest*, or
  * selects the majority samples for which the average distance to the $k$ *farthest* neighbors of the minority class is the *smallest*, or
  * first keel the $M$-nearest neighbors are kept, then, the majority samples selected are the one for which the average distance from the $k$ nearest neighbors is the  *largest*.
* **Neighborhood Cleaning Rule**[^Laurikkala, 2001] focuses on cleaning the data without condensing them.
* **Instance Hardness Threshold** [^Smith, 2014] trains any classifier on the data, and the samples with lower probabilities are removed from the dataset. It is not guaranteed to output a balanced dataset, though.

### Oversampling and synthetic data generations

In this section we present the most commonly used oversampling techniques and their variants:

* Synthetic Minority Oversampling TEchnique (SMOTE) based
* ADAptative SYNthetic sampling (ADASYN).

#### SMOTE based methods

**SMOTE**[^Chawla, 2002] is a class of resampling algorithms that use the following approach:

* a random sample from the minority class is chosen
* his $k$-neighbors are found (default k=5)
* lines are drawn from the original sample to the neighbors
* new examples are drawn randomly along these lines, with $x_{new} =x_i +\lambda * (x_{nn}-x_i)$, where $\lambda$ is drawn from $Uniform(0,1)$, or other distributions.

<img src="1*6UFpLFl59O9e3e38ffTXJQ.png" alt="Image for post" style="zoom: 50%;" />

*Fig. 3 : SMOTE general resampling concept. New samples are generated along the lines connecting minority sample, with different distributions and strategies.*

There are many variants of SMOTE that has been developed to improve its performance. 

**Borderline SMOTE** [^Han, 2005] will classify each sample $x_i$ to be:

1. *noise*, when all k-neighbors are of a different class from $x_i$ 
1. *in danger*, when at least half of the neighbors belongs to the same class
1. *safe*, when all neighbors belongs to the same class.

The algorithm will then use "*in danger*" samples to generate new samples, with the same procedure of SMOTE.

**K-Means SMOTE** [^Last, ArXiV] uses a K-Means clustering method before to apply SMOTE. The clustering will group samples together and generate new samples depending of the cluster density.

**SMOTENC** [^Chawla, 2002] slightly change the way a new sample is generated by performing something specific for the categorical features. In fact, the categories of a new generated sample are decided by picking the most frequent category of the nearest neighbors present during the generation.

**SVMSMOTE** [^Nguyen, 2009] fits a Support Vector Classifier to find support vectors and generate samples considering them. Tuning the $C$ parameter of the SVM classifier allows to select more or less support vectors.

#### ADASYN

ADASYN [^He, 2008] works similarly to the regular SMOTE. However, the number of samples generated for each $x_i$ is proportional to the number of samples which are not from the same class than $x_i$ in a given neighborhood. Therefore, more samples will be generated in the area where the nearest neighbor rule is not respected.

#### Combination and Ensemble methods

Combinations of different methods can be used efficiently. SMOTE based methods can generate noise when generating point between marginal outliers and inliers. After the resampling this issue can be solved by cleaning the space resulting from oversampling.

Two methods used for this purpose are:

* **Tomek's links**: [^Batista, 2004] an undersampling technique used to remove unwanted overlaps between classes, where majority class links are removed until minimally-distanced neighbors pairs belong to the same class. Two instances form a Tomek's link if:

  * one of them is noise (*see Borderline SMOTE definition of noise)*, or
  * both are near a border

  In other words, if they are each other's closest neighbor, and of different classes.

* **Edited nearest-neighbors** [^Batista, 2003] uses asymptotic convergence properties of nearest neighbor rules that use an editing procedure to reduce the number of preclassified samples and to improve performance [^WIlson, 1972]

Ensemble methods can be used generate undersampled subsets of many different oversampled datasets, or by bagging different undersamplers. Additionally, pipelines can be assembled, to chain different methods.

# Random over-sampling examples (ROSE)

ROSE[^Menardi, 2014] provides a different methodology to deal with imbalanced samples. As its alternatives do, it alters the distribution of the classes, using the following solution, based on the generation of new artificial data from the classes, according to a smoothed bootstrap approach [^Efron, 1993]. It focuses on $\mathcal X$ domains included in $\Bbb R^d$ , that is $P(\pmb x)=f(\pmb x)$, a probability density function on $\mathcal X$. We consider that $n_j\lt n $ is the size of $\mathcal Y_j,j=0,1$. The ROSE procedure to generate a single new artificial sample consists in drawing a sample from $K_{\mathbf H_j}(\bullet,\mathbf x_i )$, with $K_{\mathbf H_j}$ a probability distribution centered at $\mathbf x_i $, and $\mathbf H_j$ a matrix of scale parameters, determining the width of the extracted sample neighborhood.

Usually $\mathbf H_j$ is chosen in the set of unimodal symmetric distributions. Once a class has been selected,
$$
\label{eq:sampling}
\begin{align}
\hat f(\mathbf x|y=\mathcal Y_j) &= \sum_{i=1}^{n_j}p_i Pr(\mathbf x|\mathbf x_i) \\
&=\sum_{i=1}^{n_j} \frac{1}{n_j}Pr(\mathbf x|\mathbf x_i) \\
&=\sum_{i=1}^{n_j}\frac{1}{n_j} K_{\mathbf H_j}(\mathbf x - \mathbf x_i).
\end{align}
$$

such as, in this framework, the generation of new examples from the class $\mathcal Y_j$ will correspond to the generation of data from the kernel density estimate of $f(\mathbf X|\mathcal Y_j)$, to generate a new synthetic balanced training set $\mathbf T^*_m$. Usually $m$ is set to the size of majority class, but can be set lower to perform under-sampling. The choice of $K$ and $\mathbf H_j$ was addressed by a large specialized literature on kernel density estimation [^Bowman, 1997]. By letting the elements of  $\mathbf H_j$ to be small or even zero, ROSE collapses to a standard combination of over- and under-sampling.

Apart from enhancing learning, the generation of synthetic examples from an estimate of conditional densities of the classes may aid the estimation of learner accuracy and overcome the limits of both resubstitution and holdout methods. Resampled datasets can be efficiently employed in leave-K-out or bootstrap estimation. 

## Assumptions

ROSE requires that the resampled variables are of numeric type, being impossible to fit a multivariate kernel on unordered categorical variables. This can include variables with limited  numeric support (e.g. $\{0,1\}$, or percentage values. For the latter, problems rise near the extreme values 0 and 100).  In some cases, this problem can be solved using transformations, like taking the logarithm.

Variables belonging to $\mathbb N$ could generate non-integer samples. This problem can be contained by rounding.

Variables belonging to $\mathbb N^+$ or $\mathbb R^+$ domains poses another problem, since samples drawn from the kernel function are not guaranteed to be positive. This particular problem can be contained by a log-transform of the original dataset parameters.

Relatively to our work described in Chapter 4, future development of ROSE will consider the option to extend the class by including type inference or by collecting `numpy.array` and `pandas.DataFrame` dtypes data to dynamically change the random sampling function.

## Kernel methods

Since 90s estimation and learning methods using positive definite kernels have become popular, particularly in machine learning [^Hofmann, 2008]. Real world analysis problems ofter require nonlinear methods to detect the kind of dependencies that allows successful prediction of properties of interests.

The operational use of ROSE requires a prior specification of the  $\mathbf H_j$  matrices. In principle this leads to a criticality, since different choices of the smoothing matrices leads to larger or smaller $K_{\mathbf  H_j}$, namely larger or smaller neighborhoods of the observations from which the synthetic samples are generated. There is a large body of literature on methods of choice of the smoothing parameters [^Silverman, 1986] , [^Bowman, 1997]. The idea beyond these methods is to minimize an optimality criterion, as the asymptotic mean integrated squared error (AMISE).
$$
\begin{equation}
AMISE(h;r)=\frac{R(K^{(r)})}{nh^{2r+1}}+\frac{1}{4}h^4\mu_2^2(K)r(f^{(r+2)})
\end{equation}
$$
Among all possible alternatives, Menardi and Torelli's proposal is to start by using   a Gaussian Kernel with diagonal smoothing matrices $\mathbf H_j = diag(h_1^{(j)},\dots,h_d^{(d)})$ , and minimize AMISE.

This leads to:
$$
\begin{equation}
h_q^{(j)}=\left(\frac{4}{(d+2)n}\right)^{1/(d+4)}\hat\sigma_q^{(j)} (q=1,\dots,d; j=class)
\end{equation}
$$
where $\hat\sigma_q^{(j)}$ is the sample estimate of the standard deviation of the $q$th dimension of the observation belonging to the class $\mathcal Y_j$. Despite the naivety of this approach, authors reports good results, since the only interest is producing a reasonable neighborhood where to sample the new data from, and happens to perform well even if $f(\mathbf x|y=\mathcal Y_j)$ is not $Normal$ , just unimodal.

Choice of $\mathbf H_j$ smoothing matrix gives control on data generation:

In the following image we generated three blobs of examples from multivariate normal distributions. For the three classes, $n$ will be respectively 33, 50 and 170.

![image-20201013111232379](image-20201013111232379.png)

*Fig: 4  Example of unbalanced classes before resampling.*

In the next figure, we used ROSE to rebalance the datasets, and bring $n$ to a total of 300 examples per class.

![image-20201013111251826](image-20201013111251826.png)

*Fig 5 : rebalanced datasets. Original data points in fig. 4 are marked with gray crosses.*

Rose can use a **shrink factor** vector, to shrink kernels independently for each class. The following figure show how, decreasing the shrink factors, new data will be more and more closely clustered around original data points.

![image-20201013111732827](image-20201013111732827.png)

*Fig 6 The same resampling of figure 5, but using different shrink factors: $grey=0.2, orange=0.5, red=1$. Note how new data are more or less tightly clustered around original individual examples.*



# Metrics

Evaluating performance is a critical part of building a machine learning model. In this chapter we will describe some of these tools, and how to choose the best one for our purposes in imbalanced data problems.

## Confusion matrix

Confusion Matrices (henceforth CM) are tables that can be used to describe the performance of a classifier on a test set of data for which true values are known. They are detailed and simple to understand, but does not summarize well the performance.

|     n = 165 | Predicted: NO | Predicted: YES |
| ----------: | :-----------: | :------------: |
|  Actual: No |      50       |       10       |
| Actual: Yes |       5       |      100       |

*Table 1 : An example of a confusion matrix for a binary classifier.*

On the diagonal we find correctly predicted samples (true negatives, or TN, and true positives, or TP), leaving misclassified data on other cells (false positives, or FP, and false negatives, or FN). Confusion matrices can be extended to multiclass classifiers, their size becoming $j\times j$, for classes in $\mathcal Y_j$. Sums over rows and column will describe the total of actual vs predicted predictions. We have seen how secondary indexes can be computed form these values and their ratios.

When describing a model's performance, the simplest yet most common classification metric is its $Accuracy$ , defined as 
$$
Accuracy = (TP + TN)/(TP + TN + FP + FN)
$$
This can be misleading, when the problem uses imbalanced data. Consider a sample with a 100:1 imbalance ratio. Classifying all values as the majority class will gives a $\sim 99\%  Accuracy$ score. [^Mower, 2005]. Different solutions has been proposed to solve this issue. For example, $Balanced Accuracy$ score, defined as 
$$
Balanced\ Accuracy = \frac{\frac{TP}{P}+\frac{TN}{TN+FP}}{2}
$$
can help. Another metric is $Predicted\ positive\ condition\ rate$, defined as
$$
Predicted\ positive\ condition\ rate = \frac{TP+FP}{TP+FP+TN+FN}
$$
which identifies the proportion of the total population correctly identified. Two other commonly used index is $F1$ score and Matthews correlation coefficient. Is this case there is no need of considering a threshold for algorithms that outputs a probability score, instead of the guessed class.

More informative visualizations of model performances can be given not by indices, but by plots, like Receiver Operating Characteristics and $Precision$ vs $Recall$ plots and associated indexes like Area Unde the Curve (AUC), that deserve a dedicated description in the following sub-chapters.

Additional metrics that can be extracted from CM are 

* Cohen's Kappa, that is a measure of how well the classifier performed as compared to how well it would have performed simply by chance. We left it out after bibliography reported unreliable results due to high sensitivity to the distribution of the marginal totals [^Flight, 2015]
* Null Error Rate, that is how often you would have been wrong if you always predicted the majority class. This can be used as a useful baseline metric to compare a classifier against. Still, the Accuracy Paradox tells us that sometimes the best classifier will still have an high error rate than the null error rate.
* $F_1$ score. Since we will use it in our test suite later, we will dedicate next sub-chapter to its description.
* $K$ measure, a theoretically grounded measure that relies on a strong axiomatic base.[^Sebastiani, 2015]
* confusion entropy, a statistical score comparable with Matthews correlation coefficient, treated below.
* Power's informedness and markedness [^Powers, 2011], a couple of interesting alternative metrics that respectively describe how a binary predictor is informed in relation to the opposite condition, and the probability that the predictor correctly marks a specific condition.
* Mattew's correlation Coefficient (MCC), exhaustively treated in a following sub-chapter.

Despite their effectiveness, most of the aforementioned measures does not appear to have achieved such a diffusion in the literature to be considered a solid alternative to MCC and $F_1$ score. They are good single-valued indicators of performance, supported by a strong bibliography, and useful to compare large numbers of tests.

To have a deeper comprehension of a model's performance we used other two plotted tools: Receiving Operator Characteristic and Precision/Recall plots. The following sub-chapters will describe our four tools in depth.

## $F_1$ Score 

Called also F-score or F-measure, is an accuracy metric, calculated from the precision and recall of the test. 


$$
\begin{align}
Precision &= \frac{TP}{TP+FP}\\
Recall&=\frac{TP}{TP+FN}\\
F_1&=2*\frac{precision* recall}{precision+recall}\\
&=\frac{2*TP}{2*TP+FP+FN}\\
\end{align}
$$


It is a particular case of the more general $F_\beta$ score, defined as
$$
\begin{align}
F_\beta&=(1+\beta^2)\frac{precision *recall}{(\beta^2*precision)+recall}
\\
&=\frac{(1+\beta^2)*TP}{(1+\beta^2)*TP+\beta^2*FN+FP}
\end{align}
$$
where recall is considered $\beta$ times as important as precision. a $\beta\gt1$ will increase recall importance, while $0\lt\beta\lt1$ will weight recall lower than precision [^Van Rijsbergen, 1986]. It has recently been criticized as less informative and truthful than Mattews Correlation Coefficient (see below), especially for imbalanced classes.[^Chicco, 2020], and the adoption of new metrics is being suggested, like Informedness (Youden's J statistic)[^Youden, 1950] and Markedness[^Henning, 1989], in fields like biology and linguistics. When using geometric mean instead of harmonic mean of recall and precision it is known as Fowlkes-Mallows index [^Fowlkes, 1983]. In multiclass cases, researchers can employ the $F_1$ micro-macro averaging procedure. [^Tague, 1992]. Micro-averaging puts more emphasis on common labels in the dataset, since it gives each sample the same importance, measuring $F_1$ score of the aggregated contribution of all classes. In macro-averaging the same importance is instead given at every class, regardless of their frequency: a separate $F_1$ score is computed for each class, and then they are averaged. It may overestimate the score for imbalanced problems.



## Matthews correlation coefficient (MCC)

Accuracy and $F_1$ score computed on confusion matrices have been (and still are) among the most popular adopted metrics in binary classification task [^Chicco, 2020]. However these measures can show overoptimistic inflated results, especially on imbalanced datasets. The Mattews correlation coefficient (henceforth, MCC) is instead a more reliable statistical rate which encompass all for confusion matrix categories (TP, FP, TN, FN), proportionally both to the size of positive and negative elements in the dataset. 



$$
\begin{align}
MCC &=  \frac{TP\times TN-FP\times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} \\
&=\sqrt{\frac{\chi^2}{n}}
\end{align}
$$

It derives from Guilford's $\phi$ coefficient[^Guilford, 1954]. Originally developed by Mattews in 1975 for comparing chemical structures, it has been re-proposed by Baldi et al [^Baldi, 2000] as a standard performance metric in the multiclass case, and American Food and Drug Administration (FDA) employed it as main evaluation measure in Microarray II / Sequencing Quality Contro (MAQC/SEQC)[^FDA, 2010]. Nonetheless, it has been reported to suffers from instability in case of imbalanced outcomes. [^Brown, 2018]. Despite the existence of Bayesian based improvements and mathematical workarounds, they have not been adopted yet.

## Receiver Operating Characteristic (ROC) and AUC

A Receiver Operating Characteristic (ROC) curve is a plot that summarizes the performance of a binary classification model on the positive class. The x-axis indicates the False Positive Rate and the y-axis indicates the True Positive Rate. 

![image-20201013125956512](image-20201013125956512.png)

*Fig 7 : Example of ROC curve. AUC (Area under the curve) are shown in the bottom-right legend.*

A ROC gives an intuitive visualization of a classifier performance: the dotted diagonal represent a classifier with no discriminative power, and the more the curve tends to the upper-left corner, the better the classifier is. The area under the curve (AUC) gives a commonly used single-valued index of performance. The threshold is applied to the cut-off point in probability between the positive and negative classes, which by default for any classifier would be set at 0.5, halfway between each outcome (0 and 1) or in some cases, the observed proportions of 1s in the dataset. A trade-off exists between the TP rate an FP rate, such that changing the threshold of classification will change the balance of predictions towards improving the TP rate at the expense of FP rate, or the reverse case.

By evaluating the true positive and false positives for different threshold values, the ROC curve is drawn. An interesting property is that the ROC is unbiased towards model that performs well on the minority class at the expense of the majority class, or vice versa, making it an interesting choice when dealing with imbalanced data.

## Precision-recall plots

Precision-recall plots are a powerful visualization tool to evaluate binary classifiers, closely related to the Receiver Operating Characteristic described in the precedent sub-chapter. It shows the relation between these indexes, at the variation of a threshold

![image-20201013133214575](image-20201013133214575.png)

*Fig 8: Precision-recall plot of a logistic regression model. Bands are confidence interval around values. Queue rate can be seen as the "spam folder" or the inbox of the fraud investigation  desk. This metric describes the percentage of instances that must be  reviewed. If review has a high cost (e.g. fraud prevention) then this  must be minimized with respect to business requirements; if it doesn’t  (e.g. spam filter), this could be optimized to ensure the inbox stays clean.*

# Implementation of ROSE in the `imbalanced-learn` Python package

As we said, a tool is useful only if it available. ROSE has an already available R implementation [^ROSE CRAN]. Despite R being the favored programming language among statistician, Python is quickly rising in popularity, and over the years tens of thousands of packages were offered to help researches in mathematic and statistic fields. We decided to avoid contributing on closed source, expensive or ineffective softwares like MatLab, Excel, Stata, SPSS, and contributing to the community by choosing Python.

As of the date of this writing, the best way to start an argument in a group of data scientists is posing the question "So, Python or R?". This work will stay as far as possible from taking a side in this dilemma, both languages offering many pro and cons, opportunities and flaws.

Instead of the simpler choice of publishing a stand-alone library, we decided to maximize the availability of the code extending the already-available `imbalanced-learn` library[^imblearn], that is a contributor of the well known `scikit-learn` project[^sklearn]. 

This package offers a lot of functionalities, models and mathematical tools, and its main characteristic is the standard API of its classes, that makes them versatile.

Computationally speaking, ROSE resampling is obtained with the following algorithm (pseudocode):

```
define make_samples (X,y,n,h_shrink):
	 n = number of samples to be created
	 p = number of features
	 S = subset of samples randomly selected from X
	 minAMISE = (4/((p+2)*n))**(1/(p+4))
	 vars = variance/convariance matrix of all classes
	 hOPT=h_shrink*minAMISE*vars
	 randoms = multivariate_normal(size=(n,p))
	 rose = randoms*hOPT + S
	 return rose
```

It uses the well known `numpy` library for matrix calculations and sampling.

## `scikit-learn` context

`scikit-learn` (also known as `sklearn`) is a free software machine learning library for Python. It features algorithms for classification, regression, and clustering, including Support vector machines, tree-based models, boosted models, k-means, and DBSCAN. It is built around the famous `numpy` and `scipy` packages, with some routines written in Cython, to improve performance. Some functions are just wrapping other libraries, like LIBSVM or LIBLINEAR.

It was born in 2007 for the Google Summer of Code competition as "SciKit" (SciPy Toolkit), a third party extension of `SciPy`. The original codebase has been rewritten in 2010 by Fabian  Pedregosa, Gael Varoquaux, Alexandre Gramfort and Vincent Michel.

It offers a curated integration with different other Python libraries, like `matplotlib` or `plotly` for plotting, `Pandas` dataframes, `sparse` arrays, `numpy` objects, `scipy`, `tensorflow`, `keras`, and more. Among these API-compatible packages we can find `imbalanced-learn`.

At the moment of this writing, the last version number is 0.23.0 (released in May 2020).

## Test driven development

Development in `imbalanced-learn` packages follows strict guidelines, as explained in the project documentation. Pull requests are to be submitted at https://github.com/scikit-learn-contrib/imbalanced-learn/pulls .

If accepted, they can be marked for review by the sender. With a fast and effective peer review process, they enter the project Continuous Integration / Continuous Deployment process (henceforth, CI/CD).

At the moment there are 1588 test units for `imbalanced-learn`, embracing library compatibility, mathematical correctness, error tolerance and numeric problems.

Most test units already encompass mathematical correctness, but we still added a unit to check if the variance/covariance matrix of resampled data is similar to the one of the original dataset, and some check about correct handling of sparse arrays and Pandas dataframes.

Extra test units verify PEP-8[^PEP8] compliance about linting and code style. Commit history  and review process is available. [^commits] An example of a successful pipeline build can be read at https://lgtm.com/projects/g/scikit-learn-contrib/imbalanced-learn/logs/languages/lang:javascript

## Github and Azure CI/CD

CI/CD is a modern DevOps tool. Code is automatically and continuously pushed to the master branch of the project's repository (we used Github, but Gitlab and other repositories offer the same service). `imbalanced-learn` employs an Azure pipeline.

When a code change is detected, the CI/CD pipeline starts:

* the CI/CD cluster reads a YAML file, with a matrix of configurations: different operative systems, different versions of Python, different version of any used library.
* for every combination, a virtual machine (deployed as Kubernetes containers, in our case) are instantiated.
* at the launch, the pod loads the configuration, and runs all the code test units
* the results of the test units are fed back to the repository
* if all tests are passed, the code can be merged.

Our implementation has been correctly merged, and will be published with the next release of the library. Meanwhile, it can be imported from the `ROSE` branch of the official `imbalanced-learn` repository. All details about test operative systems, library versions and virtual machine setup can be found at https://github.com/scikit-learn-contrib/imbalanced-learn/pull/754/checks  Automatic code review is performed through https://lgtm.com services.

## Documentation

Documentation correctness is integral part of the review process. Functions API are automatically harvested from the code by the `sphinx` documentation library, while theoretical descriptions, application and user guide has been written by the author, and can be found at the official website of the project's documentation, at https://imbalanced-learn.readthedocs.io .

# Empirical analysis

With the aim of obtaining benchmark the real effectiveness of ROSE, a simple test suite has been written, in a Jupyter Notebook.

## Materials & methods

The pipeline evaluates the performance every combination on a grid of models, resampling methods, and parameters.

### Datasets

A total of 27 datasets has been used. All datasets come from the following repositories, and are available for repeatability. All datasets are loaded from Zenodo repository through `imblearn.datasets.fetch_datasets()` API. A detailed description of every dataset can be found in Appendix 3. Additional informations can be found on `imbalanced-learn` repository documentation.

| Short name | Source                                                       | Website                                                  |
| ---------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| UCI        | UCI Machine Learning Repository, University of Californa, School of Information and Computer Science | http://archive.ics.uci.edu/ml                            |
| LIBSVM     | National Taiwan University                                   | https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ |
| KDD        | SIGKDD International Conference on Knowledge Discovery and Data Mining | https://www.biostat.wisc.edu/~craven/kddcup/index.html   |

*Table 2 : data sources for empirical testing.*

| ID   | Name           | Source repository | Target         | Shape(n,p)  | imbalance<br />ratio |
| ---- | -------------- | ----------------- | -------------- | ----------- | -------------------- |
| 1    | ecoli          | UCI               | imU            | (336,7)     | 8.6:1                |
| 2    | optical_digits | UCI               | 8              | (5620,64)   | 9.1:1                |
| 3    | satimage       | UCI               | 4              | (6435,36)   | 9.3:1                |
| 4    | pen_digits     | UCI               | 5              | (10992,16)  | 9.4:1                |
| 5    | abalone        | UCI               | 7              | (4177,10)   | 9.7:1                |
| 6    | sick_euthyroid | UCI               | sick euthyroid | (3163,42)   | 9.8:1                |
| 7    | spectrometer   | UCI               | $\ge 44$       | (531,93)    | 11:1                 |
| 8    | car_eval_34    | UCI               | good, v.good   | (1728,21)   | 12:1                 |
| 9    | isolet         | UCI               | A,B            | (7797,617)  | 12:1                 |
| 10   | us_crime       | UCI               | $\gt 0.65$     | (1994,100)  | 12:1                 |
| 11   | yeast_ml8      | LIBSVM            | 8              | (2417,103)  | 13:1                 |
| 12   | scene          | LIBSVM            | $\gt$1 label   | (2407,294)  | 13:1                 |
| 13   | libras_move    | UCI               | 1              | (360,90)    | 14:1                 |
| 14   | thyroid_sick   | UCI               | sick           | (3772,52)   | 15:1                 |
| 15   | coil_2000      | KDD               | minority       | (9822,85)   | 16:1                 |
| 16   | arrhytmia      | UCI               | 06             | (452,278)   | 17:1                 |
| 17   | solar_flare_m0 | UCI               | M->0           | (1389,32)   | 19:1                 |
| 18   | oil            | UCI               | minority       | (937,49)    | 22:1                 |
| 19   | car_eval_4     | UCI               | vgood          | (1728,21)   | 26:1                 |
| 20   | wine_quality   | UCI               | $\le 4$        | (4898,11)   | 26:1                 |
| 21   | letter_img     | UCI               | Z              | (20000,16)  | 26:1                 |
| 22   | yeast_me2      | UCI               | ME2            | (1484,8)    | 28:1                 |
| 23   | webpage        | LIBSVM            | minority       | (34780,300) | 33:1                 |
| 24   | ozone_level    | UCI               | ozone          | (2536,72)   | 34:1                 |
| 25   | mammography    | UCI               | minority       | (11183,6)   | 42:1                 |
| 26   | protein_homo   | KDD               | minority       | (145751,74) | 11:1                 |
| 27   | abalone_19     | UCI               | 19             | (4177,10)   | 130:1                |

*Table 3 : Details on dataset used for empirical test. Columns are internal ID, short name, source repository (see above for complete reference), target column, or value, of the binary classifier, dataset shape, and imbalanced ratio, as numbers of samples in the majority class divided by numbers of samples in the minority class.*

### Models

The following list of models has been trained for every different dataset/resampler combination. All models used the relative `scikit-learn` implementation.

* k-neighbors classifier
  * k=3
* Support Vector Classifier (linear kernel)
  * C=0.025
  * max_iterations = 4000
* Support Vector Classifier (RBF kernel)
  * $\gamma$ = 2
  * C=1
  * max_iterations = 4000
* Decision Tree classifier
  * max_depth = 5
* Gaussian Naive Bayes Classifier
* Random Forest Classifier
  * max_depth = 5
  * n. of estimators = 10
  * max_features = 1
* Multi layer perceptron
  * 1 hidden layer, 30 neurons
  * learning rate = adaptive
  * alpha = 1
  * max_iterations = 1000
* ADABoost classifier
* Quadratic Discriminant Analysis

All unspecified parameters were left at default values. For further details on models and default parameters, check `scikit-learn` API reference.

An additional model was tested, a Gaussian Process Classifier, but it was found to be too computational heavy for the large number of tests required.

### Resamplers

We have tested the following already described resamplers:

* no resampling (original dataset)
* Random Over Sampler (ROS)
* Random Under Sampler (RUS)
* SMOTE
* ADASYN
* and, of course, ROSE.

### Chosen metrics

The following metrics has been collected, for every model/dataset/resampler combination:

* precision
* recall
* $F_1$
* support
* AUC
* Matthews correlation coefficient



## Results

We report the tables of Matthews Correlation Coefficient for each model.

![image-20201013165631877](image-20201013165631877.png)

![image-20201013165826295](image-20201013165826295.png)

![image-20201013165834607](image-20201013165834607.png)

![image-20201013165843620](image-20201013165843620.png)

![image-20201013165853579](image-20201013165853579.png)

![image-20201013165904876](image-20201013165904876.png)



![image-20201013165915415](image-20201013165915415.png)

![image-20201013165926426](image-20201013165926426.png)

![image-20201013165932412](image-20201013165932412.png)

Other metrics are reported in Appendix 2, for completeness.

For different algorithms we can observe different effects of using Rose, compared to other algorithms, for resampling. For K-neighbors classifiers, random forest classifiers, or Gaussian naive Bayes, we observe no difference due to resampling methods.

Some algorithms are not improved by Rose resampling, like support vector machines, on most datasets.

For neural network based models like multi-layer perceptrons, decision tree classifiers, ADABoost classifiers, and quadratic discriminant analysis instead, Rose performs equally or even better that state-of-the-art resamplers, independently from cardinality, sample size, with a tendency to perform better for high imbalance ratio problems, in the lower part of the tables reported above. For some problems, nonetheless, Rose perform inexplicably much better or much worse than other resamplers. This effect nudge to the commonly shared knowledge that asserts than, in these kinds of problems, different resamplers should be tested and benchmarked, and the absence of a universally better algorithm.

# ORBIS Dataset: a real world ROSE application

Benchmark test datasets usually are convenient. The data are clean, there is an actual relationship, all the variables are used.

We decided to test ROSE in a real world problem belonging to a field considered difficult to handle: econometrics. 

## Problem description

In this particular project the Client asked the following question:

> Is it possible to foresee which firm have potential for becoming an High Growth Firm, given their economic status at the first year of activity?

It is generally understood that the outcome of such prevision is not feasible with 100% accuracy, but over last years many techniques were advanced to improve forecast from economic datasets.[^Coad, 2019] One of the main issues about this topic is the imbalanced nature of the problem. Without the aim of generating the best model, we want to explore the effect of data rebalancing using ROSE on this dataset, when training some basic, unoptimized model. Model choice and parameter optimization has been left on for future work, being out of scope for this project.

## Dataset description

The Client provided a dataset that is a subset of ORBIS database, a collection of information on listed company across the globe, curated  by Bureau Van Djik (henceforth BvD), a Moody's Analytics controlled private society. BvD collects information from about 375 millions of public and private firms in a standardized way, allowing for comparison and analytics. ORBIS data comes from more than 160 providers and hundreds of internal sources. The firm activity itself revolves about the reconstruction of proprietary assets and recognition of effective owners, providing firm structure hierarchy diagrams to rebuild dependencies among groups and controlled societies. Those data can be used to find informations about a firm, can be filtered to find firms that satisfy certain criteria, analyze peer groups, retrieve market informations about competitors and potential collaborations, and analyze stakeholders interdependence and financial strength.

ORBIS is used by enterprise, governments and public administrations, academic entities, financial institutes and professional studies, and is focused on efficiency aimed at decisional processes. Different targets can be optimized by ORBIS data:

* Credit risk
* Compliance and financial frauds
* Supply chain risk
* Transfer pricing
* Commercial development
* M&A and corporate finance
* Master Data Management projects

We had no direct source to the original data, that were provided as a comma separated values (CSV) archived version with the data of 115840 firms. Being expensive data, we are not allowed to publish them for repeatability, but we included a MD5 checksum of the provided file.

```
HGFfinal.merge.csv
Size: 44,7 MB (44745032 bytes)
MD5 checksum: 420d345c68dc3998b8403ab07d0fecf8
```

Our datasets encompasses 3 different categories of information:

* Company information, like name, location, contacts, sector, NACE code, etc.
* Economic information:
  * Balance sheet
  * Profit and Loss (P&L) statement
* BvD evaluations, like trust level, default chances, and independence score.

For most of them, where numeric values were not available, qualitative informations where provided. Still, a lot of data were missing.

### Exploratory Data Analysis

We report here the procedure of data import and cleaning that has been done before performing any other test.

#### Data import

Data has been imported in a Pandas DataFrame, and analyzed in a dedicated Python 3.6 `conda` environment in a Jupyter Notebook on a local Linux machine.

Numeric data has been parsed to `int` and `float` data types accordingly, while ordered categories, like `BvD.Independence.Indicator` has been cast in `pd.api.types.CategoricalDtype()` format.

#### Variables Description

A report for univariate analysis can be read in Appendix 1. Every variable has been explored for completeness, cardinality, range, and basic statistics.

## High Growth Firms

The first issue to track was negotiating with the Client an objective definition of High Growth Firm (henceforth, HGF). HGF is a dichotomic variable, defining whether a firm is a good performer.

There are multiple definition of HGF in literature, that leads to the choice of our metric.

### HGF metrics

There are three different accepted definition of HGF:

1. **Compound Annual Growth Rate (CAGR)**. Companies with an average growth rate $\ge$20% for the first 5 years:
   $$
   CAGR =\left(\frac{turnover_{2014}}{turnover_{2010}}\right)^{1/4}-1 \ge 20\%​$
   $$

1. **Gazelle**[^Birch, 1995] gazelles are firm with a growth rate that remains $\gt$20% for the first 5 years
   $$
   Gazele = all \left( \frac{turnover_t}{turnover_{t-1}}\ge 20\% \right), for\ t=2010,\dots,2014
   $$

1. **Eurostat**[^Chianca, 2008], employed by Eurostat, being HGF means having a growth rate $\ge$20% for 3 consecutive years.
   $$
   \begin{align}
   Eurostat =  \exist\ t\in\{2010,2011,2012\}:\\\left( \frac{turnover_t}{turnover_{t-1}}\ge 20\% \and \\\frac{turnover_t+1}{turnover_{t}}\ge 20\% \and \\\frac{turnover_t+2}{turnover_{t+1}}\ge 20\% \right)
   \end{align}
   $$
   

In this dataset the Client chose to compute HGF by the second option, **Gazelle**.

![img](Wed, 14 Oct 2020 161521.png)

The dataset is heavily unbalanced, with a ratio of $\sim 33:1$ against non-HGF firm.                                       

$p(HGF|country)$:

![image-20200624145801088](image-20200624145801088.png)

The uneven distribution of the variable frequency on different countries, especially in Turkey, may indicate different dataset inclusion criteria for companies in different countries. This could be due to state-specific requirements, with governments or financial entities requiring the company to give their data to BvD for transparency purposes. Special care should be take for every inference about models, that will be forced to marginalize on firm location data.

## Using ROSE on ORBIS dataset

We applied ROSE on the dataset, and checked the performance of different models pre- and post- resampling. To be able to do that, we cleaned the dataset, assigning correct data types, filtering typos, dropping redundant columns. The entire process is explained in this sub-chapter.

### data cleaning

From bibliography, consultation with financial experts and understanding the dataset we recognized that a lot of columns where just sum of other columns. The graphs in variable descriptions (Appendix 1) helps understanding this collinearity. It was judged safe to just drop derived variables, keeping only original ones. The following variables were dropped:

```
'Fixed.assets.th.EUR.2010',
'Current.assets.th.EUR.2010',
'Total.assets.th.EUR.2010',
'Shareholders.funds.th.EUR.2010', 
'Non.current.liabilities.th.EUR.2010', 
'Other.current.liabilities.th.EUR.2010',
'Sales.th.EUR.2010',
'Financial.revenue.th.EUR.2010',
'Financial.expenses.th.EUR.2010', 
'Taxation.th.EUR.2010',
'Cash.flow.th.EUR.2010',
```

For the same reason, we dropped variables derived from NACE code:

```
'NACE.Rev..2.main.section',
'NACE.Rev..2.Primary.code.s.'
```

The client wanted to focus only on private small companies, so the dataset was filtered in that sense, and the variable was dropped.

```python
df = df[df['Standardised.legal.form']=='Private limited companies']
df = df.drop('Standardised.legal.form', axis=1)

df = df[df['Category.of.the.company']=='Small company']
df = df.drop('Category.of.the.company', axis=1)
```

Consolidation Code was deemed irrelevant by the Client, and hence dropped. Given ROSE inability to work on string values, and the excessive cardinality of postcodes, the following variables were dropped:

```
"Company.name",
"City",
"trust",
"Postcode",
"Postcode2"
```

Categorical variables were then one-hot-encoded:

```python
for var in ["Country.ISO.Code",
           "BvD.Independence.Indicator",
           "NACE.Rev..2.Core.code..4.digits.",
           "BvD.major.sector",
            "trustVal",
            "Trademarks...Type",]:

    temp = pd.get_dummies(df[var])
    df = df.join(temp)
    df = df.drop(var, axis=1)
```

After the cleaning, the dataset's shape was 90711 examples, with 832 variables, most of them due to the one-hot-encoding. Of these, in 2343 samples $HGF=True$, while in 88368 $HGF=False$.

### Data visualization

Given the high dimensionality, we used $t$-distributed stochastic neighbor embedding ($t$-SNE) to plot a representation of the original data. The parameters of the $t$-SNE were: $perplexity=100, iterations = 250, n\_components=2$. Extra iteration and different perplexities has been tested, withtout significant improvement. 

![image-20201014140225746](image-20201014140225746.png)

Fig _ $t$-SNE of original dataset. Green sample ($HGF=True$) size has been exaggerated on purpose. 



### ROSE Resampling

A default `imblearn.over_sampling.ROSE()` instance has been generated, with `random_state` parameter set on 42 for reasons pertaining the life, the universe, and everything else. We used the same $t$-SNE methodology as above to visualize the balanced dataset. 

![image-20201014124045802](image-20201014124045802.png)

Fig _ $t$-SNE plot of resampled dataset

The resampler was used to even the classes, and different models has been tested, without optimization. To begin, we tested a Gaussian Naive Bayes model:

**Performance on original dataset:**

| HGF   | Precision | Recall | F1    | Support |
| ----- | --------- | ------ | ----- | ------- |
| True  | 0.029     | 0.762  | 0.055 | 202     |
| False | 0.985     | 0.376  | 0.545 | 8343    |

**Performance on balanced dataset:**

| HGF   | Precision | Recall | F1    | Support |
| ----- | --------- | ------ | ----- | ------- |
| True  | 0.691     | 0.956  | 0.738 | 8332    |
| False | 0.894     | 0.367  | 0.521 | 8358    |

Then we checked ROC Curves for a non-optimized logistic regression model, encompassing all variables.

**Performance on original dataset:**

![image-20201014114940353](image-20201014114940353.png)

**Performance on balanced dataset:**

![image-20201014115021537](image-20201014115021537.png)

To better visualize the tradeoff between precision and recall in both models, we plotted threshold plots of the logistic classifier, and precision/recall curves of a ridge classifier:

**Performance on original dataset:**

![image-20201014115211086](image-20201014115211086.png)

![image-20201014120353543](image-20201014120353543.png)

**Performance on balanced dataset:**

![image-20201014120212609](image-20201014120212609.png)

![image-20201014120420092](image-20201014120420092.png)

# Discussion

This work's first objective, ROSE implementation in Python's package `imbalanced-learn` has been successfully achieved, and with the next release it will be available for all users.

Binary classifier metrics evaluation and choice has proven a big challenge. Matthews correlation coefficient has proven a severe judge, performing better than $F_1$ score in describing, in a single number, the model performance.

Additional models could have been tested, like bigger ANNs, different NN architectures, or Gaussian Process classifiers, but additional computational power is required to do that, given the number of models to train and compare. By expanding the set, given the high repeatability of the tests, we could be able to propose a standard suite for testing resamplers.

Testing ROSE under different datasets and algorithms shown that, in some cases, its performance can equal and even be better than other resamplers. The difference is exacerbated when the imbalance ratio of the dataset is higher.

This is only the first part of ROSE development for Python. The algorithm still have unsolved issues, like incapacity of treating categorical data, or variables with limited support. Ideas for solution has been discussed, and will be implemented in the future, but their implementation and validation was out of scope for this project.

# Bibliography

[^Yu, 2013]: Yu, H., Hong, S., Yang, X., Ni, J., Dan, Y., Qin, B.: Recognition of Multiple imbalanced cancer types based on DNA microarray data using ensemble classifiers. BioMed Res. Int. 2013, 1–13 (2013)
[^Zhao, 2008]:Zhao, X.M., Li, X., Chen, L., Aihara, K.: Protein classification with imbalanced data. Proteins Struct. Funct. Bioinf. 70(4), 1125-1132(2008)
[^Cerf, 2013]:Cerf, L., Gay, D., Selmaoui-Folcher, N., Crémilleux, B., Boulicaut, J.F.: Parameter-free classification in multi-class imbalanced data sets. Data Knowl. Eng. 87, 109–129 (2013)
[^Gao, 2014]: Gao, X., Chen, Z., Tang, S., Zhang, Y., Li, J.: Adaptive weighted imbalance learning with application to abnormal activity recognition. Neurocomputing 173, 1927–1935 (2016)
[^Razakarivony, 2016]: Razakarivony, S., Jurie, F.: Vehicle detection in aerial imagery: a small target detection benchmark. J. Vis. Commun. Image Represent. 34, 187–203 (2016)
[^Efron, 1993]: Tibshirani, Robert J.; Efron, Bradley. An introduction to the bootstrap. *Monographs on statistics and applied probability*, 1993, 57: 1-436.
[^Gao2, 2014]: Gao, Z., Zhang, L., Chen, M.-yu., Hauptmann, A.G., Zhang, H., Cai, A.N.: Enhanced and hierarchical structure algorithm for data imbalance problem in semantic extraction under massive video dataset. Multimed. Tools Appl. 68(3), 641–657 (2014)
[^Wang, 2010]: Wang, S., Chen, H., Yao, X.: Negative correlation learning for classification ensembles. In: 2010 International Joint Conference on Neural Networks (IJCNN), pp. 1–8. IEEE (2010)
[^Provost, 2001]: Provost, Foster & Fawcett, Tom. (2001). Robust Classification for Imprecise Environments. Machine Learning. 42. 203-231. 10.1023/A:1007601015854. 
[^Ganganwar, 2012]: Ganganwar, Vaishali. (2012). An overview of classification algorithms for imbalanced datasets. International Journal of Emerging Technology and Advanced Engineering. 2. 42-47. 
[^King, 2001]: King, Gary, and Langche Zeng. "Logistic regression in rare events data." *Political analysis* 9.2 (2001): 137-163.
[^Menardi, 2014]: Menardi, Giovanna, and Nicola Torelli. "Training and assessing classification rules with imbalanced data." *Data Mining and Knowledge Discovery* 28.1 (2014): 92-122.
[^Chawla, 2003]: Chawla, Nitesh V., et al. "SMOTEBoost: Improving prediction of the minority class in boosting." *European conference on principles of data mining and knowledge discovery*. Springer, Berlin, Heidelberg, 2003.
[^Gue, 2003]:Gue, Kevin R. "A dynamic distribution model for combat logistics." *Computers & Operations Research* 30.3 (2003): 367-381.
[^Ndour, 2012]: Ndour, Cheikh, Aliou Diop, and Simplice Dossou-Gbété. "Classification  approach based on association rules mining for unbalanced data." *arXiv preprint arXiv:1202.5514* (2012).
[^Liu, 2006]:Liu, Xu-Ying, and Zhi-Hua Zhou. "The influence of class imbalance on cost-sensitive learning: An empirical study." *Sixth International Conference on Data Mining (ICDM'06)*. IEEE, 2006.
[^Zhou, 2010]:Zhou, Zhi‐Hua, and Xu‐Ying Liu. "On multi‐class cost‐sensitive learning." *Computational Intelligence* 26.3 (2010): 232-257.
[^He, 2009]: He, Haibo, and Edwardo A. Garcia. "Learning from imbalanced data." *IEEE Transactions on knowledge and data engineering* 21.9 (2009): 1263-1284.
[^Weiss, 2003]:Weiss, Roger D., et al. "Long-term outcomes from the national drug abuse treatment clinical trials network prescription opioid addiction  treatment study." *Drug and alcohol dependence* 150 (2015): 112-119.
[^Chawla, 2002]: N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE: synthetic minority over-sampling technique,” Journal of artificial intelligence research, 321-357, 2002.
[^Han, 2005]: Han, Hui, Wen-Yuan Wang, and Bing-Huan Mao. "Borderline-SMOTE: a new over-sampling method in imbalanced data sets learning." *International conference on intelligent computing*. Springer, Berlin, Heidelberg, 2005.
[^Batista, 2004]: G. Batista, R. C. Prati, M. C. Monard. “A study of the behavior of several methods for balancing machine learning training data,” ACM Sigkdd Explorations Newsletter 6 (1), 20-29, 2004.
[^Batista, 2003]:G. Batista, B. Bazzan, M. Monard, “Balancing Training Data for Automated Annotation of Keywords: a Case Study,” In WOB, 10-18, 2003.
[^Wilson, 1972]: Wilson, Dennis L. "Asymptotic properties of nearest neighbor rules using edited data." *IEEE Transactions on Systems, Man, and Cybernetics* 3 (1972): 408-421.
[^Hart, 1968]:P. Hart, “The condensed nearest neighbor rule,” In Information Theory, IEEE Transactions on, vol. 14(3), pp. 515-516, 1968.
[^Kubat, 1997]: M. Kubat, S. Matwin, “Addressing the curse of imbalanced training sets: one-sided selection,” In ICML, vol. 97, pp. 179-186, 1997.
[^Mani, 2003]: I. Mani, I. Zhang. “kNN approach to unbalanced data distributions: a case study involving information extraction,” In Proceedings of workshop on learning from imbalanced datasets, 2003.
[^Smith, 2014]: D. Smith, Michael R., Tony Martinez, and Christophe Giraud-Carrier. “An instance level analysis of data complexity.” Machine learning 95.2 (2014): 225-256.
[^Nguyen, 2009]:H. M. Nguyen, E. W. Cooper, K. Kamei, “Borderline over-sampling for imbalanced data classification,” International Journal of Knowledge Engineering and Soft Data Paradigms, 3(1), pp.4-21, 2009.
[^Last, ArXiV]: Felix Last, Georgios Douzas, Fernando Bacao, “Oversampling for Imbalanced Learning Based on K-Means and SMOTE” https://arxiv.org/abs/1711.00837
[^He, 2008]:He, Haibo, Yang Bai, Edwardo A. Garcia, and Shutao Li. “ADASYN: Adaptive synthetic sampling approach for imbalanced learning,” In IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence), pp. 1322-1328, 2008.

[^Hofmann, 2008]: Hofmann, Thomas, Bernhard Schölkopf, and Alexander J. Smola. "Kernel methods in machine learning." *The annals of statistics* (2008): 1171-1220.
[^Bowman, 1997]: Bowman, Adrian W., and Adelchi Azzalini. *Applied smoothing techniques for data analysis: the kernel approach with S-Plus illustrations*. Vol. 18. OUP Oxford, 1997
[^Silverman, 1986]:Silverman, Bernard W. *Density estimation for statistics and data analysis*. Vol. 26. CRC press, 1986.
[^Mower, 2005]:Mower, Jeffrey P. "PREP-Mt: predictive RNA editor for plant mitochondrial genes." *BMC bioinformatics* 6.1 (2005): 96.
[^Van Rijsbergen, 1986]:Van Rijsbergen, Cornelis J. "A new theoretical framework for information retrieval." *Acm Sigir Forum*. Vol. 21. No. 1-2. New York, NY, USA: ACM, 1986.
[^Chicco, 2020]: Chicco, Davide, and Giuseppe Jurman. "The advantages of the Matthews  correlation coefficient (MCC) over F1 score and accuracy in binary  classification evaluation." *BMC genomics* 21.1 (2020): 6.
[^Fowlkes, 1983]: Fowlkes, Edward B., and Colin L. Mallows. "A method for comparing two hierarchical clusterings." *Journal of the American statistical association* 78.383 (1983): 553-569.
[^Henning, 1989]:Henning, Andersen. "Markedness: The First 150 Years." *Markedness in Synchrony and Diachrony, Olga M. Tomic (ed.), Mouton de Gruyter, Berlin–Germany* (1989): 11-46.
[^Youden, 1950]:Youden, William J. "Index for rating diagnostic tests." *Cancer* 3.1 (1950): 32-35.
[^Guilford, 1954]: Guilford, Joy Paul. "Psychometric methods." (1954).
[^Baldi, 2000]:Baldi P, Brunak S, Chauvin Y, Andersen CA, Nielsen H. Assessing the  accuracy of prediction algorithms for classification: an overview.  Bioinformatics. 2000; 16(5):412–24.
[^FDA, 2010]:The MicroArray Quality Control (MAQC) Consortium. The MAQC-II Project: a comprehensive study of common practices for the development and  validation of microarray-based predictive models. Nat Biotechnol. 2010;  28(8):827–38.
[^Brown, 2018]: Brown JB. Classifiers and their metrics quantified. Mol Inform. 2018; 37:1700127
[^Tague, 1992]: Tague-Sutcliffe J. The pragmatics of information retrieval experimentation, revisited. Informa Process Manag. 1992; 28:467–90.
[^Flight, 2015]: Flight L, Julious SA. The disagreeable behaviour of the kappa statistic. Pharm Stat. 2015; 14:74–8.
[^Sebastiani, 2015]:Sebastiani F. An axiomatically derived measure for the evaluation of  classification algorithms. In: Proceedings of ICTIR 2015 – the ACM SIGIR 2015 International Conference on the Theory of Information Retrieval.  New York City: ACM: 2015. p. 11–20.
[^Powers, 2011]: Powers DMW. Evaluation: from precision, recall and F-measure to ROC,  informedness, markedness & correlation. J Mach Learn Technol. 2011;  2(1):37–63.
[^ROSE CRAN]: Nicola Lunardon, Giovanna Menardi, Nicola Torelli: https://cran.r-project.org/web/packages/ROSE/ROSE.pdf
[^imblearn]: https://imbalanced-learn.org/stable/
[^sklearn]:https://scikit-learn.org/stable/
[^PEP8]: https://www.python.org/dev/peps/pep-0008/
[^commits]: https://github.com/scikit-learn-contrib/imbalanced-learn/pull/754

[^Birch, 1995]: Birch, David L., Anne Haggerty, and William Parsons. *Who's creating jobs?: 1995*. Cognetics, Inc., 1995.
[^Chianca, 2008]: Chianca, Thomaz. "The OECD/DAC criteria for international development evaluations: An assessment and ideas for improvement." *Journal of Multidisciplinary Evaluation* 5.9 (2008): 41-51.

[^Coad, 2019]:Coad, A., Srhoj, S. Catching Gazelles with a Lasso: Big data techniques for the prediction of high-growth firms. *Small Bus Econ* **55,** 541–565 (2020).

# Appendix 1: Univariate analysis

## Company informations

### BvD.ID.number

This is our dataset primary key. Unique (cardinality = n). It is composed by 2 letters and 8$\div$12 digits. The two letters appear to be the `Country.ISO.Code`.

```
CategoricalIndex(['IS4203100990', 'GR997722505', 'BG201066368', 'BG201251947',
                  'BG201331418', 'IS4102101180', 'BG201222746', 'BG201124711',
                  'BG201005899', 'BA4281217330002',
                  ...
                  'SK45369747', 'SK45284300', 'SK45349304', 'SK45457824',
                  'SK45432112', 'SK45480371', 'SK45452245', 'SK45407851',
                  'SK45430268', 'SK45418527'],
                 categories=['AT9010104250', 'AT9030242392', 'AT9070278738', 'AT9070279036', 'AT9090150166', 'AT9110712698', 'AT9110713446', 'AT9110713447', ...], ordered=False, name='BvD.ID.number', dtype='category', length=115840)
```

### Company.name

`dtype: string`

Contains the firm name. All caps, sometimes includes firm's juridic form.

```
BvD.ID.number
BG201056516               AW TRONICS OOD
IT02637960606       HOME DESIGN - S.R.L.
FR519561369                FINANCIERE HL
SK45371296         EU MANAGEMENT, S.R.O.
FR529210692      TOTAL E&P WELL RESPONSE
Name: Company.name, dtype: string
```

### Country.ISO.Code

`dtype: string`

Contains a two-letter ISO 3166 alpha-2 code from 38 countries. 

```python
data['Country.ISO.Code'].unique()
```

```bash
['IS', 'GR', 'BG', 'BA', 'BE', 'IE', 'CY', 'DE', 'AT', 'DK', 'GB', 'CH', 'CZ',
 'EE', 'ES', 'FI', 'FR', 'HR', 'HU', 'IT', 'RS', 'PL', 'UA', 'ME', 'NL', 'LU',
 'MT', 'MK', 'TR', 'LT', 'LV', 'NO', 'PT', 'RO', 'RU', 'SE', 'SI', 'SK']
 Length: 38, dtype: string
```

![image-20200624094626444](image-20200624094626444.png)

*Involved countries*

```
Country.ISO.Code  AT  BA   BE   BG  CH  CY    CZ  DE  DK    EE  ...   PL  \
HGF                                                             ...        
0                 19  97  190  626   2   3  1985  28   1  2489  ...  446   
1                  0  14   11   31   0   0    60   0   0   114  ...   46   

Country.ISO.Code    PT     RO   RS     RU    SE    SI    SK  TR   UA  
HGF                                                                   
0                 6535  15500  361  11731  2512  1693  3497   5  374  
1                  265    758   20    192    76    92   106   3    6  

[2 rows x 38 columns]

Chi² =  4.79e+02
p    =   3.4e-78
degrees of freedom = 37
```

### Postcode

Postcode of the firm. Refers to a different encoding for each country. 

```
BvD.ID.number
FR522454743     81640
RU66322917     129085
EE11951827      10151
FR519806830     18700
RO27831630        nan
Name: Postcode, dtype: category
Categories (29914, object): [00-024, 00-042, 00-066, 00-102, ..., YO19 6ED, YO26 4GB, YO41 5NS, nan]
```

Present only in 107506 rows. Missing in 8334 rows.

### City

All caps name of the firm's city. Missing in 88 entries.

### NACE codes

 Statistical Classification of Economic Activities in the European Community code, known as NACE, is the industry standard classification of European Union. Established by Regulation (EC) No 1893/2006, it uses four hierarchical levels:

- Level 1: 21 sections identified by alphabetical letters A to U;
- Level 2: 88 divisions identified by two-digit numerical codes (01 to 99);
- Level 3: 272 groups identified by three-digit numerical codes (01.1 to 99.0);
- Level 4: 615 classes identified by four-digit numerical codes (01.11 to 99.00).

The first four digits of the code, which is the first four levels of the classification system, are the same in all European countries. National implementations may introduce additional levels. The fifth digit might vary from country to country and further digits are sometimes placed by  suppliers of databases.

links: [Reference to all NACE codes](https://ec.europa.eu/competition/mergers/cases/index/nace_all.html) , [Wikipedia: NACE codes](https://en.wikipedia.org/wiki/Statistical_Classification_of_Economic_Activities_in_the_European_Community). 

### NACE.Rev..2.main.section

Level 1 NACE code. A letter, and the sector description.

```
BvD.ID.number
FR521201111                                       F - Construction
FR523714624      G - Wholesale and retail trade; repair of moto...
ESB85891794                           S - Other service activities
IT03972880235        I - Accommodation and food service activities
PT509412866      M - Professional, scientific and technical act...
Name: NACE.Rev..2.main.section, dtype: category
```

All 21 sections are being represented.

```python
pd.DataFrame(data['NACE.Rev..2.Core.code..4.digits.']).reset_index().groupby('NACE.Rev..2.main.section').count().plot.barh()
```



![img](Wed, 14 Oct 2020 161140.png)

*number of entries per section*

```
NACE     A    B     C     D    E      F      G     H     I     J  ...     L  \
HGF                                                               ...         
0     2938  213  9621  2429  593  14212  28334  5038  5942  4779  ...  8340   
1       99    8   444    33   43    385   1192   294   154   266  ...   146   

NACE      M     N   O     P     Q     R     S  T  U  
HGF                                                  
0     13611  4899  16  1116  2504  1646  2151  2  1  
1       460   244   0    66   129    75    64  0  0  

[2 rows x 21 columns]

Chi² =  3.96e+02
p    =   1.6e-71
degrees of freedom = 20
```

### NACE.Rev..2.Core.code..4.digits.

4 digit NACE code. 729 different categories.

```
BvD.ID.number
NO995138697      4110
PT509493599      4339
RO27726219       4711
FR520957143       161
IT03102890831    4120
Name: NACE.Rev..2.Core.code..4.digits., dtype: category
Categories (729, int64): [100, 110, 111, 112, ..., 9609, 9700, 9810, 9900]
```

```
NACE  100   110   111   112   113   115   119   120   121   122   ...  9529  \
HGF                                                               ...         
0       18    60   571     8   227     1    64    17   117     1  ...    34   
1        2     1    20     0     5     0     1     1     7     0  ...     1   

NACE  9600  9601  9602  9603  9604  9609  9700  9810  9900  
HGF                                                         
0        4   131   957   129   186   351     1     1     1  
1        0     3    20     4     6    16     0     0     0  

[2 rows x 729 columns]

Chi² =  1.69e+03
p    =   9.4e-78
degrees of freedom = 728
```

### NACE.Rev..2.Primary.code.s.                   

4 digit NACE code. Similar to the former, it contains duplicates. 729 different categories.

```
BvD.ID.number
RU65230449      4633
NO995818558     4759
SE5568024276    7311
NO995182165     3312
RO27703886      4613
Name: NACE.Rev..2.Primary.code.s., dtype: category
Categories (729, int64): [100, 110, 111, 112, ..., 9609, 9700, 9810, 9900]
```

### Cons..code

Bankscape Consolidation Code. It indicates the level of consolidation for the different financial statements

- **C1**: statement of a mother company integrating the statements of its controlled subsidiaries or branches with no unconsolidated companion,
- **C2**: statement of a mother company integrating the statements of its controlled subsidiaries or branches with an unconsolidated companion,
- **U1**: statement not integrating the statements of the possible controlled subsidiaries or branches of the concerned company with no consolidated companion.
- **U2**: statement not integrating the statements of the possible controlled subsidiaries or branches of the concerned company  with an consolidated companion.
- **LF**: limited financials: information based on rounded figures officially available, sometimes collected from other directories or websites.

```
BvD.ID.number
RO26440005       U1
IT03232890982    U1
IT11001531000    U1
RU67267304       LF
PT509588980      U1
Name: Cons..code, dtype: category
Categories (5, object): [C1, C2, LF, U1, U2]

```

![img](Wed, 14 Oct 2020 161151.png)

```
CONS   C1   C2   LF      U1   U2
HGF                             
0     116  103  784  110504  140
1       7    3    7    4170    6

Chi² =      18.9
p    =   0.00084
degrees of freedom = 4
```

### BvD.Independence.Indicator

It characterizes  the  degree  of  independence  of a  company  with  regard  to  its  shareholders. It has been mapped to an ordered category, with null value (`U`) being set at the lowest value.

links: [Variable description](https://aisre.it/images/aisre/5583cfea457478.86345032/Succurro1_Costanzo_Explaining%20the%20propensity%20to%20patent_evidence%20from%20Italy.pdf)

```
BvD.ID.number
RO26798381       D
RU67068071       D
FR527515381      U
RU64795818       D
IT06649101216    D
Name: BvD.Independence.Indicator, dtype: category
Categories (13, object): [U < D- < D < D+ ... B+ < A- < A < A+]
```

![img](Wed, 14 Oct 2020 161154.png)



```
INDEP  -   A    A+   A-  B     B+   B-    C   C+      D      U
HGF                                                           
0      3  20  2170  257  1  20777  346  104  197  68757  19015
1      0   1    63   21  0    837   16   13    8   2588    646

Chi² =      46.4
p    =   1.2e-06
degrees of freedom = 10
```

  ### BvD.major.sector

A different sector encoding from BvD. It encompass 19 categories.

```
BvD.ID.number
RO26474245                                          Other services
HR76526891156    Chemicals, rubber, plastics, non-metallic prod...
RU68874348                      Textiles, wearing apparel, leather
CZ28117956                                          Primary sector
RO15587044                                          Other services
Name: BvD.major.sector, dtype: category
Categories (19, object): [Banks, Chemicals, rubber, plastics, non-metallic prod..., Construction, Education, Health, ..., Textiles, wearing apparel, leather, Transport, Wholesale & retail trade, Wood, cork, paper]
```

There are no missing values.

![img](Wed, 14 Oct 2020 161206.png)



```
SECT  Banks  Chemicals,  Constructi  Education,  Food, beve  Gas, Water  \
HGF                                                                       
0       947        1156       14212        3809        1252        2494   
1        22          58         385         211          74          36   

SECT  Hotels & r  Insurance   Machinery,  Metals & m  Other serv  Post & tel  \
HGF                                                                            
0           5942          19        3466        1712       38654         447   
1            154           2         149          64        1340          39   

SECT  Primary se  Public adm  Publishing  Textiles,   Transport  Wholesale   \
HGF                                                                           
0           3148          17        1042        1000       4919       26666   
1            107           0          43          52        286        1137   

SECT  Wood, cork  
HGF               
0            745  
1             34  

Chi² =   2.7e+02
p    =   7.4e-47
degrees of freedom = 18
```

### Standardised.legal.form 

Two level factor, stating if the company is public or private.

```
BvD.ID.number
ESB85955946    Private limited companies
FR519336481    Private limited companies
PT509284035     Public limited companies
EE11920653     Private limited companies
NO995237563    Private limited companies
Name: Standardised.legal.form, dtype: category
Categories (2, object): [Private limited companies, Public limited companies]
```

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAfQAAAD4CAYAAAAaYxRFAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAfj0lEQVR4nO3debwWZf3/8debwxpuiMsPxZ+AEgJqiEfTNMXcw9SMvmrupIaaa8tPv65Y/bLy10KYSIv4yzQUrQhTMkMlczsowlFBLVHpa4l8ExcWWT7fP+Y6x1s8y9yHc59lfD8fj/txZq6Za+Zz3cODzz3XXDOjiMDMzMw6ty7tHYCZmZltOCd0MzOzAnBCNzMzKwAndDMzswJwQjczMyuAru0dgH04bbHFFjFgwID2DsPMrFOZM2fO6xGxZUPLnNCtXQwYMICampr2DsPMrFOR9FJjy9zlbmZmVgBO6GZmZgXghG5mZlYAvoZuZlZAq1evZvHixaxcubK9Q7EW6NmzJ/3796dbt2656zihm5kV0OLFi9l4440ZMGAAkto7HCtDRLB06VIWL17MwIEDc9dzl7uZWQGtXLmSvn37Opl3QpLo27dv2b0rTuhmZgXlZN55teTYOaGbmZkVgK+hm5l9CAy4+K5W3d6ia0Y3u05VVRW77LILEUFVVRUTJ07kE5/4RKPr33///Rx11FEMGjSI5cuXs/XWW/P1r3+dI444osF1r732WmbMmMGUKVP42te+Rv/+/Xn77bcZNGgQV155ZZP7am2LFi3iiCOOoLa2ts32uT4ndDMzq4hevXoxd+5cAGbOnMkll1zCAw880GSdT37yk8yYMQOAuXPncvTRR9OrVy8OPPDAJusde+yxTJw4EYBZs2ZxzDHHMGvWLIYOHdoKLam8NWvW0LXrhqVkd7mbmVnFvfnmm/Tp0weA4447jrvueq/H4NRTT2XatGkfqDNixAiuuOKK+kSd1wEHHMCZZ57J5MmTP7DsqquuYuzYsYwaNYpBgwYxYcIEIDvD3nnnnevXu/baa7nqqqsAGDVqFBdeeCHV1dUMHTqUxx9/nGOOOYbBgwdz2WWX1ddZs2YNJ5xwAkOHDmXMmDEsX74cgDlz5rD//vuz++67c+ihh/Lqq6/Wb/eCCy6gurqaH/3oR2W1sSFO6GZmVhErVqxgxIgR7LTTTpx++ulcfvnlQHY2fdtttwHw7rvvct999zF6dMNd+CNHjmTBggVl77upegsWLGDmzJk89thjjB8/ntWrVze7ve7du1NTU8O4ceM46qijuO6666itrWXKlCksXboUgIULF3L22Wfz7LPPsskmm/CTn/yE1atXc+655zJt2jTmzJnD2LFjufTSS+u3++6771JTU8NXvvKVstu4Pne5m5lZRZR2uT/88MOcfPLJ1NbWcvjhh3P++eezatUq7rnnHvbbbz969erV4DYiokX7bqre6NGj6dGjBz169GCrrbbiX//6V7PbO/LIIwHYZZddGD58OP369QNg0KBBvPLKK2y22WZst9127LPPPgCceOKJTJgwgcMOO4za2loOPvhgANauXVtfF7IfN63FCd3MzCpu77335vXXX2fJkiVstdVWjBo1ipkzZzJ16lSOO+64Rus9+eSTLboO3lS9Hj161E9XVVXVX79et25dffn694DX1enSpcv76nfp0oU1a9YAH7zVTBIRwfDhw3n44YcbjKV3795ltKpp7nI3M7OKW7BgAWvXrqVv375AdmZ64403Mnv2bA477LAG68ybN49vfOMbnHPOOWXt64EHHmDy5MmcccYZuetsvfXWvPbaayxdupRVq1bVD8wrx8svv1yfuG+55Rb23XdfhgwZwpIlS+rLV69ezdNPP132tvPwGbqZ2YdAntvMWlvdNXTIusBvuukmqqqqADjkkEM46aSTOOqoo+jevXt9ndmzZ7PbbruxfPlyttpqKyZMmFA/wn369OnU1NRw9dVXf2BfU6dO5S9/+QvLly9n4MCB3HHHHfVn6JMmTQJg3LhxjcbarVs3rrjiCvbcc0+23XZbdtppp7LbO2TIEK677jrGjh3LsGHDOOuss+jevTvTpk3jvPPOY9myZaxZs4YLLriA4cOHl7395qil1yfMNkR1dXXU1NS0dxhmhfXss892mlu2rGENHUNJcyKiuqH13eVuZmZWAE7oZmZmBeCEbmZWUL6k2nm15Ng5oZuZFVDPnj1ZunSpk3onVPc+9J49e5ZVz6PczcwKqH///ixevJglS5a0dyjWAj179qR///5l1XFCNzMroG7dujFw4MD2DsPakLvczczMCsAJ3czMrACc0M3MzArA19CtXcz/xzIGXHxXg8va4xGVZmadnc/QzczMCsAJ3czMrACc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)



```
FORM  Private  Public 
HGF                   
0      109204     2443
1        4056      137

Chi² =      21.6
p    =   3.3e-06
degrees of freedom = 1
```

### Category.of.the.company 

4 level factor stating the dimension of the company. It was impossible to retrieve information about objective inclusion criteria anywhere. Different legislations use different criteria, and despite their similarity, this does not allow a unequivocal definition.

To give an approximation of this classification, we will report Australian's definition of large company. A company is considered large if it satisfies at least two of the following criteria:

*  the consolidated revenue for the financial year of the company and the companies it controls is AU$50 millions or more,
*  the value of the consolidated gross assets at the end of the financial year of the company and any entities it controls is AU$25 millions or more, and
*  the company and any entity it controls have 100 or more employees at the end of the fiscal year.

European Union EUROSTAT website reports a different classification, based only on employees:

| number of employees | enterprise size         |
| ------------------- | ----------------------- |
| $\lt10$             | micro enterprise        |
| $10\le e\lt 50$     | small enterprise        |
| $50\le e \lt 250$   | medium-sized enterprise |
| $e\ge250$           | large enterprise        |

```
BvD.ID.number
RO26541921             Small company
RO26543973             Small company
RO27249764      Medium sized company
PT509553923            Small company
SE5568008675           Small company
Name: Category.of.the.company, dtype: category
Categories (4, object): [Large company, Medium sized company, Small company, Very large company]
```

![img](data:image/png;base64,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)



```
CAT  Large compan  Medium sized  Small compan  Very large c
HGF                                                        
0            2485         19276         89542           344
1             267          1522          2373            31

Chi² =  1.41e+03
p    =  2.3e-306
degrees of freedom = 3
```

### No.of.companies.in.corporate.group

Number of companies in the corporate group. The largest part of entries has 0 companies in the group (assuming: no group).

![img](data:image/png;base64,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)

outliers:

```
BvD.ID.number
IE488184         SKY HIGH III LEASING DESIGNATED ACTIVITY COMPANY
IT07063570969                                    HB SERVIZI S.R.L
IT07182390968                      POLIAMBULATORIO BICOCCA S.R.L.
NO995590271                                         ELKEM RANA AS
RO1590899                        ADAMA AGRICULTURAL SOLUTIONS SRL
RO25221180                                       EDPR ROMANIA SRL
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for No.of.companies.in.corporate.group
Welch's t-test statistic = -0.7536
p-value = 0.4511
```

```
Optimization terminated successfully.
         Current function value: 0.155660
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               1.205e-05
Time:                        15:23:49   Log-Likelihood:                -18032.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                    0.5098
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2825      0.016   -208.368      0.000      -3.313      -3.252
CORPGRP     2.408e-05   3.39e-05      0.710      0.478   -4.24e-05    9.06e-05
==============================================================================
```

### No.of.recorded.shareholders

Numbers of enterprise recorded stakeholders.

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXcAAAEICAYAAACktLTqAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAXo0lEQVR4nO3df5BdZX3H8ffHBBCzyA+hW0lSFruRGpOpylZAHbuLOknEiHW0kjJqNBDtFH+0VAnWjtGpJbalVBB1UkmpLWalSCUkUSraHWtFhbTWJERKtNEkYhYIrG6Mg5Fv/zhP5OZy7+7Z+2Pv7pPPa+ZO9pzznHO+z30233vuc559jiICMzPLy1M6HYCZmbWek7uZWYac3M3MMuTkbmaWISd3M7MMObmbmWXIyd0aJunFku6XNCrpNZ2Opx5JIal3svaVtFrSPzdyvomS1C9pT4P7Lpf0tTG2D0m6pPHorJOc3DMkaZekYUmzKtZdImmoxaf6EPCxiOiKiM+3+Nhm1gQn93zNAN7V5nOcAWwfr5CkmW2OY1LPM5mmY52mY8w5cnLP118DfyrppOoNkl4k6W5JI+nfF9U7iKRLJe2UtF/SBkmnp/XfA54F3J66ZY6r2m+XpCskfQc4IGmmpHMlfV3So5L+R1J/RflTJP2DpB9JekTS58eLIW0LSX8k6X7g/rTuPZIeSMd6a1Vcx0n6G0k/lLRP0iclHV+xve6+Nd6bKyTtlfRTSfdJelnF5mMlfTpt2y6pr2K/VZK+l7bdK+n3KrYtl/Sfkq6R9DCweryY036Xp29rD0h6S8X6E1McD0r6gaT3S6r5/17SKyR9N/1efAxQ1fa3StqR2ucOSWfUawcVrkkx/UTSVkkLxno/rcUiwq/MXsAu4OXArcBfpHWXAEPAKcAjwBuBmcCytPyMGsc5H3gIeAFwHHAd8NXq84wRw7eBucDxwGzgYeCVFBcVr0jLp6Xym4DPAicDxwC/WzKGAL6U6nU8sBjYBywAZgGfSWV6U/lrgA2p/AnA7cBVaduY+1bV7yxgN3B6Wu4BfjP9vBr4earrDOAq4BsV+74eOD29D28ADgDPTNuWA4eAd6T2OX6cmPtT+Q+l9+2VwM+Ak9P2TwO3pf16gP8FVlSc62vp51OBnwKvS8f543TcS9L2C4GdwHNSXO8Hvj5GOywCtgAnUXxIPOdwHf2apDzQ6QD8akOjPpHcFwAjwGk8kdzfCHyrqvxdwPIax7kB+KuK5S7gF0BP5XnGiOGtFctXAP9UVeYO4M3AM4HHDyekCcYQwPkV29cBayqWn53K9KYkc4CUhNP284D/G2/fGnH1AsPpfT6mattq4M6K5fnAwTHa69vAhenn5cAPK7aNF3M/cBCYWbF9GDiX4oPlMWB+xba3AUMV5zqc3N/EkR9AAvbwRHL/AulDIS0/heJD5Iw67XA+xQfJucBTOv1/4mh8uVsmYxGxDdgIrKpYfTrwg6qiP6C4sq52RNmIGKW42q5VtpbdFT+fAbw+dck8KulR4CUUiX0usD8iHmkwht1V5SuXK+t6GvA0YEtFDF9M68fb9wgRsRN4N0UiH5Y0WNldBPy44uefAU893Bct6U2Svl0RwwKKK+da9RkvZoCHI+JQ1fm60jGPqarHWG39q/NGkaGr2++jFTHsp/gAqNkOEfEV4GPA9RTvz1pJT69xXmsTJ/f8fQC4lCf+E/6I4j9qpd8A9tbY94iyKkbfPKNO2VoqpxzdTXHlflLFa1ZErEnbTql1f6BkDJXneYDiw+Kw36j4+SGKq9znVsRwYkR0ldj3yZWL+ExEvCTFF8BHxiqf4j8D+HvgMoqusJOAbRzZv11Zn/FiHstDFN9yKtu7XlsfUXdJ4sj3Yjfwtqr2Oz4ivl4nbiLi2og4m+Kby7OB95SI2VrEyT1z6Qrzs8A706rNwLMl/UG6yfkGiv98G2vsvh54i6TnpRumfwl8MyJ2NRDKPwNLJS2SNEPSU1WM0Z4TEQ9QfO3/uKSTJR0j6aUNxnAzsFzSfElPo/hwO/xePE6RWK+R9GsAkmZLWjTevtUknSXp/BTTzykS8OMl3odZFEnwwXSct1BcuddUIua6IuKXqU4flnRC+mD5E4q2qLYJeK6k16ZvGO8Efr1i+yeBKyU9N8VwoqTX1zu3pN+RdI6kYyi6lX5OuffHWsTJ/ejwIYqkQkQ8DLwKuJyie+O9wKsi4iGANLLj4lT2TuDPgc9RXNn9JnBRrRNIep+kL9QLICJ2U9yUex9FYttNcSV3+HfwjRRXmd+l6DN+90RjSOW/APwd8BWKG4BfqSpyRVr/DUk/Ae6kuDk67r5VdTwOWENxdfxj4NeAK+vFVRHfvcDVFPc59gELgf8cZ7e6MZfwDork+n3gaxQ3idfViOshihu9ayh+L+ZVxhUR/0rxzWQwxbANWDLGeZ9O8aH0CEVX0MMUI7hskqjoWjMzs5z4yt3MLENO7mZmGXJyNzPLkJO7mVmGpsQEP6eeemr09PQ0tO+BAweYNWvW+AWnOdczL65nXjpVzy1btjwUEafV2jYlkntPTw/33HNPQ/sODQ3R39/f2oCmINczL65nXjpVT0l1/4ra3TJmZhlycjczy5CTu5lZhpzczcwy5ORuZpYhJ3czsww5uZuZZajlyT3N0f0f6SG+/a0+vpmZja/UHzFJWkcxB/hwRCyoWL8Y+CjFsxo/lZ6qE8Ao8FSKZzC21da9IyxftWnccrvWXNDuUMzMpoyyV+43UjwZ/lckzaB4PuISiif5LJM0H/iPiFhC8YCBD7YuVDMzK6v0wzok9QAbD1+5SzoPWB0Ri9LylQARcVVaPhb4TES8rs7xVgIrAbq7u88eHBxsqALD+0fYd3D8cgtnn9jQ8aeK0dFRurrKPDZzenM98+J6ttfAwMCWiOirta2ZuWVmc+TT0fcA50h6LbAIOIni6ec1RcRaYC1AX19fNDovw3U33cbVW8evxq6LGzv+VOE5OvLieuZlKtaz5ROHRcStwK1lykpaCizt7e1tdRhmZke1ZkbL7AXmVizPSetKi4jbI2LliSdO7y4TM7OpppnkfjcwT9KZqX/9ImDDRA4gaamktSMjI02EYWZm1Uold0nrgbuAsyTtkbQiIg4BlwF3ADuAmyNi+0RO7it3M7P2KNXnHhHL6qzfDGxu9OTuczcza4+OTj/gK3czs/bw3DJmZhnqaHL3DVUzs/Zwt4yZWYbcLWNmliF3y5iZZcjdMmZmGXK3jJlZhtwtY2aWIXfLmJllyN0yZmYZcnI3M8uQk7uZWYZ8Q9XMLEO+oWpmliF3y5iZZcjJ3cwsQ07uZmYZcnI3M8uQk7uZWYY8FNLMLEMeCmlmliF3y5iZZcjJ3cwsQ07uZmYZcnI3M8uQk7uZWYac3M3MMuTkbmaWobYkd0mzJN0j6VXtOL6ZmY2tVHKXtE7SsKRtVesXS7pP0k5Jqyo2XQHc3MpAzcysvLJX7jcCiytXSJoBXA8sAeYDyyTNl/QK4F5guIVxmpnZBCgiyhWUeoCNEbEgLZ8HrI6IRWn5ylS0C5hFkfAPAr8XEY/XON5KYCVAd3f32YODgw1VYHj/CPsOjl9u4ezpPcXB6OgoXV1dnQ6j7VzPvLie7TUwMLAlIvpqbZvZxHFnA7srlvcA50TEZQCSlgMP1UrsABGxFlgL0NfXF/39/Q0Fcd1Nt3H11vGrsevixo4/VQwNDdHoezSduJ55cT07p5nkPqaIuHG8MpKWAkt7e3vbFYaZ2VGpmdEye4G5Fctz0rrSPCukmVl7NJPc7wbmSTpT0rHARcCGiRzA87mbmbVH2aGQ64G7gLMk7ZG0IiIOAZcBdwA7gJsjYvtETu4rdzOz9ijV5x4Ry+qs3wxsbvTk7nM3M2sPP4nJzCxDnlvGzCxDfkC2mVmG3C1jZpYhd8uYmWWobX+hWsZkjpbpWbWpdNlday5oYyRmZu3nbhkzswy5W8bMLENO7mZmGfJQSDOzDLnP3cwsQ+6WMTPLkJO7mVmGnNzNzDLkG6pmZhnyDVUzswy5W8bMLENO7mZmGXJyNzPLkJO7mVmGPFrGzCxDHi1jZpYhd8uYmWXIyd3MLENO7mZmGXJyNzPLkJO7mVmGnNzNzDLk5G5mlqGWJ3dJz5H0SUm3SPrDVh/fzMzGVyq5S1onaVjStqr1iyXdJ2mnpFUAEbEjIt4O/D7w4taHbGZm4yl75X4jsLhyhaQZwPXAEmA+sEzS/LTt1cAmYHPLIjUzs9IUEeUKSj3AxohYkJbPA1ZHxKK0fCVARFxVsc+miLigzvFWAisBuru7zx4cHGyoAsP7R9h3sKFd61o4e+pNhzA6OkpXV1enw2g71zMvrmd7DQwMbImIvlrbZjZx3NnA7orlPcA5kvqB1wLHMcaVe0SsBdYC9PX1RX9/f0NBXHfTbVy9tZlqPNmuixuLpZ2GhoZo9D2aTlzPvLiendParAhExBAwVKaspKXA0t7e3laHYWZ2VGtmtMxeYG7F8py0rjTPCmlm1h7NJPe7gXmSzpR0LHARsGEiB/B87mZm7VF2KOR64C7gLEl7JK2IiEPAZcAdwA7g5ojYPpGT+8rdzKw9SvW5R8SyOus308RwR/e5m5m1h5/EZGaWoZaPlslBz6pNpcrtWlNzCL+ZWcf5AdlmZhlyt4yZWYY85a+ZWYbcLWNmliF3y5iZZcjdMmZmGXJyNzPLkPvczcwy5D53M7MMuVvGzCxDTu5mZhlycjczy5BvqJqZZcg3VM3MMuRuGTOzDDm5m5llyMndzCxDTu5mZhlycjczy5CHQpqZZchDIc3MMuRuGTOzDDm5m5llyMndzCxDTu5mZhlycjczy9DMTgcwnfWs2lSq3K41F7Q5EjOzI7UluUt6DXAB8HTghoj4t3acx8zMaivdLSNpnaRhSduq1i+WdJ+knZJWAUTE5yPiUuDtwBtaG7KZmY1nIn3uNwKLK1dImgFcDywB5gPLJM2vKPL+tN3MzCaRIqJ8YakH2BgRC9LyecDqiFiUlq9MRdek15ci4s46x1oJrATo7u4+e3BwsKEKDO8fYd/BhnadNAtnN/8XuKOjo3R1dbUgmqnN9cyL69leAwMDWyKir9a2ZvvcZwO7K5b3AOcA7wBeDpwoqTciPlm9Y0SsBdYC9PX1RX9/f0MBXHfTbVy9dWrfF951cX/TxxgaGqLR92g6cT3z4np2TluyYkRcC1w7XjlJS4Glvb297QjDzOyo1ew4973A3IrlOWldKZ44zMysPZpN7ncD8ySdKelY4CJgQ9mdPeWvmVl7TGQo5HrgLuAsSXskrYiIQ8BlwB3ADuDmiNhe9pi+cjcza4/Sfe4RsazO+s3A5pZFZGZmTevoMJOj5Yaqpykws8nmJzGZmWXIz1A1M8uQr9zNzDLk+dzNzDLkbhkzswy5W8bMLEPuljEzy5CTu5lZhtznbmaWIfe5m5llaGo/5eIoM9Y0BZcvPMTytN3TFJjZeNznbmaWISd3M7MMeVbIacizTJrZeHxD1cwsQ+6WMTPLkJO7mVmGnNzNzDLk5G5mliEndzOzDHluGTOzDHkopJlZhtwtY2aWISd3M7MMObmbmWXIyd3MLENO7mZmGfLDOjLm2SPNjl4tv3KX9CxJN0i6pdXHNjOzckold0nrJA1L2la1frGk+yTtlLQKICK+HxEr2hGsmZmVU/bK/UZgceUKSTOA64ElwHxgmaT5LY3OzMwaUiq5R8RXgf1Vq18I7ExX6o8Bg8CFLY7PzMwaoIgoV1DqATZGxIK0/DpgcURckpbfCJwDfAD4MPAK4FMRcVWd460EVgJ0d3efPTg42FAFhvePsO9gQ7tOK93H07Z6Lpw9daZ/GB0dpaurq9NhtJ3rmZdO1XNgYGBLRPTV2tby0TIR8TDw9hLl1kp6AFh6wgknnN3f39/Q+a676Tau3pr/oJ/LFx5qXz23HihdtOzImkZH6gwNDdHo78J04nrmZSrWs5nRMnuBuRXLc9K60jxxmJlZezST3O8G5kk6U9KxwEXAhtaEZWZmzSj1PV/SeqAfOFXSHuADEXGDpMuAO4AZwLqI2D6Rk0taCizt7e2dWNTWMWW7W8yss0ol94hYVmf9ZmBzoyePiNuB2/v6+i5t9BhmZvZkfhKTmVmG/CQmM7MMdXQMofvcrboP//KFh1heo1/fk5uZTYyv3M3MMuT53M3MMuTkbmaWIY+WMTPLkPvczcwy5G4ZM7MMeSik2RTlZ+BaM9wtY2aWIXfLmJllyMndzCxDTu5mZhnyDVWbFlo9j7xvQlrufEPVzCxD7pYxM8uQk7uZWYac3M3MMuTkbmaWISd3M7MMeSik2RgmMgSz7PDKnlWb6j5OcCro5Jw2nk+ndTwU0swsQ+6WMTPLkJO7mVmGnNzNzDLk5G5mliEndzOzDDm5m5llyMndzCxDLf8jJkmzgI8DjwFDEXFTq89hZmZjK3XlLmmdpGFJ26rWL5Z0n6Sdklal1a8FbomIS4FXtzheMzMroWy3zI3A4soVkmYA1wNLgPnAMknzgTnA7lTsl60J08zMJkIRUa6g1ANsjIgFafk8YHVELErLV6aie4BHImKjpMGIuKjO8VYCKwG6u7vPHhwcbKgCw/tH2HewoV2nle7jcT2nuIWzy02jsXXvSEvrOZHzTrbD9Wx1jFPteM20Z9lz1zIwMLAlIvpqbWumz302T1yhQ5HUzwGuBT4m6QLg9no7R8RaYC1AX19f9Pf3NxTEdTfdxtVbOzr/2aS4fOEh13OK23Vxf6lyy9PEYa2q50TOO9kO17PVMU614zXTnmXPPVEt/18UEQeAt5Qp61khzczao5mhkHuBuRXLc9K60jwrpJlZezST3O8G5kk6U9KxwEXAhokcQNJSSWtHRia/L9DMLGdlh0KuB+4CzpK0R9KKiDgEXAbcAewAbo6I7RM5ua/czczao1Sfe0Qsq7N+M7C50ZO7z93MrD38JCYzswx5bhkzswx1NLn7hqqZWXuU/gvVtgYhPQj8oMHdTwUeamE4U5XrmRfXMy+dqucZEXFarQ1TIrk3Q9I99f78NieuZ15cz7xMxXq6z93MLENO7mZmGcohua/tdACTxPXMi+uZlylXz2nf525mZk+Ww5W7mZlVcXI3M8vQtE7udZ7hOu1Jmivp3yXdK2m7pHel9adI+pKk+9O/J3c61mZJmiHpvyVtTMtnSvpmatPPphlHpz1JJ0m6RdJ3Je2QdF6m7fnH6Xd2m6T1kp6aQ5vWeo50vfZT4dpU3+9IekEnYp62yX2MZ7jm4BBweUTMB84F/ijVbRXw5YiYB3w5LU9376KYVfSwjwDXREQv8AiwoiNRtd5HgS9GxG8Bv01R56zaU9Js4J1AX3oc5wyKqcBzaNMbqXqONPXbbwkwL71WAp+YpBiPMG2TO/BCYGdEfD8iHgMGgQs7HFNLRMQDEfFf6eefUiSC2RT1+8dU7B+B13QmwtaQNAe4APhUWhZwPnBLKjLt6wgg6UTgpcANABHxWEQ8SmbtmcwEjpc0E3ga8AAZtGlEfBXYX7W6XvtdCHw6Ct8ATpL0zMmJ9AnTObnXeobr7A7F0jbpweTPB74JdEfEA2nTj4HuDoXVKn8HvBd4PC0/A3g0PSsA8mnTM4EHgX9IXVCfkjSLzNozIvYCfwP8kCKpjwBbyLNNoX77TYncNJ2Te/YkdQGfA94dET+p3BbFGNZpO45V0quA4YjY0ulYJsFM4AXAJyLi+cABqrpgpnt7AqQ+5wspPsxOB2bx5K6MLE3F9pvOyb3pZ7hOZZKOoUjsN0XErWn1vsNf79K/w52KrwVeDLxa0i6KLrXzKfqlT0pf6SGfNt0D7ImIb6blWyiSfU7tCfBy4P8i4sGI+AVwK0U759imUL/9pkRums7JvelnuE5Vqe/5BmBHRPxtxaYNwJvTz28Gbpvs2FolIq6MiDkR0UPRdl+JiIuBfwdel4pN6zoeFhE/BnZLOiutehlwLxm1Z/JD4FxJT0u/w4frmV2bJvXabwPwpjRq5lxgpKL7ZvJExLR9Aa8E/hf4HvBnnY6nhfV6CcVXvO8A306vV1L0SX8ZuB+4Ezil07G2qL79wMb087OAbwE7gX8Bjut0fC2q4/OAe1Kbfh44Ocf2BD4IfBfYBvwTcFwObQqsp7iP8AuKb2Ir6rUfIIqRfN8DtlKMHpr0mD39gJlZhqZzt4yZmdXh5G5mliEndzOzDDm5m5llyMndzCxDTu5mZhlycjczy9D/A/PdiHzMNsmmAAAAAElFTkSuQmCC)

```
HGF vs non-HGF for No.of.recorded.shareholders
Welch's t-test statistic = -4.809
p-value = 1.571e-06

Optimization terminated successfully.
         Current function value: 0.155556
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0006832
Time:                        15:25:05   Log-Likelihood:                -18020.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                 6.913e-07
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.3319      0.018   -183.049      0.000      -3.368      -3.296
SHA            0.0315      0.006      5.698      0.000       0.021       0.042
==============================================================================

```

### No.of.recorded.subsidiaries

![img](data:image/png;base64,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)

outlier: 

```
BvD.ID.number
RS20661283    AKCIONARSKI FOND
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for No.of.recorded.subsidiaries
Welch's t-test statistic = -3.294
p-value = 0.0009946
```

```
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               8.499e-05
Time:                        15:25:58   Log-Likelihood:                -18030.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                   0.08000
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2831      0.016   -208.501      0.000      -3.314      -3.252
SUB            0.0052      0.003      2.045      0.041       0.000       0.010
==============================================================================
```

### No.of.recorded.branch.locations

![img](data:image/png;base64,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)

outliers:

```
BvD.ID.number
FR528648892                          CHAUSSON MATERIAUX
FR524237351    S A S LOT AGRICULTURE ET ENERGIE SOLAIRE
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for No.of.recorded.branch.locations
Welch's t-test statistic = -3.193
p-value = 0.001417


Optimization terminated successfully.
         Current function value: 0.155622
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0002588
Time:                        15:26:33   Log-Likelihood:                -18027.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                  0.002250
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2833      0.016   -208.582      0.000      -3.314      -3.252
BRA            0.0107      0.003      3.303      0.001       0.004       0.017
==============================================================================
```

### Number.of.employees.2010

![img](Wed, 14 Oct 2020 161420.png)

![image-20200624145031203](image-20200624145031203.png)

outlier:

```
BvD.ID.number
GB07158140    CARE UK HEALTH & SOCIAL CARE INVESTMENTS LIMITED
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Number.of.employees.2010
Welch's t-test statistic =  8.77
p-value = 2.022e-18

Optimization terminated successfully.
         Current function value: 0.155502
         Iterations 9
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:                0.001025
Time:                        15:41:13   Log-Likelihood:                -18013.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                 1.198e-09
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2594      0.016   -201.417      0.000      -3.291      -3.228
E             -0.0070      0.001     -4.796      0.000      -0.010      -0.004
==============================================================================
```

### Innovation.strength...Number.of.patents

![img](Wed, 14 Oct 2020 161426.png)

sum:![image-20200624145120622](image-20200624145120622.png)

mean:

![image-20200624145157993](image-20200624145157993.png)

outliers:

```
BvD.ID.number
IE507678        HORIZON THERAPEUTICS PUBLIC LIMITED COMPANY
DE8190460728                                 WAVELIGHT GMBH
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Number.of.patents
Welch's t-test statistic = -2.068
p-value = 0.03868

Optimization terminated successfully.
         Current function value: 0.155654
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               5.332e-05
Time:                        15:41:34   Log-Likelihood:                -18031.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                    0.1655
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2822      0.016   -208.621      0.000      -3.313      -3.251
PAT            0.0034      0.002      1.641      0.101      -0.001       0.008
==============================================================================
```

### Innovation.strength...Number.of.inventions

![img](Wed, 14 Oct 2020 161434.png)

sum:![image-20200624145308653](image-20200624145308653.png)

mean:

![image-20200624145347838](image-20200624145347838.png)

outlier:

```
BvD.ID.number
DE8190460728    WAVELIGHT GMBH
Name: Company.name, dtype: string

```

```
HGF vs non-HGF for Innovation.strength...Number.of.inventions
Welch's t-test statistic = -1.361
p-value = 0.1737

Optimization terminated successfully.
         Current function value: 0.155656
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               3.980e-05
Time:                        15:41:56   Log-Likelihood:                -18031.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                    0.2309
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2824      0.016   -208.594      0.000      -3.313      -3.252
INV            0.0141      0.010      1.425      0.154      -0.005       0.034
==============================================================================
```

### Number.of.patents

![img](Wed, 14 Oct 2020 161441.png)

sum:

![image-20200624145435096](image-20200624145435096.png)

mean:

![image-20200624145505067](image-20200624145505067.png)

outlier:

```
BvD.ID.number
DE8190460728    WAVELIGHT GMBH
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Number.of.patents
Welch's t-test statistic = -2.068
p-value = 0.03868

Optimization terminated successfully.
         Current function value: 0.155657
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               3.073e-05
Time:                        15:42:17   Log-Likelihood:                -18031.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                    0.2925
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2823      0.016   -208.605      0.000      -3.313      -3.251
P              0.0116      0.009      1.267      0.205      -0.006       0.029
==============================================================================
```

### Number.of.trademarks

![img](Wed, 14 Oct 2020 161449.png)

sum:

![image-20200624145636863](image-20200624145636863.png)

mean (Lituania!):

![image-20200624145542482](image-20200624145542482.png)

outliers:

```
BvD.ID.number
DE4250480683         VOLMARY GMBH
DE8190460728       WAVELIGHT GMBH
IT07237530964       SALROS S.R.L.
LULB157784       MEDA PHARMA SARL
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Number.of.trademarks
Welch's t-test statistic = -5.38
p-value = 7.824e-08

Optimization terminated successfully.
         Current function value: 0.155560
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0006523
Time:                        15:42:41   Log-Likelihood:                -18020.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                 1.234e-06
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2852      0.016   -208.489      0.000      -3.316      -3.254
T              0.0918      0.018      5.037      0.000       0.056       0.128
==============================================================================
```

### Trademarks...Type

```
BvD.ID.number
RO26941545       No
PT509482171      No
IT06789671218    No
FR519321806      No
NO995113562      No
Name: Trademarks...Type, dtype: category
Categories (4, object): [Figurative, No, Other, Word]
```

| Trademarks...Type | n          |
| ----------------- | ---------- |
| Figurative        | 779        |
| Word              | 400        |
| Other             | 3          |
| No                | 114658     |
| **Total**         | **115840** |

### lat / lon

Geographic coordinates of the firm. Not available for ~8000 samples.

![image-20200624121029091](image-20200624121029091.png)

![image-20200624121054469](image-20200624121054469.png)

![image-20200624121137893](image-20200624121137893.png)

HGF=True:

![image-20200624121213148](image-20200624121213148.png)

(These maps are navigable like Google Maps on my pc, I can provide an interactive version.)

### trust 

Trust evaluation, as string made of two parts. 

* A class {Low, Medium, High}
* A firm name, maybe the evaluator?

```
BvD.ID.number
RO26547053                                                    <NA>
HR05541138225    Medium: [DAKOVO, 31400, Croatia] instead of [C...
IT06738671210    High: [ESSECI ITALIA S.R.L., POGGIOMARINO, 800...
ESB72172646          High: [NUMENTI SL, PUERTO REAL, 11510, Spain]
IT02439920352      High: [SAN TOMMASO S.R.L., CANTU, 22063, Italy]
Name: trust, dtype: string
```

### trustVal

To give meaning to `trust`, We have transformed the variable, creating a variable with the ranked level of trust. Class labels are in {'Low','Med','Hig'}.

```
BvD.ID.number
IT04844800658    Hig
SE5568281116     Med
IT07016280963    Med
GB07234353       NaN
FR519497507      Hig
Name: trustVal, dtype: category
Categories (3, object): [Low < Med < Hig]
```

![img](Wed, 14 Oct 2020 161505.png)

Percentage of companies with "High" rating:

![image-20200624150153685](image-20200624150153685.png)

Percentage of companies with "Low" rating (*note the different y axis*):

![image-20200624150445416](image-20200624150445416.png)

mean trust: (high = 2, low = 0)

![image-20200624152916462](image-20200624152916462.png)



## Balance sheet

This section includes informations about the balance sheet of the company. It represents the description of equities, liabilities and assets of a company. In the following graph we reconstructed the way different variables are obtained by linear combinations of other variables. For every variable included in the balance sheet, we repeated this graph, or part of it, to point out every variable role in the balance sheet itself.

```mermaid
graph LR
subgraph A[ASSETS]
    intF[Intangible.fixed.assets.th.EUR.2010] --> |+| F[Fixed.assets.th.EUR.2010]
    tF[Tangible.fixed.assets.th.EUR.2010] --> F
    oF[Other.fixed.assets.th.EUR.2010]-->F
    S[Stock.th.EUR.2010] --> C[Current.assets.th.EUR.2010]
    OC[Other.current.assets.th.EUR.2010]--> C
    D[Debtors.th.EUR.2010]--> C
    F--> assets[Total.assets.th.EUR.2010]
	C-->assets
end

subgraph 2EL[EQUITY AND LIABILITIES]
 	subgraph EQ[EQUITY]
  		subgraph SHF[SHAREHOLDERS FUNDS]
   			Ca[Capital.th.EUR.2010]-->sf[Shareholders.funds.th.EUR.2010]
   			osf[Other.shareholders.funds.th.EUR.2010]-->sf
  		end
  		pl[P.L.for.period...Net.income..th.EUR.2010]
     end
     subgraph LI[LIABILITIES]
     	ltd[Long.term.debt.th.EUR.2010]--> ncl[Non.current.liabilities.th.EUR.2010]
     	oncl[Other.non.current.liabilities.th.EUR.2010] --> ncl
     	cr[Creditors.th.EUR.2010] --> cl[Current.liabilities.th.EUR.2010]
     	ocl[Other.current.liabilities.th.EUR.2010]-->cl
	end
end

style LI fill:#cfd, stroke:#000
style EQ fill:#cfd, stroke:#000
style SHF fill:#dfc, stroke:#000
style 2EL fill:#bfb, stroke:#000
style A fill:#bfb
```

### Assets

Assets are resources with an economic value controlled by a company. They can be bought or created. They include every entity that, now or in future, can generate cash flow, reduce expenses or improve sales, regardless of its nature.

#### Fixed.assets.th.EUR.2010 

```mermaid
graph LR
subgraph A[ASSETS]
    intF[Intangible.fixed.assets.th.EUR.2010] --> |+| F[Fixed.assets.th.EUR.2010]
    tF[Tangible.fixed.assets.th.EUR.2010] --> F
    oF[Other.fixed.assets.th.EUR.2010]-->F
    S[Stock.th.EUR.2010] --> C[Current.assets.th.EUR.2010]
    OC[Other.current.assets.th.EUR.2010]--> C
    D[Debtors.th.EUR.2010]--> C
    F--> assets[Total.assets.th.EUR.2010]
	C-->assets
end


style A fill:#bfb
style F fill:#fcc
```

A fixed asset is a long-term tangible piece of property or equipment  that a firm owns and uses in its operations to generate income. Fixed  assets are not expected to be consumed or converted into cash within a  year. Fixed assets most commonly appear on the balance sheet as [property, plant, and equipment](https://www.investopedia.com/terms/p/ppe.asp) (PP&E).

![img](data:image/png;base64,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)

```
outlier: 
BvD.ID.number
IE486605    SAP IRELAND US-FINANCIAL SERVICES DESIGNATED A...
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Fixed.assets.th.EUR.2010
Welch's t-test statistic = 3.679
p-value = 0.0002362

Optimization terminated successfully.
         Current function value: 0.155638
         Iterations 9
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0001559
Time:                        15:27:08   Log-Likelihood:                -18029.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                   0.01773
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2793      0.016   -207.847      0.000      -3.310      -3.248
FX         -6.137e-06   3.82e-06     -1.607      0.108   -1.36e-05    1.35e-06
==============================================================================
```

#### Intangible.fixed.assets.th.EUR.2010

```mermaid
graph LR
subgraph A[ASSETS]
    intF[Intangible.fixed.assets.th.EUR.2010] --> |+| F[Fixed.assets.th.EUR.2010]
    tF[Tangible.fixed.assets.th.EUR.2010] --> F
    oF[Other.fixed.assets.th.EUR.2010]-->F
    S[Stock.th.EUR.2010] --> C[Current.assets.th.EUR.2010]
    OC[Other.current.assets.th.EUR.2010]--> C
    D[Debtors.th.EUR.2010]--> C
    F--> assets[Total.assets.th.EUR.2010]
	C-->assets
end


style A fill:#bfb
style intF fill:#fcc
```

Intangible assets include operational assets that lack physical substance. For example, goodwill is a fixed asset, as are patents, copyrights, trademarks and franchises. A company's  intangible assets are often not reported on a company's financial  statements, or they may be reported at significantly less than their  actual value. This is because assets are accounted for at their  historical cost.

Unlike tangible fixed assets such as a building or machinery, intangibles are often  developed internally without any direct, measurable cost that can be  capitalized. When an intangible is purchased, however, or when costs can be directly traced to the development of the asset, the cost is  recorded as an intangible asset on the balance sheet.

Intangible assets are valued at their cost of acquisition. A purchased intangible is valued based on the amount paid  for the asset. Research and development costs associated with developing an intangible are expensed for the year in which they were incurred.

However, costs of registering patents or trademarks and legal fees incurred to  defend a company's right of use are included in the cost of acquisition, which is reported as an intangible asset on the balance sheet.

![img](data:image/png;base64,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)

x 10³ €

outlier:

```
BvD.ID.number
IT10969001006    LOTTERIE NAZIONALI S.R.L.
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Intangible.fixed.assets.th.EUR.2010
Welch's t-test statistic = 1.822
p-value = 0.06857
```

```
Optimization terminated successfully.
         Current function value: 0.155657
         Iterations 9
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               3.442e-05
Time:                        15:28:39   Log-Likelihood:                -18031.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                    0.2652
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2814      0.016   -208.487      0.000      -3.312      -3.251
INT        -9.435e-06   1.24e-05     -0.760      0.447   -3.38e-05    1.49e-05
==============================================================================
```

#### Tangible.fixed.assets.th.EUR.2010

```mermaid
graph LR
subgraph A[ASSETS]
    intF[Intangible.fixed.assets.th.EUR.2010] --> |+| F[Fixed.assets.th.EUR.2010]
    tF[Tangible.fixed.assets.th.EUR.2010] --> F
    oF[Other.fixed.assets.th.EUR.2010]-->F
    S[Stock.th.EUR.2010] --> C[Current.assets.th.EUR.2010]
    OC[Other.current.assets.th.EUR.2010]--> C
    D[Debtors.th.EUR.2010]--> C
    F--> assets[Total.assets.th.EUR.2010]
	C-->assets
end


style A fill:#bfb
style tF fill:#fcc
```

Tangible fixed assets generally refer to assets that have a physical  value. Examples of this are your business premises, equipment, inventory and machinery. Tangible fixed assets have a market value that needs to  be accounted for when you file your annual accounts. Some of these  assets, for example computer equipment, will incur depreciation, which  needs to be factored into your accounts.

![img](data:image/png;base64,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)

outlier: 

```
BvD.ID.number
GB07145051    CAPITAL & COUNTIES PROPERTIES PLC
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Tangible.fixed.assets.th.EUR.2010
Welch's t-test statistic = 1.394
p-value = 0.1635

Optimization terminated successfully.
         Current function value: 0.155655
         Iterations 8
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               4.600e-05
Time:                        15:29:04   Log-Likelihood:                -18031.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                    0.1977
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2808      0.016   -208.167      0.000      -3.312      -3.250
T          -4.673e-06   4.82e-06     -0.969      0.332   -1.41e-05    4.78e-06
==============================================================================
```

#### Other.fixed.assets.th.EUR.2010

```mermaid
graph LR
subgraph A[ASSETS]
    intF[Intangible.fixed.assets.th.EUR.2010] --> |+| F[Fixed.assets.th.EUR.2010]
    tF[Tangible.fixed.assets.th.EUR.2010] --> F
    oF[Other.fixed.assets.th.EUR.2010]-->F
    S[Stock.th.EUR.2010] --> C[Current.assets.th.EUR.2010]
    OC[Other.current.assets.th.EUR.2010]--> C
    D[Debtors.th.EUR.2010]--> C
    F--> assets[Total.assets.th.EUR.2010]
	C-->assets
end


style A fill:#bfb
style oF fill:#fcc
```



![img](Wed, 14 Oct 2020 161211.png)

outlier:

```
BvD.ID.number
IE486605    SAP IRELAND US-FINANCIAL SERVICES DESIGNATED A...
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Other.fixed.assets.th.EUR.2010
Welch's t-test statistic = 4.793
p-value = 1.65e-06

Optimization terminated successfully.
         Current function value: 0.155642
         Iterations 10
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0001271
Time:                        15:29:34   Log-Likelihood:                -18030.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                   0.03230
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2804      0.016   -208.259      0.000      -3.311      -3.249
OF         -1.366e-05   1.07e-05     -1.272      0.203   -3.47e-05    7.38e-06
==============================================================================
```

#### Current.assets.th.EUR.2010

```mermaid
graph LR
subgraph A[ASSETS]
    intF[Intangible.fixed.assets.th.EUR.2010] --> |+| F[Fixed.assets.th.EUR.2010]
    tF[Tangible.fixed.assets.th.EUR.2010] --> F
    oF[Other.fixed.assets.th.EUR.2010]-->F
    S[Stock.th.EUR.2010] --> C[Current.assets.th.EUR.2010]
    OC[Other.current.assets.th.EUR.2010]--> C
    D[Debtors.th.EUR.2010]--> C
    F--> assets[Total.assets.th.EUR.2010]
	C-->assets
end


style A fill:#bfb
style C fill:#fcc
```



![img](Wed, 14 Oct 2020 161214.png)

```
outlier:
BvD.ID.number
GB07450219    LONG ISLAND ASSETS LIMITED
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Current.assets.th.EUR.2010
Welch's t-test statistic = 6.732
p-value = 1.687e-11
```

```
Optimization terminated successfully.
         Current function value: 0.155487
         Iterations 11
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:                0.001126
Time:                        15:30:01   Log-Likelihood:                -18012.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                 1.856e-10
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2613      0.016   -201.717      0.000      -3.293      -3.230
CA            -0.0001    2.7e-05     -4.235      0.000      -0.000   -6.15e-05
==============================================================================
```

#### Stock.th.EUR.2010

```mermaid
graph LR
subgraph A[ASSETS]
    intF[Intangible.fixed.assets.th.EUR.2010] --> |+| F[Fixed.assets.th.EUR.2010]
    tF[Tangible.fixed.assets.th.EUR.2010] --> F
    oF[Other.fixed.assets.th.EUR.2010]-->F
    S[Stock.th.EUR.2010] --> C[Current.assets.th.EUR.2010]
    OC[Other.current.assets.th.EUR.2010]--> C
    D[Debtors.th.EUR.2010]--> C
    F--> assets[Total.assets.th.EUR.2010]
	C-->assets
end


style A fill:#bfb
style S fill:#fcc
```



![img](Wed, 14 Oct 2020 161218.png)

outlier:

```
BvD.ID.number
IT07099900966    MILANOSESTO S.P.A.
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Stock.th.EUR.2010
Welch's t-test statistic = 10.69
p-value = 1.155e-26

Optimization terminated successfully.
         Current function value: 0.155221
         Iterations 11
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:                0.002831
Time:                        15:30:21   Log-Likelihood:                -17981.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                 5.302e-24
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2474      0.016   -201.604      0.000      -3.279      -3.216
S             -0.0015      0.000     -6.485      0.000      -0.002      -0.001
==============================================================================
```

#### Debtors.th.EUR.2010

```mermaid
graph LR
subgraph A[ASSETS]
    intF[Intangible.fixed.assets.th.EUR.2010] --> |+| F[Fixed.assets.th.EUR.2010]
    tF[Tangible.fixed.assets.th.EUR.2010] --> F
    oF[Other.fixed.assets.th.EUR.2010]-->F
    S[Stock.th.EUR.2010] --> C[Current.assets.th.EUR.2010]
    OC[Other.current.assets.th.EUR.2010]--> C
    D[Debtors.th.EUR.2010]--> C
    F--> assets[Total.assets.th.EUR.2010]
	C-->assets
end


style A fill:#bfb
style D fill:#fcc
```

![img](Wed, 14 Oct 2020 161226.png)

outlier: 

```
BvD.ID.number
GB07246104    MACSCO 22 LIMITED
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Debtors.th.EUR.2010
Welch's t-test statistic = 7.135
p-value = 1.014e-12

Optimization terminated successfully.
         Current function value: 0.155553
         Iterations 10
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0007008
Time:                        15:30:49   Log-Likelihood:                -18019.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                 4.972e-07
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2697      0.016   -204.720      0.000      -3.301      -3.238
DEB           -0.0002    7.3e-05     -3.356      0.001      -0.000      -0.000
==============================================================================
```

#### Other.current.assets.th.EUR.2010

```mermaid
graph LR
subgraph A[ASSETS]
    intF[Intangible.fixed.assets.th.EUR.2010] --> |+| F[Fixed.assets.th.EUR.2010]
    tF[Tangible.fixed.assets.th.EUR.2010] --> F
    oF[Other.fixed.assets.th.EUR.2010]-->F
    S[Stock.th.EUR.2010] --> C[Current.assets.th.EUR.2010]
    OC[Other.current.assets.th.EUR.2010]--> C
    D[Debtors.th.EUR.2010]--> C
    F--> assets[Total.assets.th.EUR.2010]
	C-->assets
end


style A fill:#bfb
style OC fill:#fcc
```



![img](Wed, 14 Oct 2020 161229.png)

outlier:

```
BvD.ID.number
GB07450219    LONG ISLAND ASSETS LIMITED
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Other.current.assets.th.EUR.2010
Welch's t-test statistic = 4.135
p-value = 3.56e-05

Optimization terminated successfully.
         Current function value: 0.155609
         Iterations 10
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0003421
Time:                        15:31:18   Log-Likelihood:                -18026.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                 0.0004442
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2750      0.016   -206.020      0.000      -3.306      -3.244
OCA        -6.435e-05   2.72e-05     -2.366      0.018      -0.000   -1.11e-05
==============================================================================
```

#### Total.assets.th.EUR.2010

```mermaid
graph LR
subgraph A[ASSETS]
    intF[Intangible.fixed.assets.th.EUR.2010] --> |+| F[Fixed.assets.th.EUR.2010]
    tF[Tangible.fixed.assets.th.EUR.2010] --> F
    oF[Other.fixed.assets.th.EUR.2010]-->F
    S[Stock.th.EUR.2010] --> C[Current.assets.th.EUR.2010]
    OC[Other.current.assets.th.EUR.2010]--> C
    D[Debtors.th.EUR.2010]--> C
    F--> assets[Total.assets.th.EUR.2010]
	C-->assets
end

style A fill:#bfb
style assets fill:#fcc
```



![img](Wed, 14 Oct 2020 161233.png)

outlier:

```
BvD.ID.number
GB07450219    LONG ISLAND ASSETS LIMITED
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Total.assets.th.EUR.2010
Welch's t-test statistic = 5.512
p-value = 3.645e-08

Optimization terminated successfully.
         Current function value: 0.155604
         Iterations 10
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0003744
Time:                        15:31:50   Log-Likelihood:                -18025.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                 0.0002385
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2751      0.016   -206.277      0.000      -3.306      -3.244
TA         -1.137e-05   4.72e-06     -2.406      0.016   -2.06e-05   -2.11e-06
==============================================================================
```

#### Shareholders.funds.th.EUR.2010

```mermaid
graph LR

subgraph 2EL[EQUITY AND LIABILITIES]
 	subgraph EQ[EQUITY]
  		subgraph SHF[SHAREHOLDERS FUNDS]
   			Ca[Capital.th.EUR.2010]-->sf[Shareholders.funds.th.EUR.2010]
   			osf[Other.shareholders.funds.th.EUR.2010]-->sf
  		end
  		pl[P.L.for.period...Net.income..th.EUR.2010]
     end
     subgraph LI[LIABILITIES]
     	
	end
end

style LI fill:#cfd, stroke:#000
style EQ fill:#cfd, stroke:#000
style SHF fill:#dfc, stroke:#000
style 2EL fill:#bfb, stroke:#000
style sf fill:#fcc

```





![img](data:image/png;base64,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)

![image-20200624142754096](/home/andrea/.config/Typora/typora-user-images/image-20200624142754096.png)

```
outlier:

BvD.ID.number
GB07450219    LONG ISLAND ASSETS LIMITED
Name: Company.name, dtype: string

HGF vs non-HGF for Shareholders.funds.th.EUR.2010
Welch's t-test statistic = 1.909
p-value = 0.05634

Optimization terminated successfully.
         Current function value: 0.155657
         Iterations 8
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               3.438e-05
Time:                        15:32:19   Log-Likelihood:                -18031.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                    0.2655
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2812      0.016   -208.351      0.000      -3.312      -3.250
SHA        -2.844e-06   3.41e-06     -0.834      0.404   -9.53e-06    3.84e-06
==============================================================================
```

### Equity

Equity represents the difference between assets and liabilities, or the amount of money that should be returned to shareholders if all assets were to be sold, and all the debts paid off.

#### Capital.th.EUR.2010

```mermaid
graph LR

subgraph 2EL[EQUITY AND LIABILITIES]
 	subgraph EQ[EQUITY]
  		subgraph SHF[SHAREHOLDERS FUNDS]
   			Ca[Capital.th.EUR.2010]-->sf[Shareholders.funds.th.EUR.2010]
   			osf[Other.shareholders.funds.th.EUR.2010]-->sf
  		end
  		pl[P.L.for.period...Net.income..th.EUR.2010]
     end
     subgraph LI[LIABILITIES]
     	
	end
end

style LI fill:#cfd, stroke:#000
style EQ fill:#cfd, stroke:#000
style SHF fill:#dfc, stroke:#000
style 2EL fill:#bfb, stroke:#000
style Ca fill:#fcc

```



![img](data:image/png;base64,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)

![image-20200624142543360](/home/andrea/.config/Typora/typora-user-images/image-20200624142543360.png)

outlier: 

```
BvD.ID.number
FR519720643    IRIDIUM FRANCE
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Capital.th.EUR.2010
Welch's t-test statistic = 4.127
p-value = 3.699e-05

Optimization terminated successfully.
         Current function value: 0.155645
         Iterations 9
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0001079
Time:                        15:32:46   Log-Likelihood:                -18030.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                   0.04852
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2801      0.016   -208.097      0.000      -3.311      -3.249
CAP        -1.822e-05   1.36e-05     -1.337      0.181   -4.49e-05    8.49e-06
==============================================================================
```

#### Other.shareholders.funds.th.EUR.2010

```mermaid
graph LR

subgraph 2EL[EQUITY AND LIABILITIES]
 	subgraph EQ[EQUITY]
  		subgraph SHF[SHAREHOLDERS FUNDS]
   			Ca[Capital.th.EUR.2010]-->sf[Shareholders.funds.th.EUR.2010]
   			osf[Other.shareholders.funds.th.EUR.2010]-->sf
  		end
  		pl[P.L.for.period...Net.income..th.EUR.2010]
     end
     subgraph LI[LIABILITIES]
     	
	end
end

style LI fill:#cfd, stroke:#000
style EQ fill:#cfd, stroke:#000
style SHF fill:#dfc, stroke:#000
style 2EL fill:#bfb, stroke:#000
style osf fill:#fcc

```



![img](Wed, 14 Oct 2020 161237.png)

![image-20200624142847135](image-20200624142847135.png)

42359 companies has negative values in the range (-)

outlier:

```
BvD.ID.number
GB07450219    LONG ISLAND ASSETS LIMITED
Name: Company.name, dtype: string
```

Distribution of negative values:

![img](Wed, 14 Oct 2020 161242.png)

outlier:

```
vD.ID.number
FR519720643    IRIDIUM FRANCE
Name: Company.name, dtype: string BvD.ID.number
FR519720643   -502140.0
Name: Other.shareholders.funds.th.EUR.2010, dtype: float64
```

```
HGF vs non-HGF for Other.shareholders.funds.th.EUR.2010
Welch's t-test statistic = 0.8908
p-value = 0.3731

Optimization terminated successfully.
         Current function value: 0.155661
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               4.778e-06
Time:                        15:33:13   Log-Likelihood:                -18032.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                    0.6781
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2818      0.016   -208.591      0.000      -3.313      -3.251
OSF        -6.672e-07      2e-06     -0.334      0.739   -4.59e-06    3.25e-06
==============================================================================

```

### Liabilities

Liabilities are what a company typically owes or needs to pay to keep the company  running. Debt, including long-term debt, is a liability, as are rent, taxes, utilities, salaries, wages, and dividends payable.

#### Non.current.liabilities.th.EUR.2010

```mermaid
graph LR

subgraph 2EL[EQUITY AND LIABILITIES]
 	subgraph EQ[EQUITY]
  		
     end
     subgraph LI[LIABILITIES]
     	ltd[Long.term.debt.th.EUR.2010]--> ncl[Non.current.liabilities.th.EUR.2010]
     	oncl[Other.non.current.liabilities.th.EUR.2010] --> ncl
     	cr[Creditors.th.EUR.2010] --> cl[Current.liabilities.th.EUR.2010]
     	ocl[Other.current.liabilities.th.EUR.2010]-->cl
	end
end

style LI fill:#cfd, stroke:#000
style EQ fill:#cfd, stroke:#000
style 2EL fill:#bfb, stroke:#000
style ncl fill:#fcc
```



![img](Wed, 14 Oct 2020 161246.png)



![image-20200624142939181](image-20200624142939181.png)

```
HGF vs non-HGF for Non.current.liabilities.th.EUR.2010
Welch's t-test statistic = 7.868
p-value = 3.731e-15

Optimization terminated successfully.
         Current function value: 0.155577
         Iterations 10
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0005448
Time:                        15:34:00   Log-Likelihood:                -18022.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                 9.307e-06
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2743      0.016   -206.612      0.000      -3.305      -3.243
NCL        -5.339e-05   1.92e-05     -2.774      0.006   -9.11e-05   -1.57e-05
==============================================================================

```

#### Long.term.debt.th.EUR.2010

```mermaid
graph LR

subgraph 2EL[EQUITY AND LIABILITIES]
 	subgraph EQ[EQUITY]
  		
     end
     subgraph LI[LIABILITIES]
     	ltd[Long.term.debt.th.EUR.2010]--> ncl[Non.current.liabilities.th.EUR.2010]
     	oncl[Other.non.current.liabilities.th.EUR.2010] --> ncl
     	cr[Creditors.th.EUR.2010] --> cl[Current.liabilities.th.EUR.2010]
     	ocl[Other.current.liabilities.th.EUR.2010]-->cl
	end
end

style LI fill:#cfd, stroke:#000
style EQ fill:#cfd, stroke:#000
style 2EL fill:#bfb, stroke:#000
style ltd fill:#fcc
```

![img](Wed, 14 Oct 2020 161252.png)

![image-20200624143023458](image-20200624143023458.png)

outlier:

```
BvD.ID.number
GB07251526    TESCO PROPERTY FINANCE 3 PLC
Name: Company.name, dtype: string
```

```
Optimization terminated successfully.
         Current function value: 0.143145
         Iterations 10
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:                83074
Model:                          Logit   Df Residuals:                    83072
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0004459
Time:                        15:34:25   Log-Likelihood:                -11892.
converged:                       True   LL-Null:                       -11897.
Covariance Type:            nonrobust   LLR p-value:                  0.001125
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.3885      0.020   -172.120      0.000      -3.427      -3.350
LTD        -4.824e-05   2.41e-05     -1.998      0.046   -9.56e-05   -9.12e-07
==============================================================================
```

#### Other.non.current.liabilities.th.EUR.2010

```mermaid
graph LR

subgraph 2EL[EQUITY AND LIABILITIES]
 	subgraph EQ[EQUITY]
  		
     end
     subgraph LI[LIABILITIES]
     	ltd[Long.term.debt.th.EUR.2010]--> ncl[Non.current.liabilities.th.EUR.2010]
     	oncl[Other.non.current.liabilities.th.EUR.2010] --> ncl
     	cr[Creditors.th.EUR.2010] --> cl[Current.liabilities.th.EUR.2010]
     	ocl[Other.current.liabilities.th.EUR.2010]-->cl
	end
end

style LI fill:#cfd, stroke:#000
style EQ fill:#cfd, stroke:#000
style 2EL fill:#bfb, stroke:#000
style oncl fill:#fcc
```



![img](Wed, 14 Oct 2020 161258.png)

![image-20200624143129081](image-20200624143129081.png)

outliers:

```
vD.ID.number
IE487769       SCF CAPITAL DESIGNATED ACTIVITY COMPANY
FR521029926                             DIACINE FRANCE
NO995633604                   INDUSTRIINVESTERINGER AS

Optimization terminated successfully.
         Current function value: 0.143132
         Iterations 10
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:                83074
Model:                          Logit   Df Residuals:                    83072
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0005412
Time:                        15:34:51   Log-Likelihood:                -11891.
converged:                       True   LL-Null:                       -11897.
Covariance Type:            nonrobust   LLR p-value:                 0.0003325
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.3869      0.020   -171.972      0.000      -3.426      -3.348
ONCL          -0.0001   4.63e-05     -2.467      0.014      -0.000   -2.35e-05
==============================================================================
```

#### Current.liabilities.th.EUR.2010

```mermaid
graph LR

subgraph 2EL[EQUITY AND LIABILITIES]
 	subgraph EQ[EQUITY]
  		
     end
     subgraph LI[LIABILITIES]
     	ltd[Long.term.debt.th.EUR.2010]--> ncl[Non.current.liabilities.th.EUR.2010]
     	oncl[Other.non.current.liabilities.th.EUR.2010] --> ncl
     	cr[Creditors.th.EUR.2010] --> cl[Current.liabilities.th.EUR.2010]
     	ocl[Other.current.liabilities.th.EUR.2010]-->cl
	end
end

style LI fill:#cfd, stroke:#000
style EQ fill:#cfd, stroke:#000
style 2EL fill:#bfb, stroke:#000
style cl fill:#fcc
```



![img](Wed, 14 Oct 2020 161304.png)

![image-20200624143214614](image-20200624143214614.png)

outliers:

```
BvD.ID.number
IE480184          ESB FINANCE DESIGNATED ACTIVITY COMPANY
IE486122                      ICG EOS LOAN FUND I LIMITED
GB07193500           PREMIER LOTTERIES CAPITAL UK LIMITED
GB07202475       PREMIER LOTTERIES INVESTMENTS UK LIMITED
IT10319310016                    INFRATRASPORTI.TO S.R.L.
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Current.liabilities.th.EUR.2010
Welch's t-test statistic = 7.677
p-value = 1.691e-14

Optimization terminated successfully.
         Current function value: 0.155527
         Iterations 10
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0008644
Time:                        15:35:15   Log-Likelihood:                -18016.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                 2.359e-08
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2648      0.016   -202.670      0.000      -3.296      -3.233
CL          -9.74e-05   2.52e-05     -3.868      0.000      -0.000    -4.8e-05
==============================================================================
```

#### Creditors.th.EUR.2010

```mermaid
graph LR

subgraph 2EL[EQUITY AND LIABILITIES]
 	subgraph EQ[EQUITY]
  		
     end
     subgraph LI[LIABILITIES]
     	ltd[Long.term.debt.th.EUR.2010]--> ncl[Non.current.liabilities.th.EUR.2010]
     	oncl[Other.non.current.liabilities.th.EUR.2010] --> ncl
     	cr[Creditors.th.EUR.2010] --> cl[Current.liabilities.th.EUR.2010]
     	ocl[Other.current.liabilities.th.EUR.2010]-->cl
	end
end

style LI fill:#cfd, stroke:#000
style EQ fill:#cfd, stroke:#000
style 2EL fill:#bfb, stroke:#000
style cr fill:#fcc
```



![img](Wed, 14 Oct 2020 161309.png)

![image-20200624143256753](image-20200624143256753.png)

outliers:

```
BvD.ID.number
GB07254605        ED BROKING GROUP LIMITED
IT10969001006    LOTTERIE NAZIONALI S.R.L.
Name: Company.name, dtype: string

Optimization terminated successfully.
         Current function value: 0.149458
         Iterations 10
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:                98541
Model:                          Logit   Df Residuals:                    98539
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:                0.001558
Time:                        15:35:36   Log-Likelihood:                -14728.
converged:                       True   LL-Null:                       -14751.
Covariance Type:            nonrobust   LLR p-value:                 1.211e-11
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.3077      0.018   -183.576      0.000      -3.343      -3.272
CR            -0.0005      0.000     -4.689      0.000      -0.001      -0.000
==============================================================================
```

#### Other.current.liabilities.th.EUR.2010

```mermaid
graph LR

subgraph 2EL[EQUITY AND LIABILITIES]
 	subgraph EQ[EQUITY]
  		
     end
     subgraph LI[LIABILITIES]
     	ltd[Long.term.debt.th.EUR.2010]--> ncl[Non.current.liabilities.th.EUR.2010]
     	oncl[Other.non.current.liabilities.th.EUR.2010] --> ncl
     	cr[Creditors.th.EUR.2010] --> cl[Current.liabilities.th.EUR.2010]
     	ocl[Other.current.liabilities.th.EUR.2010]-->cl
	end
end

style LI fill:#cfd, stroke:#000
style EQ fill:#cfd, stroke:#000
style 2EL fill:#bfb, stroke:#000
style ocl fill:#fcc
```



![img](Wed, 14 Oct 2020 161315.png)

![image-20200624143355837](image-20200624143355837.png)

outlier:

```
BvD.ID.number
IT10319310016    INFRATRASPORTI.TO S.R.L.
Name: Company.name, dtype: string

Optimization terminated successfully.
         Current function value: 0.148949
         Iterations 10
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:                97463
Model:                          Logit   Df Residuals:                    97461
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0006357
Time:                        15:36:15   Log-Likelihood:                -14517.
converged:                       True   LL-Null:                       -14526.
Covariance Type:            nonrobust   LLR p-value:                 1.727e-05
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.3277      0.018   -185.296      0.000      -3.363      -3.292
OCL           -0.0001   4.46e-05     -3.012      0.003      -0.000   -4.69e-05
==============================================================================
```

#### Cash...cash.equivalent.th.EUR.2010



![img](Wed, 14 Oct 2020 161320.png)

outliers:

400+K bin: 

```
BvD.ID.number
NO995216604    WALLENIUS WILHELMSEN ASA
Name: Company.name, dtype: string
```

300K bin:

```
BvD.ID.number
GB07123187            ACACIA MINING PLC
Name: Company.name, dtype: string
```

200K+ bin:

```
BvD.ID.number
BE0831465984                           XIX-INVEST
GB07145051      CAPITAL & COUNTIES PROPERTIES PLC
GB07254605               ED BROKING GROUP LIMITED
GB07283266              HIGHBRIDGE COBALT LIMITED
Name: Company.name, dtype: string

Optimization terminated successfully.
         Current function value: 0.157093
         Iterations 9
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               110446
Model:                          Logit   Df Residuals:                   110444
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0001870
Time:                        15:36:36   Log-Likelihood:                -17350.
converged:                       True   LL-Null:                       -17354.
Covariance Type:            nonrobust   LLR p-value:                   0.01086
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2645      0.016   -201.981      0.000      -3.296      -3.233
OCL        -9.186e-05   5.11e-05     -1.797      0.072      -0.000    8.33e-06
==============================================================================
```

## Income statement

Income statement is a company core financial statement that shows they profit and loss (P&L) over a period of time, defined as the composition of all expenses, profits and revenues, from operating and non-operating activities. It can has different granularity (year, month, season). For our dataset, yearly data are aggregated. It includes the Earnings Before Interest and Taxes (EBIT) and taxation. The following graph represent the composition of different variables belonging to the income statement.

```mermaid
graph TB
subgraph BS[INCOME STATEMENT]
	
	
    
    subgraph EB[EBIT]
    	subgraph Fin[FINANCIAL]
            Fr[Financial.revenue.th.EUR.2010] --> FPL[Financial.P.L.th.EUR.2010]
            Fe[Financial.expenses.th.EUR.2010] --> FPL
        end
        EBIT[Operating.P.L...EBIT..th.EUR.2010]
        oc[non-operating revenue] --> EBIT
        EBIT --> PLbt[P.L.before.tax.th.EUR.2010]
    	Sal[Sales.th.EUR.2010] --> Rt
    	boh[?]--> Rt
    	Rt[Operating.revenue..Turnover..th.EUR.2010  ????] --> EBIT
    end
    
    FPL --> PLbt
    subgraph T[TAXATION]
    	Tax[Taxation.th.EUR.2010] 
    end
    Tax -->PLat[P.L.after.tax.th.EUR.2010]
    PLbt-->PLat    
end

style BS fill:#cfd, stroke:#000
style Fin fill:#dfc, stroke:#000
style EB fill:#dfc, stroke:#000
style T fill:#dfc, stroke:#000
style oc stroke-dasharray: 3, 7, fill:#dfc
style boh stroke-dasharray: 3, 7, fill:#dfc
```

### Earnings Before Interest and Taxes (EBIT)

#### Operating.revenue..Turnover..th.EUR.2010

```mermaid
graph TB
subgraph BS[INCOME STATEMENT]
	
	
    
    subgraph EB[EBIT]
    	subgraph Fin[FINANCIAL]
            FPL[Financial.P.L.th.EUR.2010]
            
        end
        EBIT[Operating.P.L...EBIT..th.EUR.2010]
        oc[non-operating revenue] --> EBIT
        EBIT --> PLbt[P.L.before.tax.th.EUR.2010]
    	Sal[Sales.th.EUR.2010] --> Rt
    	boh[?]--> Rt
    	Rt[Operating.revenue..Turnover..th.EUR.2010  ????] --> EBIT
    end
    
    FPL --> PLbt
    subgraph T[TAXATION]
    	Tax[Taxation.th.EUR.2010] 
    end
    Tax -->PLat[P.L.after.tax.th.EUR.2010]
    PLbt-->PLat    
end

style BS fill:#cfd, stroke:#000
style Fin fill:#dfc, stroke:#000
style EB fill:#dfc, stroke:#000
style T fill:#dfc, stroke:#000
style oc stroke-dasharray: 3, 7, fill:#dfc
style boh stroke-dasharray: 3, 7, fill:#dfc
style Rt fill:#fcc, stroke:#000
```



![img](Wed, 14 Oct 2020 161328.png)





![image-20200624143439652](/home/andrea/.config/Typora/typora-user-images/image-20200624143439652.png)

```
HGF vs non-HGF for Operating.revenue..Turnover..th.EUR.2010
Welch's t-test statistic = 9.251
p-value = 2.661e-20

Optimization terminated successfully.
         Current function value: 0.155219
         Iterations 11
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:                0.002844
Time:                        15:37:05   Log-Likelihood:                -17981.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                 4.183e-24
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2318      0.017   -193.739      0.000      -3.265      -3.199
ORT           -0.0003   4.57e-05     -6.834      0.000      -0.000      -0.000
==============================================================================
```

#### Sales.th.EUR.2010

```mermaid
graph TB
subgraph BS[INCOME STATEMENT]
	
	
    
    subgraph EB[EBIT]
    	subgraph Fin[FINANCIAL]
            FPL[Financial.P.L.th.EUR.2010]
            
        end
        EBIT[Operating.P.L...EBIT..th.EUR.2010]
        oc[non-operating revenue] --> EBIT
        EBIT --> PLbt[P.L.before.tax.th.EUR.2010]
    	Sal[Sales.th.EUR.2010] --> Rt
    	boh[?]--> Rt
    	Rt[Operating.revenue..Turnover..th.EUR.2010  ????] --> EBIT
    end
    
    FPL --> PLbt
    subgraph T[TAXATION]
    	Tax[Taxation.th.EUR.2010] 
    end
    Tax -->PLat[P.L.after.tax.th.EUR.2010]
    PLbt-->PLat    
end

style BS fill:#cfd, stroke:#000
style Fin fill:#dfc, stroke:#000
style EB fill:#dfc, stroke:#000
style T fill:#dfc, stroke:#000
style oc stroke-dasharray: 3, 7, fill:#dfc
style boh stroke-dasharray: 3, 7, fill:#dfc
style Sal fill:#fcc, stroke:#000
```





![img](data:image/png;base64,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)

![image-20200624143528781](/home/andrea/.config/Typora/typora-user-images/image-20200624143528781.png)

outliers:

```
BvD.ID.number
CYC266578        HMS HYDRAULIC MACHINES & SYSTEMS GROUP PLC
GB07123187                                ACACIA MINING PLC
IT10813301008                                    EOS S.R.L.
Name: Company.name, dtype: string

Optimization terminated successfully.
         Current function value: 0.160957
         Iterations 11
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               102550
Model:                          Logit   Df Residuals:                   102548
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:                0.005149
Time:                        15:37:31   Log-Likelihood:                -16506.
converged:                       True   LL-Null:                       -16592.
Covariance Type:            nonrobust   LLR p-value:                 4.804e-39
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.1452      0.018   -177.808      0.000      -3.180      -3.111
SAL           -0.0007   8.11e-05     -9.127      0.000      -0.001      -0.001
==============================================================================
```

#### Operating.P.L...EBIT..th.EUR.2010

```mermaid
graph TB
subgraph BS[INCOME STATEMENT]
	
	
    
    subgraph EB[EBIT]
    	subgraph Fin[FINANCIAL]
            FPL[Financial.P.L.th.EUR.2010]
            
        end
        EBIT[Operating.P.L...EBIT..th.EUR.2010]
        oc[non-operating revenue] --> EBIT
        EBIT --> PLbt[P.L.before.tax.th.EUR.2010]
    	Sal[Sales.th.EUR.2010] --> Rt
    	boh[?]--> Rt
    	Rt[Operating.revenue..Turnover..th.EUR.2010  ????] --> EBIT
    end
    
    FPL --> PLbt
    subgraph T[TAXATION]
    	Tax[Taxation.th.EUR.2010] 
    end
    Tax -->PLat[P.L.after.tax.th.EUR.2010]
    PLbt-->PLat    
end

style BS fill:#cfd, stroke:#000
style Fin fill:#dfc, stroke:#000
style EB fill:#dfc, stroke:#000
style T fill:#dfc, stroke:#000
style oc stroke-dasharray: 3, 7, fill:#dfc
style boh stroke-dasharray: 3, 7, fill:#dfc
style EBIT fill:#fcc, stroke:#000
```



![img](data:image/png;base64,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)

![image-20200624143602472](image-20200624143602472.png)

Positive outliers:

```
BvD.ID.number
GB07123187                   ACACIA MINING PLC
NL50397931     ATLANTIC AURUM INVESTMENTS B.V.
NO995216604           WALLENIUS WILHELMSEN ASA
PT509444229     MOTA-ENGIL AFRICA - SGPS, S.A.
Name: Company.name, dtype: string
```

Negative outliers:

```
BvD.ID.number
IE507678             HORIZON THERAPEUTICS PUBLIC LIMITED COMPANY
PL301339040    SAMSUNG ELECTRONICS POLAND MANUFACTURING SP. Z...
RU65519055                 AKTSIONERNOE OBSHCHESTVO TATTEPLOSBYT
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Operating.P.L...EBIT..th.EUR.2010
Welch's t-test statistic = 4.539
p-value = 5.771e-06

Optimization terminated successfully.
         Current function value: 0.155585
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0004971
Time:                        15:37:58   Log-Likelihood:                -18023.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                 2.296e-05
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2814      0.016   -208.555      0.000      -3.312      -3.251
EBIT          -0.0001   3.08e-05     -4.079      0.000      -0.000   -6.54e-05
==============================================================================
```

#### Financial.revenue.th.EUR.2010

```mermaid
graph TB
subgraph BS[INCOME STATEMENT]
	
	
    
    subgraph EB[EBIT]
    	subgraph Fin[FINANCIAL]
            Fr[Financial.revenue.th.EUR.2010] --> FPL[Financial.P.L.th.EUR.2010]
            Fe[Financial.expenses.th.EUR.2010] --> FPL
            
        end
        EBIT[Operating.P.L...EBIT..th.EUR.2010]
        
        EBIT --> PLbt[P.L.before.tax.th.EUR.2010]
    	
    end
    
    FPL --> PLbt
    subgraph T[TAXATION]
    	Tax[Taxation.th.EUR.2010] 
    end
    Tax -->PLat[P.L.after.tax.th.EUR.2010]
    PLbt-->PLat    
end

style BS fill:#cfd, stroke:#000
style Fin fill:#dfc, stroke:#000
style EB fill:#dfc, stroke:#000
style T fill:#dfc, stroke:#000
style Fr fill:#fcc, stroke:#000
```



![img](Wed, 14 Oct 2020 161333.png)

![image-20200624143657215](image-20200624143657215.png)

outliers:

50K cluster:

```
BvD.ID.number
FR527925143                     SOFAQUE
NO996031454           DOLPHIN INVEST AS
Name: Company.name, dtype: string
```

200K cluster:

```
BvD.ID.number
NO995633604    INDUSTRIINVESTERINGER AS
Name: Company.name, dtype: string

Optimization terminated successfully.
         Current function value: 0.155644
         Iterations 9
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0001129
Time:                        15:38:23   Log-Likelihood:                -18030.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                   0.04361
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2807      0.016   -208.461      0.000      -3.312      -3.250
FR            -0.0003      0.000     -1.646      0.100      -0.001    5.98e-05
==============================================================================
```

#### Financial.expenses.th.EUR.2010

```mermaid
graph TB
subgraph BS[INCOME STATEMENT]
	
	
    
    subgraph EB[EBIT]
    	subgraph Fin[FINANCIAL]
            Fr[Financial.revenue.th.EUR.2010] --> FPL[Financial.P.L.th.EUR.2010]
            Fe[Financial.expenses.th.EUR.2010] --> FPL
            
        end
        EBIT[Operating.P.L...EBIT..th.EUR.2010]
        
        EBIT --> PLbt[P.L.before.tax.th.EUR.2010]
    	
    end
    
    FPL --> PLbt
    subgraph T[TAXATION]
    	Tax[Taxation.th.EUR.2010] 
    end
    Tax -->PLat[P.L.after.tax.th.EUR.2010]
    PLbt-->PLat    
end

style BS fill:#cfd, stroke:#000
style Fin fill:#dfc, stroke:#000
style EB fill:#dfc, stroke:#000
style T fill:#dfc, stroke:#000
style Fe fill:#fcc, stroke:#000
```



![img](Wed, 14 Oct 2020 161343.png)

![image-20200624144029764](image-20200624144029764.png)

positive outlier:

```
BvD.ID.number
FR519720643    IRIDIUM FRANCE
Name: Company.name, dtype: string
```

negative outlier:

```
BvD.ID.number
GB07145051    CAPITAL & COUNTIES PROPERTIES PLC
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Financial.expenses.th.EUR.2010
Welch's t-test statistic = 2.789
p-value = 0.005292

Optimization terminated successfully.
         Current function value: 0.155655
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               4.376e-05
Time:                        15:38:43   Log-Likelihood:                -18031.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                    0.2090
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2817      0.016   -208.616      0.000      -3.313      -3.251
FE         -2.787e-05   1.96e-05     -1.425      0.154   -6.62e-05    1.05e-05
==============================================================================
```

#### Financial.P.L.th.EUR.2010

```mermaid
graph TB
subgraph BS[INCOME STATEMENT]
	
	
    
    subgraph EB[EBIT]
    	subgraph Fin[FINANCIAL]
            Fr[Financial.revenue.th.EUR.2010] --> FPL[Financial.P.L.th.EUR.2010]
            Fe[Financial.expenses.th.EUR.2010] --> FPL
            
        end
        EBIT[Operating.P.L...EBIT..th.EUR.2010]
        
        EBIT --> PLbt[P.L.before.tax.th.EUR.2010]
    	
    end
    
    FPL --> PLbt
    subgraph T[TAXATION]
    	Tax[Taxation.th.EUR.2010] 
    end
    Tax -->PLat[P.L.after.tax.th.EUR.2010]
    PLbt-->PLat    
end

style BS fill:#cfd, stroke:#000
style Fin fill:#dfc, stroke:#000
style EB fill:#dfc, stroke:#000
style T fill:#dfc, stroke:#000
style FPL fill:#fcc, stroke:#000
```



![img](Wed, 14 Oct 2020 161349.png)

![image-20200624144318884](image-20200624144318884.png)

Outlier:

```
BvD.ID.number
FR519720643    IRIDIUM FRANCE
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Financial.P.L.th.EUR.2010
Welch's t-test statistic =  -1.5
p-value = 0.1337

Optimization terminated successfully.
         Current function value: 0.155661
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               5.126e-06
Time:                        15:39:06   Log-Likelihood:                -18032.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                    0.6672
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2819      0.016   -208.633      0.000      -3.313      -3.251
FPL         6.226e-06   1.53e-05      0.408      0.684   -2.37e-05    3.62e-05
==============================================================================
```

#### P.L.before.tax.th.EUR.2010

```mermaid
graph TB
subgraph BS[INCOME STATEMENT]
	
	
    
    subgraph EB[EBIT]
    	subgraph Fin[FINANCIAL]
            Fr[Financial.revenue.th.EUR.2010] --> FPL[Financial.P.L.th.EUR.2010]
            Fe[Financial.expenses.th.EUR.2010] --> FPL
            
        end
        EBIT[Operating.P.L...EBIT..th.EUR.2010]
        
        EBIT --> PLbt[P.L.before.tax.th.EUR.2010]
    	
    end
    
    FPL --> PLbt
    subgraph T[TAXATION]
    	Tax[Taxation.th.EUR.2010] 
    end
    Tax -->PLat[P.L.after.tax.th.EUR.2010]
    PLbt-->PLat    
end

style BS fill:#cfd, stroke:#000
style Fin fill:#dfc, stroke:#000
style EB fill:#dfc, stroke:#000
style T fill:#dfc, stroke:#000
style PLbt fill:#fcc, stroke:#000
```



![img](Wed, 14 Oct 2020 161355.png)

![image-20200624144409866](image-20200624144409866.png)

outlier: 

```
BvD.ID.number
FR519720643    IRIDIUM FRANCE
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for P.L.before.tax.th.EUR.2010
Welch's t-test statistic = 3.793
p-value = 0.000149

Optimization terminated successfully.
         Current function value: 0.155659
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               1.779e-05
Time:                        15:39:22   Log-Likelihood:                -18032.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                    0.4232
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2819      0.016   -208.631      0.000      -3.313      -3.251
FPL        -4.184e-06   4.37e-06     -0.957      0.339   -1.28e-05    4.39e-06
==============================================================================
```

### Taxation

This single voice aggregates all expenses for taxes.

#### Taxation.th.EUR.2010

```mermaid
graph TB
subgraph BS[INCOME STATEMENT]
	
	
    
    subgraph EB[EBIT]
    	subgraph Fin[FINANCIAL]
            Fr[Financial.revenue.th.EUR.2010] --> FPL[Financial.P.L.th.EUR.2010]
            Fe[Financial.expenses.th.EUR.2010] --> FPL
            
        end
        EBIT[Operating.P.L...EBIT..th.EUR.2010]
        
        EBIT --> PLbt[P.L.before.tax.th.EUR.2010]
    	
    end
    
    FPL --> PLbt
    subgraph T[TAXATION]
    	Tax[Taxation.th.EUR.2010] 
    end
    Tax -->PLat[P.L.after.tax.th.EUR.2010]
    PLbt-->PLat    
end

style BS fill:#cfd, stroke:#000
style Fin fill:#dfc, stroke:#000
style EB fill:#dfc, stroke:#000
style T fill:#dfc, stroke:#000
style Tax fill:#fcc, stroke:#000
```



![img](Wed, 14 Oct 2020 161400.png)

![image-20200624144450752](image-20200624144450752.png)

outlier:

```
BvD.ID.number
GB07123187    ACACIA MINING PLC
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for Taxation.th.EUR.2010
Welch's t-test statistic = 5.486
p-value = 4.128e-08

Optimization terminated successfully.
         Current function value: 0.155649
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               8.533e-05
Time:                        15:39:38   Log-Likelihood:                -18030.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                   0.07940
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2814      0.016   -208.591      0.000      -3.312      -3.251
FPL           -0.0002      0.000     -1.794      0.073      -0.000    1.68e-05
==============================================================================
```

### Net profits &loss 

#### P.L.after.tax.th.EUR.2010

```mermaid
graph TB
subgraph BS[INCOME STATEMENT]
	
	
    
    subgraph EB[EBIT]
    	subgraph Fin[FINANCIAL]
            Fr[Financial.revenue.th.EUR.2010] --> FPL[Financial.P.L.th.EUR.2010]
            Fe[Financial.expenses.th.EUR.2010] --> FPL
            
        end
        EBIT[Operating.P.L...EBIT..th.EUR.2010]
        
        EBIT --> PLbt[P.L.before.tax.th.EUR.2010]
    	
    end
    
    FPL --> PLbt
    subgraph T[TAXATION]
    	Tax[Taxation.th.EUR.2010] 
    end
    Tax -->PLat[P.L.after.tax.th.EUR.2010]
    PLbt-->PLat    
end

style BS fill:#cfd, stroke:#000
style Fin fill:#dfc, stroke:#000
style EB fill:#dfc, stroke:#000
style T fill:#dfc, stroke:#000
style PLat fill:#fcc, stroke:#000
```



![img](Wed, 14 Oct 2020 161405.png)

![image-20200624144819678](image-20200624144819678.png)

outlier:

```
BvD.ID.number
FR519720643    IRIDIUM FRANCE
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for P.L.after.tax.th.EUR.2010
Welch's t-test statistic = 3.352
p-value = 0.0008042

Optimization terminated successfully.
         Current function value: 0.155660
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               1.429e-05
Time:                        15:40:03   Log-Likelihood:                -18032.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                    0.4728
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2819      0.016   -208.632      0.000      -3.313      -3.251
PL         -3.991e-06   4.65e-06     -0.858      0.391   -1.31e-05    5.13e-06
==============================================================================
```

#### P.L.for.period...Net.income..th.EUR.2010

![img](Wed, 14 Oct 2020 161410.png)

![image-20200624144909655](image-20200624144909655.png)


outlier:

```
BvD.ID.number
FR519720643    IRIDIUM FRANCE
Name: Company.name, dtype: string
```

```
HGF vs non-HGF for P.L.for.period...Net.income..th.EUR.2010
Welch's t-test statistic = 3.123
p-value = 0.001791

Optimization terminated successfully.
         Current function value: 0.155660
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:               115840
Model:                          Logit   Df Residuals:                   115838
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               1.267e-05
Time:                        15:40:20   Log-Likelihood:                -18032.
converged:                       True   LL-Null:                       -18032.
Covariance Type:            nonrobust   LLR p-value:                    0.4991
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2819      0.016   -208.632      0.000      -3.313      -3.251
PL          -3.82e-06   4.75e-06     -0.804      0.421   -1.31e-05    5.49e-06
==============================================================================
```

### Other Income Statement variables

### Cash.flow.th.EUR.2010 

![img](Wed, 14 Oct 2020 161416.png)

![image-20200624144956889](image-20200624144956889.png)

outliers:

```
BvD.ID.number
GB07123187                     ACACIA MINING PLC
GB07145051     CAPITAL & COUNTIES PROPERTIES PLC
GB07140891                           ENQUEST PLC
PT509444229       MOTA-ENGIL AFRICA - SGPS, S.A.
Name: Company.name, dtype: string

Optimization terminated successfully.
         Current function value: 0.162521
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    HGF   No. Observations:                68021
Model:                          Logit   Df Residuals:                    68019
Method:                           MLE   Df Model:                            1
Date:                Mon, 29 Jun 2020   Pseudo R-squ.:               0.0007085
Time:                        15:40:47   Log-Likelihood:                -11055.
converged:                       True   LL-Null:                       -11063.
Covariance Type:            nonrobust   LLR p-value:                 7.517e-05
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.2206      0.020   -161.248      0.000      -3.260      -3.181
CF            -0.0001   3.46e-05     -4.110      0.000      -0.000   -7.44e-05
==============================================================================

```

# Appendix 2: other compared metrics

In this section we will show tables of results, analogue to the ones shown in Chapter 6.2, for other computed metrics on the same experimental setup.

## ROC-AUC

![image-20201014143318500](image-20201014143318500.png)



![image-20201014143338517](image-20201014143338517.png)



![image-20201014143524698](image-20201014143524698.png)

![image-20201014143547873](image-20201014143547873.png)

![image-20201014143620848](image-20201014143620848.png)

![image-20201014143643155](image-20201014143643155.png)

![image-20201014143712787](image-20201014143712787.png)

![image-20201014143745004](image-20201014143745004.png)

![image-20201014143808817](image-20201014143808817.png)

## $F_1$ score (mean of the two classes)

![image-20201014143957776](image-20201014143957776.png)

![image-20201014144022459](image-20201014144022459.png)

![image-20201014144048673](image-20201014144048673.png)

![image-20201014144114626](image-20201014144114626.png)

![image-20201014144139178](image-20201014144139178.png)

![image-20201014144200000](image-20201014144200000.png)

![image-20201014144226257](image-20201014144226257.png)

![image-20201014144251158](image-20201014144251158.png)

![image-20201014144314407](image-20201014144314407.png)

# Appendix 3: benchmark datasets descriptions

This appendix includes the information on the benchmark datasets, as reported from the source websites. Descriptions has been written by the dataset's author, so every "we" here is referred to the original authors.

## ecoli

**Data Set Information:**

The references below describe a predecessor to this  dataset and its development. They also give results (not cross-validated) for classification by a rule-based expert system with  that version of the dataset. 

Reference: "Expert Sytem for Predicting Protein Localization Sites  in Gram-Negative Bacteria", Kenta Nakai & Minoru Kanehisa,   PROTEINS: Structure, Function, and Genetics 11:95-110, 1991. 

Reference: "A Knowledge Base for Predicting Protein Localization  Sites in Eukaryotic Cells", Kenta Nakai & Minoru Kanehisa, Genomics  14:897-911, 1992.

**Attribute Information:**

1. Sequence Name: Accession number for the SWISS-PROT database 
1. mcg: McGeoch's method for signal sequence recognition.
1. gvh: von Heijne's method for signal sequence recognition. 
1. lip: von Heijne's Signal Peptidase II conensus sequence score. Binary attribute. 
1. chg: Presence of charge on N-terminus of predicted lipoproteins. Binary attribute. 
1. aac: score of discriminant analysis of the amino acid content of outer membrane and periplasmic proteins. 
1. alm1: score of the ALOM membrane spanning region prediction program. 
1. alm2: score of ALOM program after excluding putative cleavable signal regions from the sequence.



**Relevant Papers:**

[^1]: Paul Horton & Kenta Nakai: "A Probablistic Classification System for Predicting the Cellular Localization Sites of  Proteins".Intelligent Systems in Molecular Biology, 109-115. St. Louis,  USA 1996. 
## optical_digits

**Data Set Information:**

We used preprocessing programs made available by NIST  to extract normalized bitmaps of handwritten digits from a preprinted  form. From a total of 43 people, 30 contributed to the training set and  different 13 to the test set. 32x32 bitmaps are divided into  nonoverlapping blocks of 4x4 and the number of on pixels are counted in  each block. This generates an input matrix of 8x8 where each element is  an integer in the range 0..16. This reduces dimensionality and gives  invariance to small distortions. 

For info on NIST preprocessing routines, see M. D. Garris, J. L.  Blue, G. T. Candela, D. L. Dimmick, J. Geist, P. J. Grother, S. A.  Janet, and C. L. Wilson, NIST Form-Based Handprint Recognition System,  NISTIR 5469, 1994.

**Attribute Information:**

All input attributes are integers in the range 0..16. 
The last attribute is the class code 0..9

**Relevant Papers:**

[^1]: C. Kaynak (1995): Methods of Combining Multiple  Classifiers and Their Applications to Handwritten Digit Recognition, MSc Thesis, Institute of Graduate Studies in Science and Engineering,  Bogazici University. 
[^2]: E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika. 
## satimage

**Dataset informations**

The database consists of the multi-spectral values of pixels in 3x3  neighbourhoods in a satellite image, and the classification associated  with the central pixel in each neighbourhood. The aim is to predict this classification, given the multi-spectral values. In the sample  database, the class of a pixel is coded as a number.

One frame of Landsat MSS imagery consists of four digital images of  the same scene in different spectral bands. Two of these are in the  visible region (corresponding approximately to green and red regions of  the visible spectrum) and two are in the (near) infra-red. Each pixel is a 8-bit binary word, with 0 corresponding to black and 255 to white.  The spatial resolution of a pixel is about 80m x 80m. Each image  contains 2340 x 3380 such pixels.

The database is a (tiny) sub-area of a scene, consisting of 82 x 100  pixels. Each line of data corresponds to a 3x3 square neighbourhood of  pixels completely contained within the 82x100 sub-area. Each line  contains the pixel values in the four spectral bands (converted to  ASCII) of each of the 9 pixels in the 3x3 neighbourhood and a number  indicating the classification label of the central pixel.

Each pixel is categorized as one of the following classes:

1. red soil
1. cotton crop
1. grey soil
1. damp grey soil
1. soil with vegetation stubble
1. mixture class (all types present)
1. very damp grey soil

NB. There are no examples with class 6 in this dataset.

The data is given in random order and certain lines of data have been removed so you cannot reconstruct the original image from this dataset.

**Attribute information** 

There are 36 predictive attributes (= 4 spectral bands x 9 pixels in  neighborhood). In each line of data the four spectral values for the top-left pixel are given first followed by the four spectral values for the top-middle  pixel and then those for the top-right pixel, and so on with the pixels  read out in sequence left-to-right and top-to-bottom. Thus, the four  spectral values for the central pixel are given by attributes 17,18,19  and 20. If you like you can use only these four attributes, while  ignoring the others. This avoids the problem which arises when a 3x3  neighbourhood straddles a boundary.

In this version, the pixel values 0…255 are normalized around 0.

Note: it is unclear why the attributes are named Aattr - Fattr in  this version, since there are only 4 bands and 9 pixels, naming them A1, B1, C1, D1, A2, B2, C2, D2, … would have made more sense.

## pen_digits

**Data Set Information:**

Authors created a digit database by collecting 250 samples  from 44 writers. The samples written by 30 writers are used for  training, cross-validation and writer dependent testing, and the digits  written by the other 14 are used for writer independent testing. This database is also available in the UNIPEN format. 

Authors used a WACOM PL-100V pressure sensitive tablet with an integrated LCD display and a cordless stylus. The input and display areas are  located in the same place. Attached to the serial port of an Intel 486  based PC, it allows us to collect handwriting samples. The tablet sends  $x$ and $y$ tablet coordinates and pressure level values of the pen at  fixed time intervals (sampling rate) of 100 milliseconds.  

These writers are asked to write 250 digits in random order inside  boxes of 500 by 500 tablet pixel resolution.  Subject are monitored only during the first entry screens. Each screen contains five boxes with  the digits to be written displayed above. Subjects are told to write  only inside these boxes.  If they make a mistake or are unhappy with  their writing, they are instructed to clear the content of a box by  using an on-screen button. The first ten digits are ignored because most writers are not familiar with this type of input devices, but subjects  are not aware of this.  

In their study, authors use only ($x, y$) coordinate information. The stylus pressure level values are ignored. First we apply normalization  to make our representation invariant to translations and scale  distortions. The raw data that we capture from the tablet consist of  integer values between 0 and 500 (tablet input box resolution). The new  coordinates are such that the coordinate which has the maximum range  varies between 0 and 100. Usually $x$ stays in this range, since most  characters are taller than they are wide.   

In order to train and test our classifiers, authors need to represent digits as constant length feature vectors. A commonly used technique  leading to good results is resampling the $( x_t, y_t)$ points. Temporal resampling (points regularly spaced in time) or spatial resampling  (points regularly spaced in arc length) can be used here. Raw point data are already regularly spaced in time but the distance between them is  variable. Previous research showed that spatial resampling to obtain a  constant number of regularly spaced points on the trajectory yields much better performance, because it provides a better alignment between  points. Our resampling algorithm uses simple linear interpolation  between pairs of points. The resampled digits are represented as a sequence of T points $( x_t, y_t )_{t=1}^T$, regularly spaced in arc  length, as opposed to the input sequence, which is regularly spaced in  time. 

So, the input vector size is $2*T$, two times the number of points resampled. We considered spatial resampling to $T=8,12,16$ points in the experiments and found that T=8 gave the best trade-off between accuracy and complexity.

**Attribute Information:**

All input attributes are integers in the range 0..100. The last attribute is the class code 0..9

**Relevant Papers:**

[^1]: F. Alimoglu (1996)Combining Multiple Classifiers for  Pen-Based Handwritten Digit Recognition, MSc Thesis, Institute of  Graduate Studies in Science and Engineering, Bogazici University. 
[^2]: F. Alimoglu, E. Alpaydin "Methods of Combining Multiple Classifiers Based on Different Representations for Pen-based Handwriting  Recognition," Proceedings of the Fifth Turkish Artificial Intelligence  and Artificial Neural Networks Symposium (TAINN 96), June 1996,  Istanbul, Turkey. 

## abalone

**Data Set Information:**

Predicting the age of abalone from physical  measurements.  The age of abalone is determined by cutting the shell  through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task.  Other measurements,  which are easier to obtain, are used to predict the age.  Further  information, such as weather patterns and location (hence food  availability) may be required to solve the problem. 

From the original data examples with missing values were removed  (the majority having the predicted value missing), and the ranges of the continuous values have been scaled for use with an ANN (by dividing by  200).

**Attribute Information:**

Given is the attribute name, attribute type, the measurement unit and a brief description.  The number of rings is the  value to predict: either as a continuous value or as a classification  problem. 

| Name           | Data Type   | Measurement Unit | Description               |
| -------------- | ----------- | ---------------- | ------------------------- |
| Sex            | nominal     |                  | {M, F, I}                 |
| Diameter       | continuous  | mm               | perpendicular to length   |
| Height         | continuous  | mm               | with meat in shell        |
| Whole weight   | continuous  | g                | whole abalone             |
| Shucked weight | continuous  | g                | weight of meat            |
| Viscera weight | continuouos | g                | gut weight after bleeding |
| Shell weight   | continuous  | g                | after dried               |
| Rings          | integer     |                  | +1.5 gives age in years   |

**Relevant Papers:**

[^1]: Sam Waugh (1995) "Extending and benchmarking  Cascade-Correlation", PhD thesis, Computer Science Department,  University of Tasmania.
[^2]: David Clark Zoltan Schreter, Anthony Adams "A Quantitative Comparison of  Dystal and Backpropagation", submitted to the Australian Conference on  Neural Networks (ACNN'96).

## sick_euthyroid

**Data Set Information:**

​	A Thyroid database suited for training ANNs. It is one of a set of different datasets, for which only a general description is given. 2800 training (data) instances and 972 test instances. Plenty of missing data. 29 or so attributes, either Boolean or continuously-valued.  

2 additional databases, also from Ross Quinlan, are also here 

Hypothyroid.data and sick-euthyroid.data (the one we used). 
Quinlan believes that these databases have been corrupted.
Their format is highly similar to the other databases.  

**Attribute Information:**

N/A

**Relevant Papers:**

[^1]:Quinlan,J.R., Compton,P.J., Horn,K.A., &  Lazurus,L. (1986). Inductive knowledge acquisition: A case study. In  Proceedings of the Second Australian Conference on Applications of  Expert Systems.  Sydney, Australia. 
[^2]: Quinlan,J.R. (1986). Induction of decision trees. Machine Learning, 1, 81--106. 

## spectrometer

**Data Set Information:**

The Infra-Red Astronomy Satellite (IRAS) was the first attempt to map the full sky at infra-red wavelengths.  This could not  be done from ground observatories because large portions of the  infra-red spectrum is absorbed by the atmosphere.  The primary observing program was the full high resolution sky mapping performed by scanning  at 4 frequencies. The Low Resolution Observation (IRAS-LRS) program  observed high intensity sources over two continuous spectral bands.   This database derives from a subset of the higher quality LRS  observations taken between 12h and 24h right ascension.  

This database contains 531 high quality spectra derived from the  IRAS-LRS database.  The original data contained 100 spectral  measurements in each of two overlapping bands.  Of these, 44 blue band  and 49 red band channels contain usable flux measurements.  Only these  are included here.  The original spectral intensities values are  compressed to 4-digits, and each spectrum includes 5 rescaling  parameters.  We have used the LRS specified algorithm to rescale these  to units of spectral intensity (Janskys).  Total intensity differences  have been eliminated by normalizing each spectrum to a mean value of  5000. 
	 
This database was originally obtained for use in development and  testing of our AutoClass system for Bayesian classification.  We have  not retained any results from this development, having concentrated our  efforts of a 5425 element version of the same data.  Our classifications were based upon simultaneous modeling of all 93 spectral intensities.  With the larger database we were able to find classes that correspond  well with known spectral types associated with particular stellar types. We also found classes that match with the spectra expected of certain  stellar processes under investigation by Ames astronomers.  These  classes have considerably enlarged the set of stars being investigated  by those researchers.   

Original Data: 

The original Fortran data file is given in spectra-2.data.  The file spectra-2.head contains information about the .data file contents and  how to rescale the compressed spectral intensities. 



**Attribute Information:**

1. LRS-name: (Suspected format: 5 digits, "+" or "-", 4 digits) 
1. LRS-class: integer - The LRS-class values range from 0 - 99  with the 10's digit giving the basic class and the 1's digit giving the  subclass. These classes are based on features (peaks, valleys, and  trends) of the spectral curves. 
1. ID-type: integer 
1. Right-Ascension: float - Astronomical longitude. 1h = 15deg 
1. Declination: float - Astronomical lattitude. -90 <= Dec <= 90 
1. Scale Factor: float - Proportional to source strength 
1. Blue base 1: integer - linear rescaling coefficient 
1. Blue base 2: integer - linear rescaling coefficient 
1. Red base 1: integer - linear rescaling coefficient 
1. Red base 2: integer - linear rescaling coefficient 
1. fluxes from the following 44 blue-band channel wavelengths: (all given as floating point numerals) 
1. from 12 onwards: different wavelength signals

**Relevant Papers:**

[^1]: A NASA-Ames research group concerned with unsupervised learning tasks may have used this database during their empirical  studies of their algorithm/system (AUTOCLASS II).  See the 1988 Machine  Learning Conference Proceedings, 54-64, for a description of their  algorithm.

## car_eval_34

Similar to car_eval_4 (see below), but with 21 variables.

## isolet

**Data Set Information:**

This data set was generated as follows. 150 subjects spoke the name of each letter of the alphabet twice. Hence, we have 52 training examples from each speaker. The speakers are grouped into sets  of 30 speakers each, and are referred to as isolet1, isolet2, isolet3,  isolet4, and isolet5. The data appears in isolet1+2+3+4.data in  sequential order, first the speakers from isolet1, then isolet2, and so  on.  The test set, isolet5, is a separate file. 

You will note that 3 examples are missing.  I believe they were dropped due to difficulties in recording. I believe this is a good domain for a noisy, perceptual task.  It is also a very good domain for testing the scaling abilities of  algorithms. For example, C4.5 on this domain is slower than backpropagation!  I have formatted the data for C4.5 and provided a C4.5-style names file as well.

**Attribute Information:**

The features are described in the paper by Cole and  Fanty cited above.  The features include spectral coefficients; contour  features, sonorant features, pre-sonorant features, and post-sonorant  features.  Exact order of appearance of the features is not known.

**Relevant Papers:**

[^1]:  Fanty, M., Cole, R. (1991). Spoken letter  recognition.  In Lippman, R. P., Moody, J., and Touretzky, D. S. (Eds).  Advances in Neural Information Processing Systems 3.  San Mateo, CA:  Morgan Kaufmann. 
[^2]:  Dietterich, T. G., Bakiri, G. (1991) Error-correcting output codes: A general method for improving multiclass inductive learning programs.  Proceedings of the Ninth National Conference on Artificial Intelligence (AAAI-91), Anaheim, CA: AAAI Press. 
[^3]: Dietterich, T. G., Bakiri, G. (1994) Solving Multiclass Learning Problems via Error-Correcting Output Codes. 

## us_crime

**Data Set Information:**

  Many variables are included so that algorithms that  select or learn weights for attributes could be tested. However, clearly unrelated attributes were not included; attributes were picked if there was any plausible connection to crime (N=122), plus the attribute to be predicted (Per Capita Violent Crimes). The variables included in the  dataset involve the community, such as the percent of the population  considered urban, and the median family income, and involving law  enforcement, such as per capita number of police officers, and percent  of officers assigned to drug units.  

 The per capita violent crimes variable was calculated using  population and the sum of crime variables considered violent crimes in  the United States: murder, rape, robbery, and assault. There was  apparently some controversy in some states concerning the counting of  rapes. These resulted in missing values for rape, which resulted in  incorrect values for per capita violent crime. These cities are not  included in the dataset. Many of these omitted communities were from the midwestern USA. 

  Data is described below based on original values. All numeric data was normalized into the decimal range  0.00-1.00 using an Unsupervised, equal-interval binning method. Attributes retain their distribution and skew (hence for example the population attribute has a mean value of   0.06 because most communities are small). E.g. An attribute described as 'mean people per household' is actually the normalized (0-1) version of that value. 

  The normalization preserves rough ratios of values WITHIN an  attribute (e.g. double the value for double the population within the  available precision - except for extreme values (all values more than 3  SD above the mean are normalized to 1.00; all values more than 3 SD  below the mean are nromalized to  0.00)). 

  However, the normalization does not preserve relationships between values BETWEEN attributes (e.g. it would not be meaningful to compare  the value for whitePerCap with the value for blackPerCap for a  community) 

  A limitation was that the LEMAS survey was of the police  departments with at least 100 officers, plus a random sample of smaller  departments. For our purposes, communities not found in both census and  crime datasets were omitted. Many communities are missing LEMAS data. 

**Attribute informations**

There are too many attributes to report here. For details, check the dataset UCI repository at http://archive.ics.uci.edu/ml/datasets/communities+and+crime.

**Relevant Papers:**

  No published results using this specific dataset.  

  Related dataset used in Redmond and Baveja 'A data-driven software tool for enabling cooperative information sharing among police  departments' in European Journal of Operational Research 141 (2002)  660-678;  
  That article includes a description of the integration of the  three sources of data, however, this data is normalized differently and  more/different attributes are included. 

## yeast_ml8

**Data Set Information:**

Predicted Attribute: Localization site of protein. (non-numeric). The references below describe a predecessor to this dataset and its  development. They also give results (not cross-validated) for  classification by a rule-based expert system with that version of the  dataset. 

**Attribute Information:**

| Attribute     | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| Sequence Name | Accession number for the SWISS-PROT database                 |
| mcg           | McGeoch's method for signal sequence recognition             |
| gvh           | von Heijne's method for signal sequence recognition          |
| alm           | Score of the ALOM membrane spanning region prediction program |
| mit           | Score of discriminant analysis of the amino acid content  of the N-terminal region (20 residues long) of mitochondrial and  non-mitochondrial proteins |
| erl           | Presence of "HDEL" substring (thought to act as a signal  for retention in the endoplasmic reticulum lumen). Binary attribute |
| pox           | Peroxisomal targeting signal in the C-terminus               |
| vac           | Score of discriminant analysis of the amino acid content of vacuolar and extracellular proteins. |
| nuc           | Score of discriminant analysis of nuclear localization signals of nuclear and non-nuclear proteins |

**Relevant Papers:**

[^1]: "Expert Sytem for Predicting Protein Localization Sites  in Gram-Negative Bacteria", Kenta Nakai & Minoru Kanehisa,  PROTEINS: Structure, Function, and Genetics 11:95-110, 1991. 
[^2]: "A Knowledge Base for Predicting Protein Localization  Sites in Eukaryotic Cells", Kenta Nakai & Minoru Kanehisa, Genomics  14:897-911, 1992.http://rexa.info/paper/fbb500f26399f3ca970053524afd131478039353)



## scene

No additional information provided by the source.

[^1]:Matthew R. Boutell, Jiebo Luo, Xipeng Shen, and Christopher M. Brown.  Learning multi-label scene classification.  *Pattern Recognition*, 37(9):1757-1771, 2004.

## libras_move

**Data Set Information:**

The dataset (movement_libras) contains 15 classes of 24 instances each, where each class references to a hand movement type in LIBRAS. 

In the video pre-processing, a time normalization is carried out selecting 45 frames from each video, in according to an uniform distribution. In each frame, the centroid pixels of the segmented objects (the hand) are found, which compose the discrete version of the curve F with 45 points. All curves are normalized in the unitary space. 

In order to prepare these movements to be analysed by algorithms, we have carried out a mapping operation, that is, each curve F is mapped in a representation with 90 features, with representing the coordinates of movement. Some sub-datasets are offered in order to support comparisons of results. 

**Attribute Information:**

90 numeric (double) and 1 for the class (integer)

**Relevant Papers:**

[^1]:DIAS, D. B.; MADEO, R. C. B.; ROCHA, T.; BÍSCARO, H. H.; PERES, S. M..  
Hand Movement Recognition for Brazilian Sign Language: A Study Using Distance-Based Neural Networks. In: 2009 International Joint Conference on Neural Networks, 2009, Atlanta, GA.  
Proceedings of 2009 International Joint Conference on Neural Networks. Eau Claire, WI, USA : Documation LLC, 2009. p. 697-704.  Digital Object Identifier 10.1109/IJCNN.2009.5178917 

## thyroid_sick

Another of sick_thyroid data sets. For description, see at §12.6.

## coil_2000

**Data Set Information:**

Information about customers consists of 86 variables  and includes product usage data and socio-demographic data derived from  zip area codes. The data was supplied by the Dutch data mining company  Sentient Machine Research and is based on a real world business problem. The training set contains over 5000 descriptions of customers,  including the information of whether or not they have a caravan  insurance policy. A test set contains 4000 customers of whom only the  organisers know if they have a caravan insurance policy.  

The data dictionary at http://kdd.ics.uci.edu/databases/tic/dictionary.txt describes the variables used and their values.  

Note: All the variables starting with M are zipcode variables. They  give information on the distribution of that variable, e.g. Rented  house, in the zipcode area of the customer.  

One instance per line with tab delimited fields.  

TICDATA2000.txt: Dataset to train and validate prediction models and build a description (5822 customer records). Each record consists of 86 attributes, containing sociodemographic data (attribute 1-43) and  product ownership (attributes 44-86).The sociodemographic data is  derived from zip codes. All customers living in areas with the same zip  code have the same sociodemographic attributes. Attribute 86,  "CARAVAN:Number of mobile home policies", is the target variable.  

TICEVAL2000.txt: Dataset for predictions (4000 customer records). It has the same format as TICDATA2000.txt, only the target is missing.  Participants are supposed to return the list of predicted targets only.  All datasets are in tab delimited format. The meaning of the attributes  and attribute values is given below.  

TICTGTS2000.txt Targets for the evaluation set.  

**Attribute Information:**

N/A

**Relevant Papers:**

[^1]:P. van der Putten and M. van Someren (eds). CoIL  Challenge 2000: The Insurance Company Case. Published by Sentient  Machine Research, Amsterdam. Also a Leiden Institute of Advanced  Computer Science Technical Report 2000-09. June 22, 2000. 

## arrhythmia

**Data Set Information:**

This database contains 279 attributes, 206 of which are linear valued and the rest are nominal.  

Concerning the study of H. Altay Guvenir: "The aim is to distinguish between the presence and absence of cardiac arrhythmia and to classify  it in one of the 16 groups. Class 01 refers to 'normal' ECG classes 02  to 15 refers to different classes of arrhythmia and class 16 refers to  the rest of unclassified ones. For the time being, there exists a  computer program that makes such a classification. However there are  differences between the cardiolog's and the programs classification.  Taking the cardiolog's as a gold standard we aim to minimise this  difference by means of machine learning tools." 

The names and id numbers of the patients were recently removed from the database. 

**Attribute Information:**

Different attributes for ECG measurement.

**Relevant Papers:**

[^1]: H. Altay Guvenir, Burak Acar, Gulsen Demiroz, Ayhan  Cekin "A Supervised Machine Learning Algorithm for Arrhythmia Analysis."  Proceedings of the Computers in Cardiology Conference, Lund, Sweden,  1997. 

## solar_flare_m0

**Data Set Information:**

Notes: 

   -- The database contains 3 potential classes, one for the number  of times a certain type of solar flare occured in a 24 hour period. 
   -- Each instance represents captured features for 1 active region on the sun. 
   -- The data are divided into two sections. The second section  (flare.data2) has had much more error correction applied to the it, and  has consequently been treated as more reliable.



**Attribute Information:**

| Attribute                                                    | values                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Code for class (modified Zurich class)                       | (A,B,C,D,E,F,H)                                              |
| Code for largest spot size                                   | (X,R,S,A,H,K)                                                |
| Code for spot distribution                                   | (X,O,I,C)                                                    |
| Activity                                                     | (1 = reduced, 2 = unchanged)                                 |
| Evolution                                                    | (1 = decay, 2 = no growth, 3 = growth)                       |
| Previous 24 hour flare activity code                         | (1 = nothing as big as an M1, 2 = one M1, 3 = more activity than one M1) |
| Historically-complex                                         | (1 = Yes, 2 = No)                                            |
| Did region become historically complex  on this pass across the sun's disk | (1 = yes, 2 = no)                                            |
| Area                                                         | (1 = small, 2 = large)                                       |
| Area of the largest spot                                     | (1 = <=5, 2 = >5)                                            |
| **targets:**                                                 |                                                              |
| C-class flares production by this region in the following 24 hours (common flares) | Number                                                       |
| M-class flares production by this region in the following 24 hours (moderate flares) | Number                                                       |
| X-class flares production by this region in the following 24 hours (severe flares) | Number                                                       |

## oil

**Data Set Information:**

To the best of its authors' knowledge, this is the  first realistic and public dataset with rare undesirable real events in  oil wells that can be readily used as a benchmark dataset for  development of machine learning techniques related to inherent  difficulties of actual data. 

More information about the theory behind this dataset is available  in the paper 'A realistic and public dataset with rare undesirable real  events in oil wells' published in the Journal of Petroleum Science and  Engineering. Specific challenges (benchmarks) that practitioners and researchers can use together with the 3W dataset are defined and proposed in this  paper. 

The 3W dataset consists of 1,984 CSV files structured as follows.  Due to the limitation of GitHub, this dataset is kept in 7z files splitted automatically and saved in the data directory. Before using 3W  dataset, they must be decompressed. After that, the subdirectory names  are the instances' labels. Each file represents one instance. The  filename reveals its source. All files are standardized as follow. There are one observation per line and one series per column. Columns are  separated by commas and decimals are separated by periods. The first  column contains timestamps, the last one reveals the observations'  labels, and the other columns are the Multivariate Time Series (MTS)  (i.e. the instance itself). 

The 3W dataset's files are in [[Web Link\]](https://github.com/ricardovvargas/3w_dataset), but we believe that the 3W dataset's publication in the UCI Machine  Learning Repository benefits the machine learning community.

**Attribute Information:**

Pressure at the Permanent Downhole Gauge (PDG); 
Pressure at the Temperature and Pressure Transducer (TPT); 
Temperature at the TPT; 
Pressure upstream of the Production Choke (PCK); 
Temperature downstream of the PCK; 
Pressure downstream of the Gas Lift Choke (GLCK); 
Temperature downstream of the GLCK; 
Gas Lift flow.

**Relevant Papers:**

[^1]:Vargas, Ricardo Emanuel Vaz, et al. "A realistic and public dataset with rare undesirable real events in oil wells." *Journal of Petroleum Science and Engineering* 181 (2019): 106223. 

## car_eval_4

**Data Set Information:**

The model evaluates cars according  to the following concept nested structure: 

1. CAR: car acceptability
   1. PRICE: overall price 
      1. buying               buying price 
      1. maint                price of the maintenance 
   1. TECH                   technical characteristics 
      1. COMFORT              comfort 
         1. doors              number of doors 
         1. persons            capacity in terms of persons to carry 
         1. lug_boot           the size of luggage boot 
      1. safety               estimated safety of the car 

Input attributes are printed in lowercase. Besides the target  concept (CAR), the model includes three intermediate concepts: PRICE,  TECH, COMFORT. Every concept is in the original model related to its  lower level descendants by a set of examples. 

The Car Evaluation Database contains examples with the structural  information removed, i.e., directly relates CAR to the six input  attributes: buying, maint, doors, persons, lug_boot, safety. 

Because of known underlying concept structure, this database may be  particularly useful for testing constructive induction and structure  discovery methods. 

**Attribute Information:**

Class Values: : unacc, acc, good, vgood 

Attributes: 

| Attribute | values                |
| --------- | --------------------- |
| buying    | vhigh, high, med, low |
| maint     | vhigh, high, med, low |
| doors     | 2,3,4,5more           |
| persons   | 2,4,more              |
| lug_boot  | small, med, big       |
| safety    | low, med, high        |

**Relevant Papers:**

[^1]: M. Bohanec and V. Rajkovic. Knowledge acquisition and  explanation for multi-attribute decision making. In 8th Intl Workshop on Expert Systems and their Applications, Avignon, France. pages 59-78,  1988. 
[^2]: B. Zupan, M. Bohanec, I. Bratko, J. Demsar. Machine learning by function decomposition. ICML-97, Nashville, TN. 1997 (to appear) 

## wine_quality

**Data Set Information:**

The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.  Due to privacy and logistic  issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand,  wine selling price, etc.).

These datasets can be viewed as  classification or regression tasks.  The classes are ordered and not  balanced (e.g. there are many more normal wines than excellent or poor  ones). Outlier detection algorithms could be used to detect the few  excellent or poor wines. Also, we are not sure if all input variables  are relevant. So it could be interesting to test feature selection  methods.



**Attribute Information:**

For more information, read [Cortez et al., 2009]. 
Input variables (based on physicochemical tests): 
   1 - fixed acidity 
   2 - volatile acidity 
   3 - citric acid 
   4 - residual sugar 
   5 - chlorides 
   6 - free sulfur dioxide 
   7 - total sulfur dioxide 
   8 - density 
   9 - pH 
   10 - sulphates 
   11 - alcohol 
Output variable (based on sensory data):  
   12 - quality (score between 0 and 10)



**Relevant Papers:****Data Set Information:**

The objective is to identify each of a large number of black-and-white rectangular pixel displays as one of the 26 capital  letters in the English alphabet.  The character images were based on 20  different fonts and each letter within these 20 fonts was randomly  distorted to produce a file of 20,000 unique stimuli.  Each stimulus was converted into 16 primitive numerical attributes (statistical moments  and edge counts) which were then scaled to fit into a range of integer  values from 0 through 15.  We typically train on the first 16000 items  and then use the resulting model to predict the letter category for the  remaining 4000.  See the article cited above for more details.



**Attribute Information:**

​	 

1.	lettr	capital letter	(26 values from A to Z) 
2.	x-box	horizontal position of box	(integer) 
3.	y-box	vertical position of box	(integer) 
4.	width	width of box			(integer) 
5.	high 	height of box			(integer) 
6.	onpix	total # on pixels		(integer) 
7.	x-bar	mean x of on pixels in box	(integer) 
8.	y-bar	mean y of on pixels in box	(integer) 
9.	x2bar	mean x variance			(integer) 
10.	y2bar	mean y variance			(integer) 
11.	xybar	mean x y correlation		(integer) 
12.	x2ybr	mean of x * x * y		(integer) 
13.	xy2br	mean of x * y * y		(integer) 
14.	x-ege	mean edge count left to right	(integer) 
15.	xegvy	correlation of x-ege with y	(integer) 
16.	y-ege	mean edge count bottom to top	(integer) 
17.	yegvx	correlation of y-ege with x	(integer)



**Relevant Papers:**

[^1]: P. W. Frey and D. J. Slate. "Letter Recognition Using Holland-style Adaptive Classifiers". (Machine Learning Vol 6 #2 March 91)
[^2]: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J.  Reis. Modeling wine preferences by data mining from physicochemical  properties.  In Decision Support Systems, Elsevier, 47(4):547-553, 2009. 

## letter_img

**Data Set Information:**

The objective is to identify each of a large number of black-and-white rectangular pixel displays as one of the 26 capital  letters in the English alphabet.  The character images were based on 20  different fonts and each letter within these 20 fonts was randomly  distorted to produce a file of 20,000 unique stimuli.  Each stimulus was converted into 16 primitive numerical attributes (statistical moments  and edge counts) which were then scaled to fit into a range of integer  values from 0 through 15.  We typically train on the first 16000 items  and then use the resulting model to predict the letter category for the  remaining 4000.  See the article cited above for more details.

**Attribute Information:**

	 1.	lettr	capital letter	(26 values from A to Z) 	 
  	 2.	x-box	horizontal position of box	(integer) 
    	 3.	y-box	vertical position of box	(integer) 
      	 4.	width	width of box			(integer) 
        	 5.	high 	height of box			(integer) 
          	 6.	onpix	total # on pixels		(integer) 
            	 7.	x-bar	mean x of on pixels in box	(integer)
              	 8.	y-bar	mean y of on pixels in box	(integer) 
                	 9.	x2bar	mean x variance			(integer) 
                  	 10.	y2bar	mean y variance			(integer) 
                    	 11.	xybar	mean x y correlation		(integer) 
                      	 12.	x2ybr	mean of x * x * y		(integer)
                        	 13.	xy2br	mean of x * y * y		(integer) 
                          	 14.	x-ege	mean edge count left to right	(integer) 
                            	 15.	xegvy	correlation of x-ege with y	(integer)
                              	 16.	y-ege	mean edge count bottom to top	(integer) 
                                	 17.	yegvx	correlation of y-ege with x	(integer)



**Relevant Papers:**

[^1]: P. W. Frey and D. J. Slate. "Letter Recognition Using Holland-style Adaptive Classifiers". (Machine Learning Vol 6 #2 March 91) 

## yeast_me2

a different version of the yeast_ml8 dataset, with a different number of variables.

## webpage

**Data Set Information:**

One of the challenges faced by our research was the  unavailability of reliable training datasets. In fact this challenge  faces any researcher in the field. However, although plenty of articles  about predicting phishing websites have been disseminated these days, no reliable training dataset has been published publically, may be because there is no agreement in literature on the definitive features that  characterize phishing webpages, hence it is difficult to shape a dataset that covers all possible features.  
In this dataset, we shed light on the important features that have  proved to be sound and effective in predicting phishing websites. In  addition, we propose some new features.

**Attribute Information:**

N/A

**Relevant Papers:**

[^1] Mohammad, Rami, McCluskey, T.L. and Thabtah, Fadi  (2012) An Assessment of Features Related to Phishing Websites using an  Automated Technique. In: International Conferece For Internet Technology And Secured Transactions. ICITST 2012 . IEEE, London, UK, pp. 492-497.  ISBN 978-1-4673-5325-0

## ozone_level

**Data Set Information:**

For a list of attributes, please refer to those two .names files.  They use the following naming convention.  All the attribute start with T means the temperature measured at  different time throughout the day; and those starts with WS indicate the wind speed at various time. 

WSR_PK:     continuous. peek wind speed -- resultant (meaning average of wind vector) 
WSR_AV:     continuous. average wind speed 
T_PK:     continuous. Peak T 
T_AV:     continuous. Average T 
T85:     continuous. T at 850 hpa level (or about 1500 m height) 
RH85:     continuous. Relative Humidity at 850 hpa 
U85:     continuous. (U wind - east-west direction wind at 850 hpa) 
V85:     continuous. V wind - N-S direction wind at 850 
HT85:     continuous. Geopotential height at 850 hpa, it is about the same as height at low altitude 
T70:     continuous. T at 700 hpa level (roughly 3100 m height) 

RH70:     continuous. 
U70:     continuous. 
V70:     continuous. 
HT70:     continuous. 

T50:     continuous. T at 500 hpa level (roughly at 5500 m height) 

RH50:     continuous. 
U50:     continuous. 
V50:     continuous. 
HT50:     continuous. 

KI:     continuous. K-Index [[Web Link\]](http://www.weather.gov/glossary/index.php?letter=k) 
TT:     continuous. T-Totals [[Web Link\]](http://www.theweatherprediction.com/habyhints/302/) 
SLP:     continuous. Sea level pressure 
SLP_:     continuous. SLP change from previous day 

Precp:    continuous. -- precipitation



**Attribute Information:**

The following are specifications for several most  important attributes that are highly valued by Texas Commission on  Environmental Quality (TCEQ). More details can be found in the two  relevant papers. 

| Attribute | Description                                               |
| --------- | --------------------------------------------------------- |
| O3        | Local ozone peak prediction                               |
| Upwind    | Upwind ozone background level                             |
| EmFactor  | Precursor emissions related factor                        |
| Tmax      | Maximum temperature in degrees F                          |
| Tb        | Base temperature where net ozone production begins (50 F) |
| SRd       | Base temperature where net ozone production begins (50 F) |
| WSa       | Wind speed near sunrise (using 09-12 UTC forecast mode)   |
| WSp       | Wind speed mid-day (using 15-21 UTC forecast mode)        |

**Relevant Papers:**

[^1]: Forecasting skewed biased stochastic ozone days:  analyses, solutions and beyond, Knowledge and Information Systems, Vol.  14, No. 3, 2008. 

## mammography

**Data Set Information:**

Mammography is the most effective method for breast cancer screening available today. However, the low positive predictive value of breast biopsy resulting from mammogram interpretation leads to approximately 70% unnecessary biopsies with benign outcomes. To reduce the high number of unnecessary breast biopsies, several computer-aided diagnosis (CAD) systems have been proposed in the last years.These systems help physicians in their decision to perform a breast biopsy on a suspicious lesion seen in a mammogram or to perform a short term follow-up 
examination instead. 

This data set can be used to predict the severity (benign or malignant) of a mammographic mass lesion from BI-RADS attributes and the patient's age. It contains a BI-RADS assessment, the patient's age and three BI-RADS attributes together with the ground truth (the severity field) for 516 benign and 445 malignant masses that have been identified on full field digital mammograms collected at the Institute of Radiology of the University Erlangen-Nuremberg between 2003 and 2006. 

Each instance has an associated BI-RADS assessment ranging from 1 (definitely benign) to 5 (highly suggestive of malignancy) assigned in a double-review process by physicians. Assuming that all cases with BI-RADS assessments greater or equal a given value (varying from 1 to 5), are malignant and the other cases benign, sensitivities and associated specificities can be calculated. These can be an indication of how well a CAD system performs compared to the radiologists. 

**Attribute Information:**

6 Attributes in total (1 goal field, 1 non-predictive, 4 predictive attributes) 


1. BI-RADS assessment: 1 to 5 (ordinal, non-predictive!)   
1. Age: patient's age in years (integer) 
1. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal) 
1. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal) 
1. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal) 
1. Severity: benign=0 or malignant=1 (binominal, goal field!) 

Missing Attribute Values: 
    \- BI-RADS assessment :    2 
    \- Age:                   	     5 
    \- Shape:                         31 
    \- Margin:                       48 
    \- Density:                       76 
    \- Severity:                       0 

**Relevant Papers:**

[^1]: M. Elter, R. Schulz-Wendtland and T. Wittenberg (2007)  The prediction of breast cancer biopsy outcomes using two CAD approaches that both emphasize an intelligible decision process. Medical Physics 34(11), pp. 4164-4172

## protein_homo

Despite being included  in the proposed benchmark dataset set, no information could be retrieved for this dataset.

## abalone_19

A different version of the abalone dataset.