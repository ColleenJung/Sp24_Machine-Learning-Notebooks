# W1_What-Do-Machines-Learn

# Types of Learning Part 1

<img width="512" alt="Screenshot 2024-03-28 at 1 28 51 PM" src="https://github.com/ColleenJung/W1_What-Do-Machines-Learn-/assets/119357849/ec723f5d-e43d-43d4-a23b-d90979d1ad71">

## 1. Supervised learning

1. The majority of practical machine learning uses supervised learning.
2. Supervised learning is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output.
𝑦 = 𝑓(𝑥)
3. **The goal** is to approximate the mapping function so well that **when you have new input data 𝑥 that you can predict the
output variables 𝑦 for that data.**
4. It is called supervised learning because the process of an algorithm learning from the training dataset can be thought of
as a teacher supervising the learning process.
5. We know the correct answers, the algorithm iteratively makes predictions on the training data and is corrected by the teacher. Learning stops when the algorithm achieves an acceptable level of performance.

## 2. Unsupervised learning

1. Unsupervised learning is where you only have input data 𝑥 and no corresponding output variables.
2. **The goal for unsupervised learning** is to **model the underlying structure or distribution in the data in order to learn more
about the data.**
3. These are called unsupervised learning because unlike supervised learning above there is no correct answers and there is no teacher. Algorithms are left to their own devises to discover and present the interesting structure in the data.
4. Unsupervised Learning problems can be further grouped into *Clustering and Association Problems.*
5. Clustering: A clustering problem is where you want to discover the inherent groupings in the data, such as grouping customers by purchasing behavior.
6. Association: An association rule learning problem is where you want to discover rules that describe large portions of your data, such as people that buy A also tend to buy B

## 3. Semi-supervised learning

1. Semi-supervised learning is halfway between supervised and unsupervised learning.
2. Traditional classification methods use labelled data to build classifiers.
3. The labelled training sets used as input in Supervised learning is very certain and properly defined.
4. However, they are limited, expensive and takes a lot of time to generate them.
5. On the other hand, unlabeled data is cheap and is readily available in large volumes.
6. Hence, semi-supervised learning is learning from a combination of both labelled and unlabeled data;
7. Where we make use of a combination of small amount of labelled data and large amount of unlabeled data to increase the accuracy of our classifiers.

## 4. Self-supervised learning

1. Form of unsupervised learning where **the data** provides the supervision.
2. Uses labeled training data.
3. The labeling is autonomous and no manual (human) labeling is needed.
4. Well suited for online learning.

# Types of Learning Part 2

<img width="440" alt="Screenshot 2024-03-28 at 1 42 59 PM" src="https://github.com/ColleenJung/W1_What-Do-Machines-Learn-/assets/119357849/ab5fcef9-d904-4184-91be-66625733735b">

## 1. Reinforcement learning -optimizing(self-trained)

<img width="327" alt="Screenshot 2024-03-28 at 1 51 41 PM" src="https://github.com/ColleenJung/W1_What-Do-Machines-Learn-/assets/119357849/6dca9ebb-35ca-4e45-bcd6-511ac5c8e3fb">

1. Reinforcement learning is an area of MLK outside of supervised and unsupervised learning.
2. RL is about teaching an agent to learn which decisions to make in an environment to maximize some reward function.
3. During the learning process, the agent receives feedback based on the actions taken and aims to maximize the overall value acquired

## 2. Transfer Learning

<img width="230" alt="Screenshot 2024-03-28 at 1 53 19 PM" src="https://github.com/ColleenJung/W1_What-Do-Machines-Learn-/assets/119357849/1ac95218-8f01-4dc5-b4df-a829b3ff4f16">

1. A machine learning technique where a model trained on one task is re-purposed on a second related task.
2. An optimization that allows rapid progress or improved performance when modeling the second task.

## 3. Active Learning

<img width="309" alt="Screenshot 2024-03-28 at 1 54 09 PM" src="https://github.com/ColleenJung/W1_What-Do-Machines-Learn-/assets/119357849/d1bee5c8-e34a-428d-9d80-917b1f8c1044">

1. Active learning (sometimes called “query learning” or “optimal experimental design” in the statistics literature) is a subfield of machine learning and, more generally, artificial intelligence.
2. The key hypothesis is that if the learning algorithm is allowed to choose the data from which it learns—to be “curious,” if you will—it will perform better with less training.
3. Active learning is a special case of semi-supervised learning.

# ML Data Assumptions (same dist. & iid)

1. Training and test data are from the same probability distribution.
2. Training and test data are iid (independent and identically distributed)

# Universal Workflow of ML

1. Define the problem
2. Assemble dataset
3. Choose a metric to quantify project outcome
4. Decide on how to calculate the metric
5. Prepare dataset
6. Define standard baseline
7. Develop model that beats baseline
8. Ideal model is at the border of overfit and underfit – cross the border to know where it is so overfit model
9. Regularize model and tune hyperparameters

# Bias-Variance Tradeoff

<img width="259" alt="Screenshot 2024-03-28 at 1 56 54 PM" src="https://github.com/ColleenJung/W1_What-Do-Machines-Learn-/assets/119357849/db79b502-aa33-4fe9-b278-c908f20cdb04">

1. Bias – low capacity model that does not fit the training data well and has high training error – underfit
2. Variance – high capacity model that “learns” the training data too well and has high generalization error – overfit
3. Increasing (decreasing) a model’s capacity / complexity / degrees-of-freedom reduces (increases) bias + increases (decreases) variance
4. NOTE – **Mean Squared Error cost function incorporates both**

# Overfitting and Underfitting - noise(=detecting patterns in training data too much)

<img width="515" alt="Screenshot 2024-03-28 at 2 04 01 PM" src="https://github.com/ColleenJung/W1_What-Do-Machines-Learn-/assets/119357849/12001290-55c8-4942-b017-7f9bdf0a34fa">

1. **Overfitting** – model fits very well to the training data, **aka detects patterns in the noise** also such that the variance dominates the test data
  1. Detect:
    1. Low training error, high generalization error. - **overfit -detecting patterns too much in test data!**
  2. **Remedies:**
    1. Reduce model capacity by removing features and/or parameters.
    2. Get more training data.
    3. Improve training data quality by reducing noise.

2. **Underfitting** – model **too simple to detect patterns** in the data
  1. Detect
    1. High training error.
  2. **Remedies:**
    1. Increase model capacity by adding more parameters and/or features.
    2. Reduce model constraints.
     
<img width="389" alt="Screenshot 2024-03-28 at 2 05 33 PM" src="https://github.com/ColleenJung/W1_What-Do-Machines-Learn-/assets/119357849/a0825feb-54ae-434b-99ab-ddc1d73dc9d1">

- Goal:finding the sweet spot! where, tuning bias, not move var/tuning var, not move bias
- Bias-variance tradeoff & MSE: **MSE=(bias)^2+(var)+(irreducible error)**

# Bayes Classifier and Bayes Error

1. Bayes Error is the lowest possible error rate.
2. Bayes Classifier produces the lowest possible error rate, aka Bayes Error.
3. Bayes Classifier assumes knowledge of the conditional distribution of response given predictors (posterior probability) – i.e. we know everything these is to know, which does not happen in reality.
4. Bayes Classifier is the gold standard that classifiers try to approximate.
5. Bayes Error – another way to think about it – assume utopia where an oracle knows the true probability distribution that generates the data; there is still some noise in the distribution that causes test error = Bayes Error.

# Parametric & Nonparametric Models

Estimate the unknown function 𝑓 as 𝑓
• Parametric Models:
1. Assume the functional form or shape of 𝑓
2. Apply methodology to train model
3. Advantage – simple estimation መ
4. Disadvantage – 𝑓 may be far from true 𝑓
• Nonparametric Models:
1. **No assumption** on the functional form or shape of 𝑓 - **does not assume any specific form for the relationship between X and Y**
2. Estimate to fit as close as possible to the data
3. Advantage – can accurately fit a wide range of possible shapes of 𝑓
4. Disadvantage – need large datasets (since there is no fixed # of params to estimate)

# Linear Regression with OLS using scikit-learn

<img width="703" alt="Screenshot 2024-03-28 at 2 08 34 PM" src="https://github.com/ColleenJung/W1_What-Do-Machines-Learn-/assets/119357849/719931b6-005b-4a81-bd35-78712d5693ee">

• Advantage – equation is linear with size of training set so it can handle large training sets efficiently
• Disadvantage –
1. computational complexity of inverting a matrix that increases with size of training set
2. difficult to do online learning with new data arriving regularly (need to recalculate estimates), i.e. no iterative parameter updates

