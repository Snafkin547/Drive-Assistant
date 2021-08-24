# Drive Assistant

Drive Assistant (DA) is Python based Deep Learning Model, which classifies road images into two category: with and without potholes.
It deploys a derivative of GoogLeNet, has achieved 88.46% of accuracy of successful detection of potholes in road images without overfitting. 

![image](https://user-images.githubusercontent.com/62607343/130652346-2ae4c0af-28a2-4010-8713-5ff9ce711dd1.png)

The big picture of DA’s architecture consists of - Data Import from Kaggle, Pre-processing, CNN, and signals Drive Instruction. 

The main body of DA is GoogLeNet structure - convolutional layers, max-pooling layers, inception layers(a bunch of convolutional layers and a max-pooling layer), Global average pooling layer instead of flattening layer, and “sigmoid” function in the output layer. 

In order to improve accuracy, I made the following changes to the original architecture. 
First, SELU is used as activation function in place of ReLU and Leaky-ReLU, in order to avoid possible gradient saturation and the dying ReLU problem. SELU is applied to inception layers. Also, I changed the numbers of feature map outputs in the first two layers from 64 to 150. Furthermore, I added three fully connected layers (with 1000, 100 and 100 neurons in this sequence) prior to the output layer. 

These additions could potentially have exposed the model to the risk of overfitting. In fact, it did experience slight overfittings when I increased to 175 and 200 feature maps, and deteriorated its result to 82.10% (-4.10%) and 80.64% (-5.56%). Likewise, four additional fully connected layers (with 1000, 1000, 100 and 100 neurons) reduced accuracy to 83.02%(-3.18%). Therefore, the present model maintains 150 feature maps in the first two layers and three fully connected layers as the following diagram shows.
![image](https://user-images.githubusercontent.com/62607343/130650910-62b161bc-af75-4f02-b446-e8f50ef4f797.png)

**1) Data Ingestion**

DA accepts any size of road images without error and standardize it to (276, 368) size.
The following screenshots show the snippets of the data injection pipeline.
![image](https://user-images.githubusercontent.com/62607343/130651177-c25b47e1-7bdd-4d8a-a5eb-c7a3d6507b2f.png)
<img src=https://user-images.githubusercontent.com/62607343/130651465-9412e38f-a542-48f5-952a-37a3032c04ab.png width="500">

**2) Data Augmentation (Unused)**

DA augments data augmentation as a part of its pipeline, which rescales RGB values into 0.0~1.0 range, randomly rotates up to 50 degrees, and vertically and horizontally shifts by max 20%. 

**3) Use of HDF5 Format**

 Using HDF5 file format, it accelerates its learning speed

**4) Activation & Initialization**

ReLU overcomes gradient saturation problem that Sigmoid and Tanh functions contained. ReLU output does not stagnate for positive input values, while it does for negative input values. Leaky-ReLu is improved version of ReLU because it does not die for negative input values since it has slight angle for negative values. Leaky-ReLu function improved DA’s accuracy by 4.51% compared with ReLU. Furthermore, I implemented SELU for DA this week. Unlike ReLU and Leaky-ReLU, SELU does not lose gradient when input is 0, hence it completely overcomes dying ReLU problem. Replacement to SELU just by itself did not improve accuracy compared to Leaky-ReLU, however, the best performance, 88.46%, with the use of SELU, L1 regularization and decaying learning rate at 0.95 %. The following table is the summary of activations and initializers, relevant for my project.

<img src="https://user-images.githubusercontent.com/62607343/130621436-9b2ae0da-3ef5-4657-a67c-5fd116c9475d.png" width="2800x">


**5) Regularizer**

<u>L1 class (Lasso Regression)<u/>
   tf.keras.regularizers.l1(l1=0.01, **kwargs)
   The L1 regularization penalty is computed as: loss = l1 * reduce_sum(abs(x))
 
<u>L2 class (Ridge Regression)<u/>
   tf.keras.regularizers.l2(l2=0.01, **kwargs)
   The L2 regularization penalty is computed as: loss = l2 * reduce_sum(square(x))

<u>L1_L2 function<u/>
   tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
   The L1 regularization penalty is computed as: loss = l1 * reduce_sum(abs(x))
   The L2 regularization penalty is computed as: loss = l2 * reduce_sum(square(x))


**6) Drive Signal**

DA signals “Keep going” if there is no pothole and “Stop” if there is any pothole as the following screenshot:

<img src=https://user-images.githubusercontent.com/62607343/130651041-e3a2b913-c86d-4f97-a6aa-d5ff06b53117.png width="500">
<img src=https://user-images.githubusercontent.com/62607343/130651080-cd5ef267-4613-499a-b891-51d853d32246.png width="500">



