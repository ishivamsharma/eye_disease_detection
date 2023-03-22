# Eye_Disease_Detection

Early detection and treatment of visual impairment diseases are critical and integral to combating avoidable blindness. To enable this, artificial intelligence–based disease identification approaches are vital for visual impairment diseases, especially for people living in areas with a few ophthalmologists. 
We have tried and developed a hierarchical deep learning network, which consists of a family of multi-task & multi-label learning classifiers representing different types/levels of eye diseases derived from a predefined hierarchical eye disease taxonomy. It can cater to various diseases such as Cataract, Glucoma, Retinopathy and corneal disease and deployed our web application eye-disease screening model for day-to-day use. 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

## About the Disease

Diabetic retinopathy: The persistently high blood sugar levels that occur with diabetes can damage the retina’s small blood vessels (capillaries), which deliver oxygen and nutrients. Diabetic retinopathy affects up to a third of people with diabetes over the age of 502.

Cataracts: A cataract is a clouding of the lens in the eye. Left untreated, cataracts can eventually lead to blindness. People with diabetes are more likely to develop cataracts at an earlier age and suffer visual impairment faster than those without the condition.1,3

Glaucoma: This is a group of conditions that can damage the optic nerve. The optic nerve transmits signals from the retina to the brain for processing. Glaucoma is often (but not always) a result of increased pressure inside the eye. The risk of glaucoma in people with diabetes is significantly higher than that of the general population.1,4 The two main types are open-angle glaucoma (also called ‘the sneak thief of sight’) and angle-closure glaucoma (this comes on suddenly and is a medical emergency).

![Screenshot (36)](https://user-images.githubusercontent.com/69316273/226872946-3c712ffa-2fb6-4a4d-a239-681356e8c4e5.png)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

## About the dataset

The dataset below consists of Normal, Diabetic Retinopathy, Cataract and Glaucoma retinal images where each class have approximately 1000 images. These images are collected from various sorces like IDRiD, Oculur recognition, HRF etc.
https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification


Use the dataset below for your eye A-I predictive algorithms for predicting eye diseases.
This data contains five types of diseases which is Bulging eyes, Cataracts, Crossed eyes, Glaucoma and Uveitis
https://www.kaggle.com/datasets/kondwani/eye-disease-dataset



Ocular Disease Intelligent Recognition (ODIR) is a structured ophthalmic database of 5,000 patients with age, color fundus photographs from left and right eyes and doctors' diagnostic keywords from doctors.
The dataset below is meant to represent "real-life’" set of patient information collected by Shanggong Medical Technology Co., Ltd. from different hospitals/medical centers in China. In these institutions, fundus images are captured by various cameras in the market, such as Canon, Zeiss and Kowa, resulting into varied image resolutions.
https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Pre-trained CNN Models


### InceptionV3

A convolutional neural network model, Inception V3, [51] is an extensively used image recognition and object detection model that has shown its credibility by achieving an accuracy of greater than 78.1% on the ImageNet dataset. This was primarily launched during the ImageNet Recognition Challenge. Multiple researchers are responsible for the production and synthesis of the Inception V3, over many years. This CNN model is 27 layers deep, consisting of an inception layer that is a coalescence of the 1*1 convolutional layer, 3*3 convolutional layer, and 5*5 convolutional layers. Each of their output filter banks merges into a distinct output vector that acts as the input of the next phase. Another 1*1 convolutional layer is added to reduce dimensionality, along with the max pooling layer that is left as a second option for the inception layer. This model works in a hierarchical order where the intrinsic details are considered the first stage, leading to the overall outline of the subject. For this, the layers demand precise filter sizes to correctly detect objects. The Inception layer, therefore, facilitates the internal layers to adopt the filter size that is ideal for their respective functions.



### ResNet-50

ResNet-50, a convolutional neural network has 50 layers. It comprises 48 convolutional layers with 64 different kernels, [47] 1 max pool layer with a stride of size 2. These layers are replicated 3 times to give a total of 9 layers. The next layer has different kernels and is repeated 4 times to give a total of 12 layers. Following layers consist of other variants of kernels which are repeated many times to form a total of 49 layers. Consequently, an average pool is done with a thoroughly networked layer consisting of 1000 nodes and a SoftMax function, giving us the last layer of this architecture. A pre-trained version of the network, trained with the images from the ImageNet database, can be loaded in this model. Thus, giving the network an enriched knowledge of feature representation for a large assortment of images.


### MobileNet_V2

In order to reduce the size of the model and the complexity of the network, Mobile Nets are a depth-wise separable convolution design that reduces the number of connections. Embedded and mobile applications benefit from the technology. The author has included two global hyperparameters into this sort of network, which are as follows: A good balance between model latency and accuracy is achieved with this technique. In addition, the hyperparameters give the capability of selecting a suitably scaled model in accordance with the problem restrictions, if necessary.


### VGG19

VGG19 is a convolutional neural network architecture for image classification that was proposed by researchers at the University of Oxford. It consists of 19 layers, including 16 convolutional layers and 3 fully connected layers. The architecture is known for its simplicity and achieved state-of-the-art performance on the ImageNet dataset. VGG19 uses small 3x3 filters throughout the network, which allows for a more detailed representation of the image features. It also uses max pooling layers to reduce the size of the feature maps and increase computational efficiency. Overall, VGG19 is a powerful and widely-used deep learning model for image classification task


###  EfficientNetB3:

* EfficientNetB3 is one of the models in the EfficientNet family of convolutional neural networks. It was introduced by Tan et al. in their paper "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (2019) and is designed to achieve high accuracy while maintaining computational efficiency.
* EfficientNetB3 has 25 million parameters and is based on a compound scaling method that uniformly scales the depth, width, and resolution of the network. Specifically, EfficientNetB3 scales up the base architecture of EfficientNetB0 by increasing the depth, width, and resolution of the network.
* The architecture of EfficientNetB3 consists of 30 layers, including a stem, multiple blocks, and a top classifier. The stem includes a sequence of convolutional, pooling, and activation layers that extract features from the input image. The blocks are composed of a sequence of depth wise and pointwise convolutions that further extract features and reduce the spatial dimension of the input. The top classifier is a fully connected layer that maps the extracted features to the output classes.
* EfficientNetB3 achieves state-of-the-art performance on a range of computer vision tasks, including image classification, object detection, and segmentation. It has been pre-trained on large datasets such as ImageNet and can be fine-tuned on specific tasks with smaller datasets. EfficientNetB3 is a larger and more complex variant of EfficientNetB0, with more layers and more parameters. It was designed to provide better accuracy and performance on large-scale image classification tasks compared to EfficientNetB0.


### We have used various pretrained models and also made CNN-Sequential(from Scratch) to get the best accuracy for our model.

![Screenshot (57)](https://user-images.githubusercontent.com/69316273/226896688-91698cd8-0f4e-4b5f-b41d-123da2eb48c0.png)


--------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Here are the general steps to train and test a model in deep learning:

1. Prepare the data: Load and preprocess the data as required by the specific problem you are trying to solve. This may involve tasks such as resizing images, normalization, or data augmentation.
2. Split the data: Split the data into training, validation, and testing sets. The training set is used to train the model, the validation set is used to tune the hyperparameters and prevent overfitting, and the testing set is used to evaluate the final performance of the model.
3. Define the model architecture: Define the layers and architecture of the deep learning model, including input and output layers, hidden layers, and activation functions.
4. Compile the model: Choose an appropriate loss function, optimizer, and evaluation metric, and compile the model.
5. Train the model: Train the model on the training data using the fit function, specifying the number of epochs and batch size.
6. Evaluate the model: Evaluate the performance of the model on the validation data using the evaluate function. Use the results to adjust the model architecture and hyperparameters as needed.
7. Test the model: Evaluate the final performance of the model on the testing data using the evaluate function.
8. Save the model: Save the trained model for later use or deployment.
9. Deploy the model: Deploy the trained model in a real-world scenario or integrate it into an application as needed.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Deployment

We have deployed our model through Streamlit. It is an open-source Python library that allows data scientists and developers to quickly and easily create interactive web applications for their machine learning or data science projects. With Streamlit, users can create interactive dashboards, data visualizations, and other applications without having to write extensive HTML, CSS, or JavaScript code.

Streamlit offers a simple and intuitive API that allows users to create interactive widgets and easily display data, charts, and other visualizations. It also provides features for sharing and collaborating on projects, making it a popular tool for teams working on data science projects.

Some key features of Streamlit include:

* Easy-to-use API: Streamlit offers a simple and intuitive API that allows users to create interactive widgets and display data and visualizations with just a few lines of code.
* Automatic layout: Streamlit automatically lays out elements on the page, making it easy to create polished and professional-looking applications.
* Collaboration features: Streamlit provides tools for sharing and collaborating on projects, making it easy for teams to work together on data science projects.
* Support for multiple data formats: Streamlit supports a wide range of data formats, including CSV, JSON, and Pandas DataFrames.
* Streamlit is a powerful tool for creating data science web applications quickly and easily. Its simplicity and ease of use make it a popular choice for data scientists and developers alike.

### Here is the interface of how the streamlit looks like:

![Screenshot (31)](https://user-images.githubusercontent.com/69316273/226897901-692bc923-ff71-4a8b-8222-2a752968973c.png)


### Here are some predictions:

![Screenshot (32)](https://user-images.githubusercontent.com/69316273/226898045-247f7bb4-3f5e-498f-91d4-5c2ef18f5257.png)

![Screenshot (34)](https://user-images.githubusercontent.com/69316273/226898077-501fe54a-c805-4990-9f85-d82054507fa8.png)


--------------------------------------------------------------------------------------------------------------------------------------------------------------------

## RESULT 
  
Our approach of comparing the transfer learning models and the CNN-Sequential proved to be effective in finding the optimal model that works best for the disease of diabetic retinopathy. Since the disease has been prevalent for a long period and is one of the major causes of blindness, early detection of such a disease can save a diabetic person from losing their vision. On the other hand, comparing the model using different metrics also helped us evaluate a model more precisely on different scales. In the first approach, we have used CNN-Sequential deep neural model with ‘relu’ activation function in convolutional layer and ‘softmax’ activation function in dense layer ,we got an accuracy of  90% , but we got lot of fluctuations in the loss and accuracy graph. As the graph was not consistent we used EfficientNetB3 pretrained model with “adamax” activation function. Through EfficientNetB3 we got accuracy of 93% with minimum fluctuations in loss graph. 


For the approaches, we have used 5 pre-trained and 1 CNN-Sequential(from scratch) to gather more intuition about the performance of the models. The compilation of the model is done with several metrics like accuracy, precision, and recall.  The number of epochs used to train the model varied for the  approaches we used in VGG19,ReseNet50,Inception_V3,MobilNet-V2,CNN-Sequential(from Scratch) and EfficientNetB3.  
