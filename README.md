# COSC 227 : Geolocation

In this project you will help the fictitious social network company MyFace+. Users on MyFace+ connect with one another and publish posts online. Despite the infrastructure, development, and maintenance costs necessary to make MyFace+ run MyFace+ is completely free to all users. In order to keep the company afloat, executives at MyFace+ have decided to sell user data. Much of the user data that companies/governments are interested in purchasing (education level, political alignment, purchasing habits, etc.) is fairly straight forward to gather — MyFace+ can simply ask its users. Other data, like geographic location, can be gathered directly from the application without inconveniencing the user with annoying questions. Richer data, like what the user watches on TV and what ads the user mutes or listens to is something MyFace+ is very interested in. They have not yet entered this arena for fear of patent infringement (US20180167677), though.

By knowing a user’s location, MyFace+ can ping nearby friends, suggest local deals, and alert authorities if the user is in danger. Some users, however, do not understand the value that MyFace+ provides by telling their Partners* where the user is currently residing. These users use 3rd party software to hide their location when they post. Focus groups have indicated that users cite “fear of political persecution”, “isn’t privacy still a thing?”, and “I’m a reporter” as leading reasons for this choice. 

Some users use their phone’s setting to turn off GPS, or use a GPS-spoofing app. This is not a problem for MyFace+. Most users are connected to a wifi network and as long as some users on the same network have GPS enabled, MyFace+ knows the location of the network. Similarly, if a user is connected to a cell tower, MyFace+ can look up the location of the cell tower. Some users, however, utilize distributed encryption-based anonymity networks (think: Tor).

MyFace+ is interested in improving the MyFace+ experience for such users. Beta-testing attempts based on asking the user to manually report their location resulted in unreliable data collection for unknown reasons. As such, MyFace+ seeks to infer user geographic location based on posting behavior and social connections. MyFace+’s legal team points out that a possible (although not verifiably intentional) side effect of this plan is an increase in MyFace+’s revenue from selling user data. 

You will be creating a system that predicts a user’s most common location.

*”Partners” includes but is not limited to anyone willing to spend 0.005 USD per user profile — bulk discounts may apply.

## How to Submit your Predictions

This project is run as a Kaggle competition.
The link to the competition is here:  https://cosc247f18.page.link/P4_Competition

You’ll submit a file in the format of `submission-example.txt`. This file has a header (the first line, which is `Id,Lat,Lon`). After that, each line should contain three values: the userid you’re predicting, the predicted latitude, the predicted longitude.


## Performance Measure

Your score, given a set of predictions, will be the Root Mean Squared Error (RMSE) of your predictions:


$\begin{aligned}
&\sqrt{\frac{1}{2 n} \sum_{i=1}^{n}\left(\operatorname{lat}^{(i)}-\operatorname{lat}^{(i)}\right)^{2}+\left(\operatorname{lon}^{(i)}-\operatorname{lon}^{(i)}\right)^{2}}\\
&=\frac{1}{\sqrt{2 n}} \sqrt{\sum_{i=1}^{n}\left(\operatorname{lat}^{(i)}-\operatorname{lat}^{(i)}\right)^{2}+\left(\operatorname{lon}^{(i)}-\operatorname{lon}^{(i)}\right)^{2}}
\end{aligned}$


## How To Run:

Simply make sure you are in the right directory. Then, type the following line into the command line:

```bash    
python3 predict_geolocation.py
```
The program will generate the predictions, saving a preds.txt file with the predictions and a preds.png with the plot of those predictions in the directory.

## Code description

The methods get_train_post_data and get_test_post_data both read from the files posts_train.txt and posts_test.txt, storing the features into a dictionary where they key is each users IDs.
    The method create_dict is used by these previous two methods.
    The method get_user_dict calls get_train_post_data and get_test_post_data to generate a user dictionary. Then, it goes through the graph.txt file, extracting the relevant features for each user and storing them in the user dictionary. It then returns the user dictionary with all the features.
    The method get_data takes in that user dictionary, and converts them into design matrices and target values, as well as a list of test users ids. Importantly, it takes in a boolean lat, specifying whether to get feautres for latitude or longitude (which were different).
    The method predict_test takes in two learners, one for latitude and one for longitude. It uses the previous methods to generate the training and test features for latitude and longitude. Then, it trains and predicts with the latitude learner, then the longitude learner. Finally, it writes these predictions to a file in the directory, and generates an image with the plotted predictions, saving it to the directory.
    Finally, we create to regressor objects: knn, a KNeighborsRegressor, and neural, an MLPRegressor. We then call predict_test with these two learners to get our predictions.