#### 3. Using Keras
The [Conversation AI](https://conversationai.github.io/) team, a research initiative founded by [Jigsaw](https://jigsaw.google.com/) and Google (both a part of Alphabet) are working on tools to help improve online conversation. One area of focus is the study of negative online behaviors, like toxic comments (i.e. comments that are rude, disrespectful or otherwise likely to make someone leave a discussion).   
  
Kaggle are currently hosting their [second competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#description) on this research. The challenge is to create a model that is capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. The competitions use a dataset of comments from Wikipedia’s talk page edits. Improvements to the current model will hopefully help online discussion become more productive and respectful.

We shall be using this dataset to benchmark a number of ML models. 
*Disclaimer: the dataset used contains text that may be considered profane, vulgar, or offensive.*

First set up a CPU based VM to run your models. We shall use sparse matrices which are better suited to CPU than GPU. 
I set the VM up like below, you may need to change the `datacenter` and `domain`.
```
ibmcloud sl vs create --datacenter=lon06 --hostname=hw04cpu --domain=darragh.com --os=UBUNTU_16_64 --flavor C1_8X8X100 --billing=hourly --san --disk=100 --disk=2000 --network 1000  --key=1418191
```
As before check the VM is created with `ibmcloud sl vs list`  
Login like `ssh -i /home/darragh/.ssh/id_rsa 158.176.93.70 -l root` or `ssh root@158.176.93.70`. You may need to wait a couple of minutes before logging in for the VM to br created. 

Once logged into the VM as `root` user, **Install docker**:
```
# Validate these at https://docs.docker.com/install/linux/docker-ce/ubuntu/
apt-get update
apt install apt-transport-https ca-certificates 
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic test" 
apt update 
apt install docker-ce
# Validated 05/11/19 - Darragh
# Test if docker hello world is working
docker run hello-world
```

Now we pull the image and start our jupyter notebook. 
```
docker run --rm -it -p 8888:8888 w251/tensorflow_hw04:latest
```

You will have an output of the location of the book running line below
```
[I 11:47:41.840 NotebookApp] The Jupyter Notebook is running at:
[I 11:47:41.841 NotebookApp] http://(5ebf32ea4e17 or 127.0.0.1):8888/?token=ffbb6d6b3a9b2e24fb8e0cc7eb8eb0657e1f58fa5595c5d4
```
Replace the domain name of the URL with your servers, public IP. For example for the above output I would go to URL. 
```
http://158.176.93.70:8888/?token=ffbb6d6b3a9b2e24fb8e0cc7eb8eb0657e1f58fa5595c5d4
```
Now open the notebook and run. And fill in the codes blocks marked for filling in and monitor your AUC. 
For the Logistic regression model you should be getting circa `0.88` AUC and `0.93` or more for the MLP. 
