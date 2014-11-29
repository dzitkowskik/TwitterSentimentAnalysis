Introduction-to-Data-Mining-DTU
===============================

All the scripts from DTU course "Introduction to Data Mining using Python"

# REQUIREMENTS

* MongoDB - [Web page](http://www.mongodb.org)
* Python 2.7 - [Web page](https://www.python.org/download/releases/2.7/)

# INSTALLATION

1. Install virtualenv (If you don't have it installed):
```bash
$ [sudo] pip install virtualenv
```
2. Clone repo and cd to directory:
```bash
$ git clone https://github.com/dzitkowskik/Introduction-to-Data-Mining-DTU.git <directory>
$ cd <directory>
```
3. Create virtual environment and activate it
```bash
$ virtualenv <directory> --no-site-packages
$ ./<directory>/scripts/activate
```
4. Install required packages
```bash
$ pip install -r <directory>/requirements.txt
```
5. Prepare configuration (This script will create configuration files)
```bash
$ [sudo] chmod +x <directory>/configure.sh
$ [sudo] ./<directory>/configure.sh
```  
    1. open file <directory>/TwitterSentimentAnalysis/configuration.cfg
    2. modify file inserting your twitter consumer keys and access tokens for twitter connection
        - Downloading predefined tweet dataset can take up to several hours, to skip downloading most of the dataset:
            - set value pred_tweet_limit to a small number, for example 100 (only 100 tweets will be downloaded)
    3. save the file

6. Install an API - (it will also download predefined tweet set with manual sentiment grades)
```bash
$ [sudo] python setup.py install
```

# RUNNING SERVER
1. Run mongo server and go to django web site package
```bash
$ mongod &
$ cd <directory>/site/tsa
```
2. Make migrations, migrate tables to db and start server
```bash
$ python manage.py makemigrations
$ python manage.py migrate
$ python manage.py syncdb <one can set a password to django admin there>
$ python manage.py runserver
```

    
