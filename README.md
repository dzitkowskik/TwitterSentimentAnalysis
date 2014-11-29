Introduction-to-Data-Mining-DTU
===============================

This project is a result of work on Twitter Sentiment Diffusion 
in the DTU course "Introduction to Data Mining using Python".

It contains TwitterSentimentAnalysis package as an API and a web
site written in Django as a visual interface.

# REQUIREMENTS

* MongoDB - [Web page](http://www.mongodb.org)
* Python 2.7 - [Web page](https://www.python.org/download/releases/2.7/)
* Numpy version 1.8.2 or higher
* Scipy version 0.11.0 or higher
* Pip (for downloading packages - one can use also easy_install or sth else)
    
Scipy, Numpy and can be installed using following instruction: 
    
    $ [sudo] apt-get install python-numpy python-scipy python-pip

# INSTALLATION

All requirements must be satisfied before installation.

1. Install virtualenv (If you don't have it installed):

    ```bash 
    $ [sudo] pip install virtualenv
    ```
2. Clone repo and cd to directory:

    ```bash 
    $ git clone https://github.com/dzitkowskik/Introduction-to-Data-Mining-DTU.git <directory>
    $ cd <directory>
    ```
3. Create virtual environment and activate it:

    ```bash
    $ virtualenv env --system-site-packages
    $ source env/bin/activate
    ```
4. Install required packages:

    ```bash
    $ pip install -r requirements.txt
    ```
5. Prepare configuration (This script will create configuration files):
    1. run configure.sh:
    
        ```bash
        $ [sudo] chmod +x <directory>/configure.sh
        $ [sudo] ./<directory>/configure.sh
        ```  
    2. open file <directory>/TwitterSentimentAnalysis/configuration.cfg
    3. modify file inserting your twitter consumer keys and access tokens for twitter connection
        - Downloading predefined tweet dataset can take up to several hours, to skip most of the dataset:
            - set value pred_tweet_limit to a small number, for example 100 (only 100 tweets will be downloaded)
    4. save the file

6. Install an API - (it will also download predefined tweet set with manual sentiment grades):

    ```bash
    $ [sudo] python setup.py install
    ```

# RUNNING SERVER

1. Run mongo server and go to django web site package:

    ```bash
    $ mongod &
    $ cd <directory>/site/tsa
    ```
2. Make migrations, migrate tables to db and start server:

    ```bash
    $ python manage.py makemigrations
    $ python manage.py migrate
    $ python manage.py syncdb
    $ python manage.py runserver
    ```

# LICENCE

The MIT License (MIT)

Copyright (c) 2014 Karol Dzitkowski & Matthias Baetens

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

    
