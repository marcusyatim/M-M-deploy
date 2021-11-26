# Marcus Y Food

The source code and data in this repo is meant for deploying for production on Heroku platform with Docker.

For testing environment on local machine, refer to the accompanying repo https://github.com/marcusyatim/M-M. 

## Setup

A Heroku account and Docker account. Refer to https://devcenter.heroku.com/articles/container-registry-and-runtime#getting-started for more information of using Docker with Heroku.

## Files

This repo only contains files that are required to deploy the Marcus Y Food bot into production.

### Code

The code seen here are generally the same as those found in https://github.com/marcusyatim/M-M. However, they have been modified slightly in order to work with Docker and Heroku.

1. `app.py`
2. `config.py`
3. `getRatings_chatbot.py`
4. `getRecommendations_chatbot.py`
5. `getTags_chatbot.py`
6. `run_BERT.py`
7. `run_transformer.py`

### Manifest

The files here are manifest that tells Docker and Heroku how to deploy the program.

1. `Dockerfile`: Contains all the commands to call on the command line to assemble the Docker image.
2. `heroku.yml`: A manifest to define the Heroku app
3. `requirements.txt`: Dependencies and libraries to install.

### Data

These only contain the required data files to work in production. They were prebuilt from https://github.com/marcusyatim/M-M/tree/main/data and are available at said repo. (As such, data files are not included in this repo).

1. getRatings

2. getRecommendations
> - `/assignments/exp3.csv`
> - `/topN/exp3.json`

3. getTags