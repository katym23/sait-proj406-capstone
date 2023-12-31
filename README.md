# sait-proj406-capstone - Predicting Hospital Admission With A Patient-Centred Framework​

Katy Mombourquette
Denis Ouellette

## Introduction
A universal experience: visiting hospital emergency.
Getting there, maybe with great haste, just to wait (usually).. Why?​

#### Triage Procedure based on:​
- Severity​
- Available Resources​

#### Severity ​
- Is your emergency life-threatening? You may not know!​
- How urgent is your condition if not life-threatening? ​

#### Available Resources​
- Staff on-shift​
- Developed and proven methods of triage assessment​
- Technology 

What are the limitation to the resources we use? How can we improve?​

## Business Understanding

#### The Client
The [Yale New Haven​ Health System (YNHHS)​](https://www.ynhhs.org/)
- Who they are: A non-profit health system​
- What they include: Hospitals, a medical foundation, several multispecialty centres, and outpatient locations​
- What areas they cover: Westchester, New York to Westerly, Rhode Island​
- The Product:​ “Integrated, high value, **patient-centred care**”

#### The Scenario
Due to pressure to increase support for individuals facing a language barrier (i.e. minority demographics) on university campuses, YNHHS hired us to look at their existing triage data and improve triage for patients of that specific case.​

#### Deliverables and Goals
​- Cleaned and preprocessed historical triage data from YNHHS hospitals.​
- A machine learning model with an accuracy of 85% or higher​
- False Negatives (i.e. patients who were falsely discharged) no more than 5%. ​
- False Positives (i.e. patients who were falsely admitted) no more than 25-35% ​
- Data visualizations and interactive dashboards presenting patient predictions and trends.​


#### The Data
Our data came from the paper "Predicting hospital admission at emergency department triage using machine learning." by Hong WS, Haimovich AD, and Taylor RA. All credit goes to this original paper, found [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201016). The dataset is too big to upload directly to github, so please find it on the original paper github [here](https://github.com/yaleemmlc/admissionprediction)

Variables are split into the response variable (disposition), demographics, triage variables, hospital usage, chief complaints, past medical history, medications, imaging, and historical vitals and labs. We broke the data into dimension tables based on these categories. See the process [here](https://github.com/katym23/sait-proj460-capstone/blob/main/data_modelling.ipynb)

## Technology Used
- Python for data cleaning, analysis, and modelling (pandas, seaborn, and sklearn were the primary libraries used)
- PowerBI for data modelling, visualization, and dashboard creation
- PowerPoint for presentation.

## Cleaning

Please see [this code](https://github.com/katym23/sait-proj460-capstone/blob/main/data_cleaning.ipynb) for the cleaning process, including undersampling the language variable so that no bias was introduced from the dominance of the English-speaking class

## Model

Please see [this code](https://github.com/katym23/sait-proj460-capstone/blob/main/data_modelling.ipynb) for the modelling code.

## Additional Analysis

Please see [this code](https://github.com/katym23/sait-proj460-capstone/blob/main/association_counts.ipynb) for the association analysis and column counting we conducted for our chief complaint and past medical history columns.

## Predictive Model

Please see [this code](https://github.com/katym23/sait-proj460-capstone/blob/main/random%20forest%20classifier.ipynb) for the Random Forest Predictive Model.

## Functions

The functions used throughout all of the previous code is found [here](https://github.com/katym23/sait-proj460-capstone/blob/main/functions.py)


## Full Presentation
To see our entire presentation, including visualizations of the dataset, analyses and results, see our powerpoint [here](https://github.com/katym23/sait-proj406-capstone/blob/main/Capstone%20Final%20Presentation.pptx).

## Acknowledgements
We refer to it several times throughout all aspects of our project, but our credit goes to the original research paper and triage modelling project "Predicting hospital admission at emergency department triage using machine learning." Check out [the research paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201016) and the [GitHub repository](https://github.com/yaleemmlc/admissionprediction). We appreciate their code being open source and available for students like us to learn from.

​

​
