# 🤖 Blogbot violations model: deep dive 

This directory contains the scripts necessary to generate synthetic data, train and evaluate a smaller, binary BERT based classifier. The proposed solutions is as follows:

[PROPOSED SOLUTION IMAGE]

## :sparkles: Assumptions and constraints

To constrain the problem, I made the following assumptions:

- I am only dealing with text inputs;
- I am only dealing with the one known violation type: personally identifiable information (PII)
- I am only dealing with blogposts from one known source 
- I am generating a small amount of synthetic data, given modal free credit constraints 
- The blogposts I am using do NOT already contain PII. 

## 💅 Next steps

In order to improve/expand the solution, I would:

### Methods next steps

- **Introduce more diversity in injected PII:** either by better prompting or stuffing randomly chosen PII from a data source and reframing the task to include it. This is because the LLM generates similar PII i.e. `Jane Doe`, `John Doe`

- **Verify the blogposts do NOT contain PII:** via a process like disagreement modelling or by training a weak learner on a small subset of gold,standard non-PII blogposts and predicting whether data contains PII. I would then include high precision blogposts in my final training set.

- **Expand to additional violation types:** I am only addressing one violation type, predicting the presence of PII or not. Given the wider business goal of an overall service that allows licensees to report data that is in violation of laws/regulations, we would need to collect/generate more data on different violation types to train different models.  


### Infrastructure next steps

- **Build a modal inference endpoint:** As we are already storing the model weights in a modal volume, I would build out an inference endpoint that makes predictions for a given input text and returns important metadata like model id etc. 

- **Automated retraining/promoting:** Once the synthetic data generation pipeline is finalised, I would build out an automated retraining pipeline that schedules a i.e. bimonthly data generation and training loop, evaluates the challenger model and promotes the model if it is outperformant on a test set.

- **Add smoke tests:** I would add smoke tests to ensure a functioning `Llm` endpoint.  

## The pipeline

Assuming you would 

```
modal deploy blogbot/llm.py ## step 1 - deploy your Llm 

modal run --detach blogbot/synthetic_data_generation.py ## step 2 - generate structured outputs of PII-injected blogposts and append non PII data

modal run --detach blogbot/train.py ## step 3 - train and evaluate a binary violations model
```