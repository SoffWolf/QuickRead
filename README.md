# QuickRead

## Overview
TLDR app: A web extension that can shrink a wall of text into a few sentences. 

The language model is initialized from Google's PEGASUS, and follows the training approach from the paper "Learning to summarize with human feedback" of OpenAI.


## Current state of development
We have finished fine-tuning the pretrained PEGASUS language model with the conventional supervised learning method on the Reddit TL;DR data set. You can try out this model from [here](https://huggingface.co/spaces/SophieTr/TextSummarizationDemo).
We are currently trying to increase the accuracy of the reward model from 0.8 to more than 0.95. 
The PPO algorithm is under development.

## File structure
The files are divided into the main stages of the development process:
1. Fine tune PEGASUS: folder fine_tune
2. Train reward model: folder rewards
3. Train policy with reinforcement learning: folder PPO_training

## Credit
Reddit TL;DR data set: [Reddit TL;DR](https://huggingface.co/datasets/reddit)

Human preference feedback data set: [Comparisons](https://github.com/openai/summarize-from-feedback#comparisons)


## Contact us
[Sophie Truong](https://www.linkedin.com/in/sophie-tr/).

[Tung Nguyen](https://www.linkedin.com/in/tung-nguyen-736154174/).