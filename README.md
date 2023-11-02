# Fraud-detection: Prevent the abnormal users from posting
## Dataset
The Reddit dataset is a Public graph dataset from Reddit posts made in the month of September, 2014. The node label in this case is the community, or “subreddit”, that a post belongs to. 50 large communities have been sampled to build a post-to-post graph, connecting posts if the same user comments on both. In total this dataset contains 232,965 posts with an average degree of 492. The first 20 days are used for training and the remaining days for testing (with 30% used for validation). For features, off-the-shelf 300-dimensional GloVe CommonCrawl word vectors are used.
## Downstream task
Train a Graph neural network with semi-supervised technique to predict whether the users are banned from posting
## Details about the project:
- Reimplement MLP-mixer - based Graph neural network as a backbone model for downstream task - node classification
- Reimplemnt the [Semi-supervised techniques](https://arxiv.org/abs/2305.13573) to train the model on the large scale of unlabeled dataset : 
- Build a Offline Batch inference processor for the large scale dataset by using Ray tool

## Run the Project
### Train model
Train the model on Reddit dataset

`
python train_node_classification.py --dataset_name reddit --model_name GraphMixer
`

### Run Batch Inference
Run offline batch inference

`
python batch_inference.py --dataset_name reddit --model_name GraphMixer
`
## References:
- https://arxiv.org/abs/2305.13573
- https://arxiv.org/pdf/2004.11362v5.pdf
- https://arxiv.org/abs/2302.11636
- https://docs.ray.io/en/latest/data/batch_inference.html
- https://github.com/twitter-research/tgn
- https://www.anyscale.com/blog/model-batch-inference-in-ray-actors-actorpool-and-datasets