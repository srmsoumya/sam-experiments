# Experiments with SAM - Week 1

### Run the streamlit app segment.py
```
# pass in the model checkpoint and model type in `segment.py`
MODEL_CHECKPOINT=<sam-checkpoint-weights> 
MODEL_TYPE=<vit_b | vit_l | vit_h>

streamlit run segment.py
```


### Run the streamlit app annotate.py
```
# pass in the model checkpoint and model type in `segment.py`
MODEL_CHECKPOINT=<sam-checkpoint-weights> 
MODEL_TYPE=<vit_b | vit_l | vit_h>

streamlit run annotate.py
```