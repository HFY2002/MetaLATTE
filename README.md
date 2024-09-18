---
license: cc-by-nc-nd-4.0
tags:
- climate
- biology
---

[preprint](https://www.biorxiv.org/content/10.1101/2024.06.26.600843v1)

# MetaLATTE: Metal Binding Prediction via Multi-Task Learning on Protein Language Model Latents

The bioremediation of environments contaminated with heavy metals is an important challenge in environmental biotechnology, which may benefit from the identification of proteins that bind and neutralize these metals. Here, we introduce a novel predictive algorithm that conducts **Metal** binding prediction via **LA**nguage model la**T**en**T** **E**mbeddings using a multi-task learning approach to accurately classify the metal-binding properties of input protein sequences. Our **MetaLATTE** model utilizes the state-of-the-art ESM-2 protein language model (pLM) embeddings and a position-sensitive attention mechanism to predict the likelihood of binding to specific metals, such as zinc, lead, and mercury. Importantly, our approach addresses the challenges posed by proteins from understudied organisms, which are often absent in traditional metal-binding databases, without the requirement of an input structure. By providing a probability distribution over potential binding metals, our classifier elucidates specific interactions of proteins with diverse metal ions. We envision that MetaLATTE will serve as a powerful tool for rapidly screening and identifying new metal-binding proteins, from metagenomic discovery or _de novo_ design efforts, which can later be employed in targeted bioremediation campaigns. 

![workflow](figures/Figure1.png)



## Interactive Demo

You can try out the MetaLATTE model directly in your browser:

<https://huggingface.co/spaces/ChatterjeeLab/MetaLATTE-demo>


## Usage

```python
import sys
from transformers import AutoTokenizer, AutoModel, AutoConfig
metalatte_path = './Chatterjeelab/MetaLATTE'
sys.path.insert(0, metalatte_path)
from metalatte import MetaLATTEConfig, MultitaskProteinModel
AutoConfig.register("metalatte", MetaLATTEConfig)
AutoModel.register(MetaLATTEConfig, MultitaskProteinModel)


tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
config = AutoConfig.from_pretrained("ChatterjeeLab/MetaLATTE")
model = AutoModel.from_pretrained("ChatterjeeLab/MetaLATTE", config=config)

model.eval()
sequence = "AVYNIGWSFNVNGARGKSFRAGDVLVFKYIKGQHNVVAVNGRGYASCSAPRGARTYSSGQDRIKLTRGQNYFICSFPGHCGGGMKIAINAK"
inputs = tokenizer(sequence, return_tensors="pt")
raw_probs, predictions = model.predict(**inputs)

id2label = config.id2label
predicted_labels = [id2label[i] for i, pred in enumerate(predictions[0]) if pred == 1]
print(predicted_labels)
['Cu']

```

# Repo Author
- Yinuo Zhang (yzhang@u.duke.nus.edu)
- Pranam Chatterjee (pranam.chatterjee@duke.edu)