# Do captions in different languages produce different images?: Efficiently training Multilingual Diffusion 🇬🇧🇩🇪🇫🇷

In this project, we ask the question: Is it worth adding non-English support to monolingual Text-to-Image models or can we simply get away with translating the non-English prompts to English. We train a multilingual Diffusion model, based on Stable Diffusion v2.1, to support German and French, in addition to English. To do so, first, we construct two high quality training datasets for our proposed training method by filtering the [WIT dataset](https://github.com/google-research-datasets/wit) and the [Conceptual Captions 12M](https://ai.google.com/research/ConceptualCaptions/) dataset. Next, we perform two stages of training: 
* 1). Teacher Learning (To add multilingual capabilities to the CLIP ViT-H/14 text encoder model)
* 2). Concept Alignment (To align Stable Diffusion with the new text encoder by fine tuning the U-Net with LoRA rank 4)

Finally, we test our multilingual diffusion model (which we dub **RKS-diffusion**) and the standard Stable Diffusion v2.1 model on our high quality WIT test subset (see results below).
  
This project was completed as part of 10-623 under the guidance of Prof. Matt Gormley and Henry Chai at CMU. For more details refer to our [poster](https://github.com/Aadit3003/genai-multilingual-tti/blob/81dd1af650e2620a808335af1d819b7823cf94db/Gen_AI_Poster_Final.pdf)

## **Main Contributions**
* Our Fine-tuned Multilingual Diffusion Model: [RKS-Diffusion](https://huggingface.co/AaditD/rks-diffusion) (Supports English, French, and German)
* Our Fine-tuned Multilingual CLIP Text Model: [RKS-CLIP-Text-Encoder](AaditD/rks-clip-text-encoder)
* Our high quality subset of WIT :  [Multilingual-RKS-WIT](AaditD/multilingual_rks) (train/test split: 6k/1.5k image-caption pairs, with equal English, German, and French representation in captions).

_Note : RKS is a reference to Arceus_
## **Results**
FID and IS scores for the images generated using the two models, on three languages: English (EN), German (DE), and French (FR)

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky" rowspan="2"><span style="font-weight:bold">Model</span></th>
    <th class="tg-c3ow" colspan="3"><span style="font-weight:bold">FID(↓)</span></th>
    <th class="tg-c3ow" colspan="3"><span style="font-weight:bold">IS(↑)</span></th>
  </tr>
  <tr>
    <th class="tg-c3ow">EN</th>
    <th class="tg-c3ow">DE</th>
    <th class="tg-c3ow">FR</th>
    <th class="tg-c3ow">EN</th>
    <th class="tg-c3ow">DE</th>
    <th class="tg-c3ow">FR</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">Stable Diffusion v2.1 (Baseline)</td>
    <td class="tg-6ic8">1.08</td>
    <td class="tg-dvpl">1.16</td>
    <td class="tg-dvpl">1.3</td>
    <td class="tg-dvpl">11.33</td>
    <td class="tg-6ic8">11.45</td>
    <td class="tg-dvpl">11.17</td>
  </tr>
  <tr>
    <td class="tg-0pky">RKS-Diffusion (Ours)</td>
    <td class="tg-dvpl">0.99</td>
    <td class="tg-dvpl">1.04</td>
    <td class="tg-6ic8">0.95</td>
    <td class="tg-dvpl">11.73</td>
    <td class="tg-6ic8">12.42</td>
    <td class="tg-dvpl">11.63</td>
  </tr>
</tbody></table>

For the baselines, English achieves the best FID score, and surprisingly, German gets the best IS score (perhaps due to its similarity to English). **RKS-Diffusion outperforms the baseline on all metrics for French and German, while still not sacrificing English performance**. French gets the biggest improvement in FID score, and German gets the biggest improvement in IS score.

## Directory Structure
* ```data``` - The dataset CSVs and the scripts to recreate them
* ```scripts```
    * ```evaluation.py```: Evaluation (FID and IS) code for the generated images
    * ```lora_inference.py```: Inference code for the trained RKS-Diffusion model with the trained RKS-CLIP-Text-Encoder (using pipeline)
    * ```manual_inference.py```: Inference code for Stable Diffusion v2.1 (from scratch, i.e. manually performing the reverse denoising process) 
    * ```teacher_learning.py```: Training Stage-1 code for the RKS-CLIP-Text-Encoder
    * ```train_text_to_image_lora_rks.py```: Training Stage-2 code for fine-tuning Stable Diffusion with LoRA (adapted from this blog by [Huggingface](https://huggingface.co/blog/lora))
    *  lora.sh: The hyperparameters for the LoRA fine-tuning
    * ```visualize.py```: Code to compare and visualize the output of the same prompt in three languages with the baseline and our model

## Reproduce the Results

Recreate the environment (Importantly, you need to perform installation of 'diffusers' from the source):
```
conda env create --file requirements.txt -n genai
conda activate genai
conda env create –file requirements.txt
```

### Training
```cd scripts
python teacher_learning.py  # Teacher Learning
bash lora.sh  # Concept Alignment
```

### Evaluation
```
python manual_inference.py
python lora_inference.py
python eval.py --model "baseline" > baseline_results.txt
python eval.py --model "rks-diffusion" > rks_diffusion_results.txt
```


## Training Runs
You can find W&B dashboards of our training runs here:
* Training Stage 1: [Teacher Learning](https://wandb.ai/aadit/Gen-AI-Multilingual-TTI/runs/0p2fhqio/overview)
* Trainign Stage 2: [Concept Alignment](https://wandb.ai/aadit/text2image-fine-tune/runs/1nkio8i8?nw=nwuseraaditd)


