---
layout: page
title: Visual Grounding with Multimodal Attention and Grad-CAM in Image-Text Alignment
description: FLAVA's image-text alignment by observing attention maps and Grad-CAM outputs in visual grounding
img: assets/img/multimodal.jpg
importance: 3
category: Research
related_publications:
---


Visual grounding is a task in computer vision that involves identifying and localizing objects in an image based on a given natural language description. The goal is to "ground" or match a specific object or region in the image that corresponds to the description or referring expression.

For example, in the task of visual grounding, a model might be given an image of a park with several people and objects, along with a textual input like "the person wearing a red jacket." The model must then correctly identify and highlight the person in the red jacket from the image.

This task is particularly challenging because it requires understanding both the semantics of the language and the visual information in the image. The model must integrate these two sources of information to precisely locate the object mentioned in the text. Visual grounding is important for various applications, including:

**Human-robot interaction**: Robots can follow natural language instructions by locating objects in their surroundings.

**Autonomous driving**: Identifying objects based on spoken commands or navigation instructions.

**Image captioning and generation**: Improving the alignment between image content and descriptive text.

We will delve into how multimodal transformers like FLAVA can be employed to visualize the alignment between images and corresponding textual descriptions. We will do this using two key techniques:

1. **Attention Maps**: Capturing how the model focuses on different regions of the image for specific text tokens.

2. **Grad-CAM**: A powerful technique from CNNs that helps us visualize which parts of the image influence the model's predictions the most.

 We will process random images from the Flickr30k dataset, visualize attention maps, and overlay Grad-CAM heatmaps. This will allow us to better understand the interaction between the image and text in the context of visual grounding.

##  **Importing Required Libraries**

Before we begin, we will need to import all the necessary libraries. This includes:
- **Pytorch**: For handling tensors and backpropagation.
- **Transformers**: To use FLAVA, the multimodal model.
- **Matplotlib** and **PIL**: For image visualization and manipulation.
- **OpenCV**: For overlaying heatmaps on images using Grad-CAM.

Here’s why each library is important:
- **Transformers** provides the FLAVA model and processor, essential for multimodal tasks.
- **Torch** provides the core infrastructure to handle deep learning operations.
- **Matplotlib** and **PIL** help in rendering the image and heatmap outputs for better interpretability.
- **OpenCV** is a robust library for computer vision tasks and is used here to blend heatmaps and images effectively.

```python
import os
import csv
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import FlavaProcessor, FlavaModel
import matplotlib.pyplot as plt
import cv2
```

## **Downloading Flickr30k Dataset**
The Flickr30k dataset is a popular benchmark used in tasks related to image-captioning, visual grounding, and multimodal learning. It contains the following key features:

**Dataset Composition**: Flickr30k includes 31,783 images, each paired with five human-generated textual descriptions. These images are primarily sourced from Flickr and feature a wide range of real-world scenarios, including people, animals, and everyday objects in various settings.

**Multiple Annotations**: Each image in the dataset is paired with multiple captions, making it valuable for exploring the relationship between visual content and language. This richness allows models to learn diverse and contextually nuanced associations between objects and the words used to describe them.

**Diversity of Visual Content**: The dataset is diverse in terms of objects, people, and environments, providing a challenging benchmark for models aiming to perform well on complex image-text tasks, such as image captioning, visual question answering, and visual grounding.

```python
!wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00"
!wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01"
!wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02"
!cat flickr30k_part00 flickr30k_part01 flickr30k_part02 > flickr30k.zip
!rm flickr30k_part00 flickr30k_part01 flickr30k_part02
!unzip -q flickr30k.zip -d ./flickr30k
!rm flickr30k.zip
!echo "Downloaded Flickr30k dataset successfully."

```

```
 Downloaded Flickr30k dataset successfully.
```
## **Hook for Capturing Attention Weights**

we define a hook to extract attention weights from the FLAVA model's image encoder. Attention mechanisms are vital in multimodal models as they allow us to understand how the model is focusing on different parts of the input (in our case, the image) while processing the text.

This step is crucial for:

**Extracting Attention Weights**: Helps us later visualize where the model is looking when aligning text with the image.

**Registering Hooks**: A feature in PyTorch that lets us capture intermediate outputs during model inference.

```python
# Define global variable to store attention weights
attention_weights = []

# Hook function to capture and print the output from attention layers
def get_attention_weights_hook(module, input, output):
    global attention_weights
    attention_weights.append(output)
```    

## **Preprocessing Images and Captions**
Since FLAVA expects both images and text inputs, we need to prepare these inputs accordingly:

**Image Preprocessing**: We resize the images to match the input size expected by the model (224x224 pixels).

**Text Processing**: We use the FLAVA processor to tokenize the caption text and prepare it for model input.

```python
# Function to preprocess image and caption
def preprocess_image_caption(image_path, caption, processor):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Resize to match input size for FLAVA
    inputs = processor(text=[caption], images=image, return_tensors="pt", padding=True)
    return inputs, image
```    

## **Applying Grad-CAM to Understand Image Regions of Interest**

we implement Grad-CAM (Gradient-weighted Class Activation Mapping), which is commonly used to interpret CNN-based models. Grad-CAM helps us visualize which parts of the image are influencing the model’s decision the most.

**Why use Grad-CAM?**

Grad-CAM provides insights into the spatial regions that are most relevant for a specific task.
It complements the attention map by showing low-level features captured by the image encoder.
The heatmap generated by Grad-CAM will later be overlayed on the image to highlight important areas.

```python
# Function to apply Grad-CAM
def apply_gradcam(activations, grads):
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    activations = activations.squeeze(0).cpu().detach().numpy()
    pooled_grads = pooled_grads.cpu().detach().numpy()

    for i in range(pooled_grads.shape[0]):
        activations[i, :, :] *= pooled_grads[i]

    heatmap = np.mean(activations, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)

    return heatmap

```    

## **Overlaying the Grad-CAM Heatmap on Image**

We use **OpenCV** to overlay the heatmap and the image, allowing us to visually inspect which regions of the image are most relevant for the model’s decision.

**Key considerations**:

Overlaying the heatmap allows us to interpret which parts of the image correspond to the model’s attention.
This method provides a holistic view of the interaction between low-level image features and high-level semantics.

```python
# Function to overlay Grad-CAM heatmap on image
def overlay_heatmap_on_image(heatmap, image):
    heatmap_resized = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    image_array = np.array(image)
    superimposed_image = cv2.addWeighted(image_array, 0.6, heatmap_resized, 0.4, 0)

    return superimposed_image
``` 

## **Visualizing Attention Maps for Image and Text**

we use the attention weights captured in previous layers to generate attention maps. These maps help us visualize how the model attends to different parts of the image when processing specific text tokens.

**Why use Attention Maps?**

**Multimodal Alignment**: Shows how different regions of the image are aligned with specific text tokens.

**Interpretability**: Helps us understand how the model processes image-text pairs, a critical step in visual grounding tasks.

```python
# Function to visualize attention map for image and text
def visualize_attention_map(image, attention_weights, text_input, processor, layer_num=11, head_num=0, alpha=0.5):
    tokenizer = processor.tokenizer
    text_tokens = tokenizer.tokenize(text_input)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)

    attention_map = attention_weights[layer_num][0][head_num].detach().cpu().numpy()
    attention_map = np.mean(attention_map, axis=-1)

    if attention_map.shape[0] == 197:
        attention_map = attention_map[1:].reshape(14, 14)

    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    attention_map_resized = F.interpolate(torch.tensor(attention_map).unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear').squeeze().numpy()

    image_np = image_tensor.permute(1, 2, 0).numpy()

    plt.imshow(image_np)
    plt.imshow(attention_map_resized, cmap='jet', alpha=alpha)
    plt.title(f"Attention Map for: {text_input}")
    plt.axis('off')
    plt.show()
```
## **Get Activations and Gradients for Grad-CAM**

we define a helper function that captures the activations and gradients necessary for Grad-CAM. We attach hooks to the model’s image embeddings and compute the gradients with respect to the image features.

This step ensures that we can access the internal representations of the model for better interpretability.

```python
# Function to get activations and gradients
def get_activations_and_grads(model, inputs):
    image_input = inputs['pixel_values']

    activations = []
    grads = []

    def hook_activations(module, input, output):
        activations.append(output)

    def hook_grads(module, grad_input, grad_output):
        grads.append(grad_output[0])

    handle_activations = model.image_model.embeddings.patch_embeddings.projection.register_forward_hook(hook_activations)
    handle_grads = model.image_model.embeddings.patch_embeddings.projection.register_backward_hook(hook_grads)

    image_embeddings = model.get_image_features(pixel_values=image_input)

    loss = image_embeddings.norm()
    loss.backward()

    handle_activations.remove()
    handle_grads.remove()

    return activations[0], grads[0]
```
## **Process and Visualize Images**

```python
import random

# Function to process and visualize 10 random unique images
def process_random_images(captions_data, processor, model, num_images=10):
    # Randomly sample 10 unique images
    random_images = random.sample(captions_data, num_images)

    for i, image_info in enumerate(random_images):
        image_path = os.path.join("./flickr30k/Images", image_info['filename'])
        caption = image_info['captions'][0]

        # Preprocess image and caption
        inputs, original_image = preprocess_image_caption(image_path, caption, processor)

        # Get activations and gradients for Grad-CAM
        activations, grads = get_activations_and_grads(model, inputs)

        # Apply Grad-CAM and visualize
        heatmap = apply_gradcam(activations, grads)
        superimposed_image = overlay_heatmap_on_image(heatmap, original_image)

        # Plot original, heatmap, and superimposed image
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')
        axes[2].imshow(superimposed_image)
        axes[2].set_title("Grad-CAM Overlaid")
        axes[2].axis('off')

        plt.suptitle(f"Visual Grounding for Random Image {i+1}: {caption}", fontsize=14)
        plt.show()

        # Visualize attention map for the corresponding text and image
        visualize_attention_map(original_image, attention_weights, caption, processor)
```
## **Loading Captions and Initializing the FLAVA Model**

```python
# Load captions and images
caption_file = "./flickr30k/captions.txt"
captions_data = []

with open(caption_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) < 2:
            continue
        image_filename = row[0].strip()
        caption = row[1].strip()
        captions_data.append({'filename': image_filename, 'captions': [caption]})

# Initialize model and processor
processor = FlavaProcessor.from_pretrained("facebook/flava-full")
model = FlavaModel.from_pretrained("facebook/flava-full").eval()

# Register hook for capturing attention weights
for layer_num, layer in enumerate(model.image_model.encoder.layer):
    layer.attention.attention.register_forward_hook(get_attention_weights_hook)
```


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flavahuggingface.PNG" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---

## **Run Visualization on Random Images**

```python
# Process and visualize random images
process_random_images(captions_data, processor, model, num_images=10)
```
---
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/1g.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/1a.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2g.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2a.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3g.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3a.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4g.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4a.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/6g.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/6a.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/7g.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/7a.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/8g.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/8a.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/9g.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/9a.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
---
## **Conclusions**

By looking at the visual output of the Grad-CAM heatmap and the attention map applied to the image-2 of a band performing with flame-throwing effects, we can draw several technical insights:

**Grad-CAM Insights**:

**Heatmap Activation Focus**:

The Grad-CAM heatmap highlights the areas in the image where the model's attention is most concentrated. The color variations from blue to red indicate different levels of focus, with red being the highest activation.
The overlaid heatmap shows that the model is strongly focusing on the flame-throwing areas and the center stage where the band is performing. This suggests that the model recognizes the key elements in the image—the performers and the dramatic flames—as important visual cues.

The strong activation over the flame-throwers suggests that the model associates "*flames*" with the text "*flame throwing effects*" in the caption. The attention on the center, where the lead performer stands, suggests that the model also recognizes a human figure, which is in line with the "*band dressed in black*" part of the caption.
This implies that the model is correctly aligning the visual cues (flames and performers) with the descriptive elements in the text.

**Use of Grad-CAM**:

Grad-CAM works by using the gradients flowing into the final convolutional layer to understand which parts of the image most strongly impact the model's decisions. In this case, the flame effects and the central figure are given the most attention, showing the model’s sensitivity to both human presence and the flames, which are critical aspects of the caption.


**Attention Map Insights**:


**Text-Image Alignment**:

The attention map overlays show how different regions of the image are linked to specific tokens in the caption. Since this is a general attention map (for all tokens), it highlights regions that contribute most to understanding the caption as a whole.
Areas like the flames and the stage, where the band is standing, receive more attention (yellow/red regions), while the background and parts without much relevance to the caption remain in cooler colors (blue/green).


**Semantic Consistency**:

The attention map suggests a high degree of semantic consistency. The regions of the image that the model attends to (flames, performers) match with the words "band," "black," and "flame throwing effects" from the caption. This shows the model's ability to correctly attend to and map visual features with descriptive textual tokens.
Broader Focus:

**Compared to Grad-CAM**, which highlights the most critical regions, the attention map provides a broader focus, giving weight to multiple parts of the image. This is typical in multimodal models, where multiple regions contribute to the interpretation of different parts of the caption. For instance, the flames and stage structure are attended to because they align with both “flame throwing effects” and “band.”

**Model’s Visual Grounding Ability:**

The model demonstrates effective visual grounding, where it successfully identifies and focuses on the key elements in the image—flames and band members—that are explicitly mentioned in the caption.
The Grad-CAM heatmap confirms that the model is highly sensitive to these regions during the classification process, while the attention map shows how well the model distributes its focus across the relevant visual features.


**Balanced Multimodal Attention:**

The overlay of the attention map suggests that the model's attention is distributed well across the important visual features without neglecting any key areas. Both the flames and the performers receive adequate attention, reflecting the importance of those features in understanding the caption.


**Further Improvements:**

While the model performs well in this case, it may sometimes overlook finer details not explicitly mentioned in the caption (e.g., audience in the background), as seen by the cooler areas in the attention map. Further fine-tuning or adjusting the model to focus on a wider range of features could improve its performance on more complex images and captions.

**Practical Applications:**

This visualization technique can be especially useful in domains like content-based image retrieval, video captioning, and human-computer interaction. By ensuring that a model's attention aligns well with relevant content, one can improve the accuracy and relevance of AI systems interpreting visual data.

## **References**
- FLAVA Paper: [FLAVA: A Foundational Language and Vision Alignment Model](https://arxiv.org/abs/2112.04482)
