---
layout: page
title: Building an image classifier using transfer learning
description: Transfer learning from pre-trained MobileNetV2
img: assets/img/tl.jpg
importance: 1
category: Deep Learning
related_publications:
---

---

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/transferlearning.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/transferlearning.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}

<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}