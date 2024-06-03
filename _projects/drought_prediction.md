---
layout: page
title: Drought prediction using satellite images
description: classifying satellite images based on forage count, likelyhood of drought is inversely correlated to the forage count.
img: assets/img/satellite.jpg
importance: 2
category: Deep Learning
related_publications:
---

---

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/drought.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/drought.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}

<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}