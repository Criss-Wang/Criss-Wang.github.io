---
layout: archive
permalink: /machine-learning/
title: "Machine Learning Projects"
arthor_profile: true
header:
    overlay_image: "/images/H1.jpg"
classes: wide
---

{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}