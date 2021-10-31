---
layout: archive
permalink: /Data-Science/
author_profile: true
# title: "Data Engineering Topics"
# header:
#     overlay_image: "/images/H1.jpg"
#     overlay_filter: 0.6
---
# Data Science Topics
{% capture written_label %}'None'{% endcapture %}

{% include group-by-array collection=site.Data-Science field="categories"%}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}