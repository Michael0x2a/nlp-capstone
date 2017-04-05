## Team ???

### Blog posts:

<ul>
    {% for post in site.posts %}
        <li>
            <a href="{{ post.url | absolute_url }}">
                {{ post.date | date: "%B %e, %Y" }} - {{ post.title }}
            </a>
        </li>
    {% endfor %}
</ul>
