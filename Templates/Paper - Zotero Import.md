---
type: paper
authors: [{% for creator in creators %}"{{creator.firstName | default('') | escape}} {{creator.lastName | default('') | escape}}"{% if not loop.last %}, {% endif %}{% endfor %}]
publish: "{% if date %}{% if date | length == 4 %}{{date}}{% else %}{{date | format('YYYY-MM-DD')}}{% endif %}{% endif %}"
venue: "{% if publicationTitle %}{{publicationTitle | escape}}{% elif conferenceName %}{{conferenceName | escape}}{% elif proceedingsTitle %}{{proceedingsTitle | escape}}{% elif meetingName %}{{meetingName | escape}}{% elif repository %}{{repository | escape}}{% elif archive %}{{archive | escape}}{% elif websiteTitle %}{{websiteTitle | escape}}{% elif bookTitle %}{{bookTitle | escape}}{% elif seriesTitle %}{{seriesTitle | escape}}{% elif institution %}{{institution | escape}}{% elif university %}{{university | escape}}{% elif publisher %}{{publisher | escape}}{% elif libraryCatalog %}{{libraryCatalog | escape}}{% elif url and 'arxiv.org' in url %}arXiv{% elif DOI and 'arXiv' in DOI %}arXiv{% elif doi and 'arXiv' in doi %}arXiv{% elif url and 'biorxiv.org' in url %}bioRxiv{% elif url and 'medrxiv.org' in url %}medRxiv{% elif url and 'ssrn.com' in url %}SSRN{% endif %}"
url: "{{url | default('') | escape}}"
zotero: "{% if attachments and attachments.length %}{{attachments[0].pdfURI | default(attachments[0].desktopURI) | default(attachments[0].uri) | default('')}}{% endif %}"
created: "{{importDate | format('YYYY-MM-DD HH:mm')}}"
abstract: "{{abstractNote | default(abstract) | default('') | escape}}"
tags: []
---
