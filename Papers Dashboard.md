# Papers Dashboard

## 全部论文

```dataview
TABLE
  choice(length(authors) > 0, join(authors, ", "), "") AS "作者",
  publish AS "发表时间",
  choice(venue, venue, "未知") AS "Venue",
  tags AS "标签",
  zotero AS "Zotero"
FROM "Papers"
WHERE type = "paper"
SORT created DESC
```

## Venue

```dataview
TABLE WITHOUT ID
  choice(venue, venue, "未知") AS "Venue",
  length(rows) AS "数量"
FROM "Papers"
WHERE type = "paper"
GROUP BY venue
SORT length(rows) DESC
```

## 标签

```dataview
TABLE WITHOUT ID
  tag AS "标签",
  length(rows) AS "数量"
FROM "Papers"
WHERE type = "paper" AND tags
FLATTEN tags AS tag
GROUP BY tag
SORT length(rows) DESC
```
