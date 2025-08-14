from newspaper import Article

url = "https:\/\/news.google.com\/rss\/articles\/CBMilwFBVV95cUxQaHZjdTR5V1B2WFhnYzZ3VVhnOVlmdVpha2c2TElmdHpOQ2ZvdEIzMnU5SVJDNFF1ZWN3SnluQUttZ3FZZG94Z1pZdGNBRzYtQ3NEQjQ1YWNnM3dvaUlTTW5Ya09tRVA2NW9ZS1c2SXZpMlZFVkZfOHZWUXpfbFdTUlM4VDJqUUR6NERkbzRDYWNSVTNwdHJR?oc=5"
article = Article(url)
article.download()
article.parse()

print(article.title)
print(article.text)
