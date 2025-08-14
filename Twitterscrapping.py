import tweepy
from transformers import pipeline
import json

# Your Twitter API Bearer Token (v2)
BEARER_TOKEN = ''

def create_twitter_client():
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
    return client

def fetch_one_tweet(client, keyword):
    sentiment_analyzer = pipeline("sentiment-analysis")
    try:
        # Search recent tweets, max_results=10 (max allowed per request)
        response = client.search_recent_tweets(query=keyword + " lang:en -is:retweet", max_results=10, tweet_fields=['created_at','text','author_id'])
        tweets = response.data

        if not tweets:
            print("No tweets found.")
            return None

        tweet = tweets[0]
        text = tweet.text

        sentiment_result = sentiment_analyzer(text[:512])[0]
        sentiment = sentiment_result['label'].lower()

        if sentiment == 'positive':
            sentiment_label = 'positive'
        elif sentiment == 'negative':
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'

        tweet_info = {
            'tweet_id': tweet.id,
            'text': text,
            'sentiment': sentiment_label,
            'created_at': str(tweet.created_at),
            'author_id': tweet.author_id
        }
        return tweet_info

    except Exception as e:
        print(f"Error fetching tweet: {e}")
        return None

def main():
    client = create_twitter_client()
    keyword = input("Enter keyword or hashtag to search tweets: ")
    print(f"Fetching and analyzing one tweet for keyword: {keyword}")
    tweet_data = fetch_one_tweet(client, keyword)
    if tweet_data:
        print(json.dumps(tweet_data, indent=2))

if __name__ == "__main__":
    main()
