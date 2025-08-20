import argparse
import concurrent.futures
import feedparser
import json
import re
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import requests
from bs4 import BeautifulSoup
from newspaper import Article
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import trafilatura  
from json import JSONEncoder
from transformers import T5Tokenizer, T5ForConditionalGeneration

import torch

# Ensure NLTK data is available
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)


@dataclass
class NewsItem:
    title: str
    link: str
    published: Optional[str]
    source: Optional[str]
    content: Optional[str] = None
    article_text: str = ""
    article_title: Optional[str] = "Google News"
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    vector: Optional[np.ndarray] = None
    vector_id: Optional[int] = None
    chunk_id: Optional[int] = None
    is_chunk: bool = False
    original_item_id: Optional[int] = None


class NumpyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def is_news_or_blog(url: str) -> bool:
    """Filter out non-news URLs based on domain keywords - borrowed from Code 2"""
    if not url:
        return False
    
    disallowed_keywords = [
        'insurance', 'lawyer', 'legal', 'attorney', 'claim',
        'rental', 'shop', 'store', 'compare', 'quote', 'loan',
        'casino', 'betting', 'gambling', 'pharmacy', 'pills',
        'dating', 'adult', 'porn', 'sex', 'escort'
    ]
    
    try:
        domain = urlparse(url).netloc.lower()
        for keyword in disallowed_keywords:
            if keyword in domain:
                return False
        return True
    except Exception:
        return False


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split long text into overlapping chunks for better vector search"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings in the last 100 characters
            last_period = text.rfind('.', max(start, end - 100), end)
            last_exclamation = text.rfind('!', max(start, end - 100), end)
            last_question = text.rfind('?', max(start, end - 100), end)
            
            sentence_end = max(last_period, last_exclamation, last_question)
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + 1, end - overlap)
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks


class VectorDB:
    def __init__(self, dimension: int = None):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get dimension from model if not specified
        if dimension is None:
            sample_text = "Sample text for dimension calculation"
            sample_vector = self.model.encode([sample_text])[0]
            dimension = len(sample_vector)
        
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.stored_items: Dict[int, NewsItem] = {}
        self.current_id = 0
    
    def encode_text(self, text: str) -> np.ndarray:
        try:
            vector = self.model.encode([text])[0]
            return vector.astype(np.float32)
        except Exception as e:
            print(f"Error encoding text: {e}")
            return np.zeros(self.dimension, dtype=np.float32)
    
    def add_item(self, item: NewsItem, enable_chunking: bool = True) -> List[NewsItem]:
        """Add item with optional chunking for long articles"""
        added_items = []
        
        try:
            # Main item text for encoding
            main_text = f"{item.title} {item.article_text}"
            
            # Check if we should chunk long articles
            if enable_chunking and len(item.article_text) > 1500:
                chunks = chunk_text(item.article_text)
                print(f"Chunking article '{item.title[:50]}...' into {len(chunks)} chunks")
                
                # Store original item
                original_id = self.current_id
                vector = self.encode_text(main_text)
                item.vector = vector
                item.vector_id = original_id
                
                self.index.add(vector.reshape(1, -1))
                self.stored_items[original_id] = item
                self.current_id += 1
                added_items.append(item)
                
                # Store chunks
                for i, chunk in enumerate(chunks):
                    chunk_item = NewsItem(
                        title=f"{item.title} (chunk {i+1})",
                        link=item.link,
                        published=item.published,
                        source=item.source,
                        content=item.content,
                        article_text=chunk,
                        article_title=item.article_title,
                        sentiment=item.sentiment,
                        sentiment_score=item.sentiment_score,
                        chunk_id=i,
                        is_chunk=True,
                        original_item_id=original_id
                    )
                    
                    chunk_vector = self.encode_text(f"{item.title} {chunk}")
                    chunk_item.vector = chunk_vector
                    chunk_item.vector_id = self.current_id
                    
                    self.index.add(chunk_vector.reshape(1, -1))
                    self.stored_items[self.current_id] = chunk_item
                    self.current_id += 1
                    added_items.append(chunk_item)
            else:
                # Store single item without chunking
                vector = self.encode_text(main_text)
                
                if len(vector) != self.dimension:
                    print(f"Vector dimension mismatch. Expected {self.dimension}, got {len(vector)}")
                    vector = np.zeros(self.dimension, dtype=np.float32)
                
                item.vector = vector
                item.vector_id = self.current_id
                
                self.index.add(vector.reshape(1, -1))
                self.stored_items[self.current_id] = item
                self.current_id += 1
                added_items.append(item)
                
        except Exception as e:
            print(f"Error adding item to vector database: {e}")
            added_items.append(item)
        
        return added_items
    
    def search(self, query: str, k: int = 5) -> List[NewsItem]:
        query_vector = self.encode_text(query)
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        results = []
        for idx in indices[0]:
            if idx != -1 and idx in self.stored_items:
                results.append(self.stored_items[idx])
        return results

    def save(self, filepath: str):
        faiss.write_index(self.index, f"{filepath}.index")
        items_data = {}
        for k, v in self.stored_items.items():
            item_dict = asdict(v)
            if item_dict['vector'] is not None:
                item_dict['vector'] = item_dict['vector'].tolist()
            items_data[k] = item_dict
        
        with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
            json.dump(
                {"current_id": self.current_id, "items": items_data},
                f,
                cls=NumpyJSONEncoder,
                ensure_ascii=False,
                indent=2
            )
    
    def load(self, filepath: str):
        self.index = faiss.read_index(f"{filepath}.index")
        with open(f"{filepath}.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.current_id = data["current_id"]
            self.stored_items = {
                int(k): NewsItem(**{
                    key: (np.array(val) if key == 'vector' and val is not None else val)
                    for key, val in v.items()
                })
                for k, v in data["items"].items()
            }

    def get_relevant_context(self, query: str, k: int = 5) -> Tuple[List[NewsItem], List[float]]:
        """Get relevant articles and their distances for RAG with deduplication"""
        query_vector = self.encode_text(query)
        distances, indices = self.index.search(query_vector.reshape(1, -1), k * 2)  # Get more to filter
        
        results = []
        seen_articles = set()
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1 and idx in self.stored_items:
                item = self.stored_items[idx]
                
                # Deduplicate chunks from same article
                article_key = item.original_item_id if item.is_chunk else item.vector_id
                if article_key not in seen_articles:
                    results.append(item)
                    seen_articles.add(article_key)
                    if len(results) >= k:
                        break
        
        return results, distances[0][:len(results)]


class RAGEngine:
    def __init__(self, vector_db: VectorDB):
        self.vector_db = vector_db
        self.model_name = "t5-small"
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map='auto' if torch.cuda.is_available() else None,
                torch_dtype=torch.float32
            )
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_response(self, query: str, max_length: int = 300) -> str:
        try:
            # Get relevant articles with improved context selection
            relevant_items, distances = self.vector_db.get_relevant_context(query, k=3)
            
            if not relevant_items:
                return "No relevant articles found in the database."
            
            # Build context more intelligently
            context_parts = []
            for item in relevant_items:
                # Use chunk text if it's a chunk, otherwise use full article text
                text_to_use = item.article_text[:300] if len(item.article_text) > 300 else item.article_text
                context_parts.append(f"Title: {item.title}. Content: {text_to_use}")
            
            context = " ".join(context_parts)
            
            # Improved prompt engineering
            prompt = f"Based on the following news articles, answer this question: {query}\n\nNews Context: {context}\n\nAnswer:"

            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=30,
                num_return_sequences=1,
                temperature=0.6,
                do_sample=True,
                no_repeat_ngram_size=3,
                length_penalty=1.2,
                early_stopping=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response.strip() if response.strip() else "Unable to generate a response based on the available articles."
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"


def build_google_news_rss_url(q: str, hl: str = "en-US", gl: str = "US") -> str:
    return f"https://news.google.com/rss/search?q={q.replace(' ', '+')}&hl={hl}&gl={gl}&ceid={gl}:en"


def get_real_article_url(google_news_url: str) -> Optional[str]:
    if not google_news_url or 'news.google.com' not in google_news_url:
        return google_news_url
    try:
        resp = requests.get(google_news_url, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        c_wiz = soup.select_one('c-wiz[data-p]')
        if not c_wiz:
            return None
        data = c_wiz.get('data-p')
        obj = json.loads(data.replace('%.@.', '["garturlreq",'))
        payload = {'f.req': json.dumps([[['Fbv4je', json.dumps(obj[:-6] + obj[-2:]), 'null', 'generic']]])}
        headers = {'content-type': 'application/x-www-form-urlencoded;charset=UTF-8',
                   'user-agent': 'Mozilla/5.0'}
        url = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        array_string = json.loads(response.text.replace(")]}'", ""))[0][2]
        return json.loads(array_string)[1]
    except Exception:
        return None


def extract_summary_text(summary_html: str) -> str:
    soup = BeautifulSoup(summary_html, "html.parser")
    for a_tag in soup.find_all("a"):
        a_tag.replace_with(a_tag.get_text())
    return re.sub(r'\s+', ' ', soup.get_text(separator=' ', strip=True)).strip()


def scrape_article_content(url: str, timeout: float = 15.0) -> dict:
    """Enhanced article scraping with better fallbacks"""
    result = {"article_text": "", "article_title": None, "status": "failure"}
    
    try:
        # First try newspaper3k
        article = Article(url)
        article.download()
        article.parse()
        if article.text.strip() and len(article.text.strip()) > 100:  # Minimum length check
            return {
                "article_text": article.text.strip(), 
                "article_title": article.title,
                "status": "success"
            }
    except Exception:
        pass

    try:
        # Fallback: trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted and len(extracted.strip()) > 100:
                return {
                    "article_text": extracted.strip(), 
                    "article_title": None,
                    "status": "success"
                }
    except Exception:
        pass

    try:
        # Last fallback: BeautifulSoup paragraph join
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs)
        if text.strip() and len(text.strip()) > 100:
            return {
                "article_text": text.strip(), 
                "article_title": soup.title.string if soup.title else None,
                "status": "success"
            }
    except Exception:
        pass

    return result


def fetch_google_news(rss_url: str, max_items: int = 10) -> List[NewsItem]:
    feed = feedparser.parse(rss_url)
    items = []
    for i, entry in enumerate(feed.entries[:max_items]):
        items.append(NewsItem(
            title=entry.get("title", ""),
            link=entry.get("link", ""),
            published=entry.get("published", ""),
            source=getattr(entry.source, 'title', "") if hasattr(entry, 'source') else "",
            content=entry.get("summary", "")
        ))
    return items


class SentimentAnalyzer:
    def __init__(self):  # Fixed from Code 2's bug
        try:
            self._analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon")
            self._analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> tuple:
        text = text or ""
        scores = self._analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            return "positive", compound
        elif compound <= -0.05:
            return "negative", abs(compound)
        return "neutral", abs(compound)


def process_news_item(item: NewsItem) -> NewsItem:
    """Enhanced processing with URL filtering"""
    real_url = get_real_article_url(item.link)
    
    if real_url:
        # Apply URL filtering from Code 2
        if not is_news_or_blog(real_url):
            print(f"🚫 Filtering out non-news URL: {real_url[:70]}...")
            item.article_text = ""
            return item
            
        if real_url != item.link:
            article_data = scrape_article_content(real_url)
            item.article_text = article_data["article_text"]
            item.article_title = article_data["article_title"]
            item.link = real_url
    
    return item


def analyze_sentiment(items: List[NewsItem]) -> List[NewsItem]:
    analyzer = SentimentAnalyzer()
    for item in items:
        if item.article_text:  # Only analyze items with content
            sentiment, score = analyzer.analyze(item.article_text or item.title)
            item.sentiment, item.sentiment_score = sentiment, round(score, 4)
    return items


def scrape_google_news(query: str, language: str = "en-US", country: str = "US", 
                      max_results: int = 10, workers: int = 3, 
                      vector_db: Optional[VectorDB] = None,
                      enable_chunking: bool = True) -> Tuple[List[dict], float, Dict[str, Any]]:
    """Enhanced scraping with better metrics and filtering"""
    try:
        rss_url = build_google_news_rss_url(query, hl=language, gl=country)
        news_items = fetch_google_news(rss_url, max_items=max_results)
        
        print(f"📰 Found {len(news_items)} articles from RSS feed")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            news_items = list(executor.map(process_news_item, news_items))
        
        # Filter out items with no content after URL filtering
        valid_items = [item for item in news_items if item.article_text.strip()]
        filtered_count = len(news_items) - len(valid_items)
        
        if filtered_count > 0:
            print(f"🚫 Filtered out {filtered_count} non-news or empty articles")
        
        news_items = analyze_sentiment(valid_items)
        
        # Enhanced metrics
        metrics = {
            "total_fetched": len(news_items),
            "filtered_out": filtered_count,
            "sentiment_distribution": {},
            "avg_article_length": 0,
            "chunked_articles": 0
        }
        
        if vector_db and news_items:
            all_added_items = []
            for item in news_items:
                added_items = vector_db.add_item(item, enable_chunking=enable_chunking)
                all_added_items.extend(added_items)
                if len(added_items) > 1:  # Item was chunked
                    metrics["chunked_articles"] += 1
        
        # Calculate metrics
        if news_items:
            sentiments = [item.sentiment for item in news_items if item.sentiment]
            for sentiment in ['positive', 'negative', 'neutral']:
                metrics["sentiment_distribution"][sentiment] = sentiments.count(sentiment)
            
            article_lengths = [len(item.article_text) for item in news_items if item.article_text]
            metrics["avg_article_length"] = int(np.mean(article_lengths)) if article_lengths else 0
        
        # Convert to dict format
        results = []
        success_count = 0
        
        for item in news_items:
            success = bool(item.article_text.strip())
            if success:
                success_count += 1
            
            results.append({
                "title": item.title,
                "link": item.link,
                "published": item.published,
                "source": item.source,
                "article_text": item.article_text,
                "article_title": item.article_title,
                "sentiment": item.sentiment,
                "sentiment_score": item.sentiment_score,
                "status": "success" if success else "failure",
                "article_length": len(item.article_text)
            })
        
        success_rate = (success_count / len(results) * 100) if results else 0
        return results, success_rate, metrics
        
    except Exception as e:
        print(f"Error in scrape_google_news: {e}")
        return [], 0.0, {}


def to_dataframe(news_items: List[dict]) -> pd.DataFrame:
    try:
        processed_items = []
        for item in news_items:
            processed_item = item.copy()
            if isinstance(processed_item.get('vector'), np.ndarray):
                processed_item['vector'] = processed_item['vector'].tolist()
            processed_items.append(processed_item)
        
        df = pd.DataFrame(processed_items)
        columns = [
            "title", "link", "published", "source", "article_text", 
            "article_title", "sentiment", "sentiment_score", "article_length"
        ]
        return df[columns] if not df.empty else pd.DataFrame()
    except Exception as e:
        print(f"Error in to_dataframe: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Google News Scraper with RAG - Best of Both Worlds")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--q", help="Search query for scraping news")
    group.add_argument("--search", help="Search query in existing vector database")
    
    parser.add_argument("--hl", default="en-US", help="Language")
    parser.add_argument("--gl", default="US", help="Country")
    parser.add_argument("--max", type=int, default=10, help="Max results")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers")
    parser.add_argument("--output", default="output.json", help="Output file (CSV or JSON)")
    parser.add_argument("--format", choices=["csv", "json"], default="json", help="Output format")
    parser.add_argument("--vector-db", default="news_vectors", help="Vector database file path")
    parser.add_argument("--rag-query", help="Query for RAG-based answer generation")
    parser.add_argument("--no-chunking", action="store_true", help="Disable article chunking")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()

    vector_db = VectorDB()
    enable_chunking = not args.no_chunking
    
    if args.rag_query:
        try:
            vector_db.load(args.vector_db)
            print(f"📚 Loaded vector database with {len(vector_db.stored_items)} items")
            
            if not args.q and not args.search:
                args.search = args.rag_query
            
            if args.q:
                print(f"🔍 Scraping new articles for: {args.q}")
                results, success_rate, metrics = scrape_google_news(
                    args.q, args.hl, args.gl, args.max, 
                    args.workers, vector_db=vector_db,
                    enable_chunking=enable_chunking
                )
                vector_db.save(args.vector_db)
                print(f"💾 Updated vector database")
            
            # Generate RAG response
            print(f"🤖 Generating answer for: {args.rag_query}")
            rag_engine = RAGEngine(vector_db)
            response = rag_engine.generate_response(args.rag_query)
            
            print(f"\n" + "="*80)
            print(f"❓ Question: {args.rag_query}")
            print(f"="*80)
            print(f"🤖 Answer: {response}")
            print(f"="*80)
            
        except FileNotFoundError:
            print("❌ No vector database found. Please run with --q first to scrape some articles.")
            print("💡 Example: python improved_news_rag.py --q 'artificial intelligence' --max 10")
            
    elif args.search:
        try:
            vector_db.load(args.vector_db)
            results = vector_db.search(args.search, k=5)
            print(f"\n🔍 Top similar articles for query: {args.search}")
            print("="*80)
            
            for i, item in enumerate(results, 1):
                chunk_info = f" (chunk {item.chunk_id + 1})" if item.is_chunk else ""
                print(f"\n{i}. {item.title}{chunk_info}")
                print(f"   📊 Sentiment: {item.sentiment} ({item.sentiment_score})")
                print(f"   📰 Source: {item.source}")
                print(f"   🔗 URL: {item.link}")
                if args.verbose and item.article_text:
                    preview = item.article_text[:200] + "..." if len(item.article_text) > 200 else item.article_text
                    print(f"   📝 Preview: {preview}")
                    
        except FileNotFoundError:
            print("❌ No vector database found. Please run the scraper first.")
            
    elif args.q:
        try:
            print(f"🚀 Starting enhanced news scraping for: '{args.q}'")
            print(f"🌐 Language: {args.hl}, Country: {args.gl}")
            print(f"📄 Max articles: {args.max}")
            print(f"📊 Chunking: {'Enabled' if enable_chunking else 'Disabled'}")
            
            results, success_rate, metrics = scrape_google_news(
                args.q, args.hl, args.gl, args.max, 
                args.workers, vector_db=vector_db,
                enable_chunking=enable_chunking
            )

            # Save vector database
            vector_db.save(args.vector_db)

            # Enhanced output with metrics
            output_data = {
                "metadata": {
                    "query": args.q,
                    "success_rate": round(success_rate, 2),
                    "total_articles": len(results),
                    "successful_extractions": sum(1 for r in results if r["status"] == "success"),
                    "chunking_enabled": enable_chunking,
                    "metrics": metrics
                },
                "articles": results
            }

            # Save results
            if args.format == "json" or args.output.endswith(".json"):
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
            else:
                df = to_dataframe(results)
                df.to_csv(args.output, index=False, encoding="utf-8")
            
            # Enhanced output summary
            print(f"\n" + "="*80)
            print(f"✅ SCRAPING COMPLETE")
            print(f"="*80)
            print(f"📊 Results saved to: {args.output}")
            print(f"🔍 Vector database saved to: {args.vector_db}")
            print(f"📈 Article extraction success rate: {round(success_rate, 2)}%")
            print(f"🎯 Articles processed: {len(results)}")
            
            if metrics:
                print(f"🚫 Articles filtered out: {metrics.get('filtered_out', 0)}")
                print(f"📏 Average article length: {metrics.get('avg_article_length', 0)} chars")
                print(f"🧩 Articles chunked: {metrics.get('chunked_articles', 0)}")
                
                sentiment_dist = metrics.get('sentiment_distribution', {})
                if sentiment_dist:
                    print(f"😊 Sentiment distribution:")
                    for sentiment, count in sentiment_dist.items():
                        print(f"   {sentiment.capitalize()}: {count}")
            
            print(f"\n💡 Try RAG queries with: --rag-query 'your question here'")
            print(f"="*80)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        parser.print_help()
