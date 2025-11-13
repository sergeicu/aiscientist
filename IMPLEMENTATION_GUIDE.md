# Implementation Guide - Getting Started

This guide provides step-by-step instructions to begin implementing the AI Scientist platform.

## Quick Start: Next 2 Weeks

### Week 1: Setup Infrastructure & PubMed Scraper

#### Day 1-2: Infrastructure Setup

**1. Set up Docker environment**

```bash
# Create docker-compose.yml (see ARCHITECTURE_DESIGN.md)
docker-compose up -d

# Verify services
docker-compose ps

# Expected output:
# postgres    - running on 5432
# neo4j       - running on 7474/7687
# chromadb    - running on 8000
# ollama      - running on 11434
```

**2. Initialize databases**

```bash
# PostgreSQL
psql -U researcher -d aiscientist -f schema/init.sql

# Neo4j (open browser at localhost:7474)
# Run Cypher:
CREATE CONSTRAINT author_orcid IF NOT EXISTS
FOR (a:Author) REQUIRE a.orcid IS UNIQUE;

CREATE INDEX author_name IF NOT EXISTS
FOR (a:Author) ON (a.name);
```

**3. Set up environment**

```bash
# .env
cp .env.example .env

# Add:
NCBI_API_KEY=your_key_here  # Get from https://www.ncbi.nlm.nih.gov/account/
NEO4J_PASSWORD=your_password
POSTGRES_PASSWORD=your_password
MAPBOX_TOKEN=your_token  # For later
```

#### Day 3-5: PubMed Scraper

**Create scraper module**

```bash
mkdir -p src/scrapers
touch src/scrapers/__init__.py
touch src/scrapers/pubmed_scraper.py
touch src/scrapers/rate_limiter.py
```

**1. Implement rate limiter**

```python
# src/scrapers/rate_limiter.py

import asyncio
import time
from collections import deque
from typing import Optional

class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(
        self,
        requests_per_second: float = 3.0,
        burst_size: int = 10
    ):
        self.rate = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make request."""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Add tokens based on elapsed time
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            if self.tokens < 1:
                # Wait for next token
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1
```

**2. Implement PubMed scraper**

```python
# src/scrapers/pubmed_scraper.py

from Bio import Entrez
import asyncio
import aiohttp
from typing import List, Dict, Optional
from loguru import logger
from .rate_limiter import RateLimiter

class PubMedScraper:
    """Scrape PubMed articles by affiliation."""

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None
    ):
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
            # With API key: 10 req/sec, without: 3 req/sec
            self.rate_limiter = RateLimiter(
                requests_per_second=9.0 if api_key else 2.5
            )
        else:
            self.rate_limiter = RateLimiter(requests_per_second=2.5)

    async def search_by_affiliation(
        self,
        affiliation: str,
        max_results: int = 10000,
        date_range: Optional[str] = None
    ) -> List[str]:
        """
        Search PubMed for articles from affiliation.

        Args:
            affiliation: e.g., "Boston Children's Hospital[Affiliation]"
            max_results: Maximum PMIDs to return
            date_range: e.g., "2020:2024[PDAT]" or None for all

        Returns:
            List of PMIDs
        """
        await self.rate_limiter.acquire()

        # Build query
        query = affiliation
        if date_range:
            query = f"{affiliation} AND {date_range}"

        logger.info(f"Searching PubMed: {query}")

        # Use history server for large queries
        search_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            usehistory="y"
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()

        pmids = search_results["IdList"]
        count = int(search_results["Count"])

        logger.info(f"Found {count} articles, retrieved {len(pmids)} PMIDs")

        # If more results than retmax, use history
        if count > max_results:
            webenv = search_results["WebEnv"]
            query_key = search_results["QueryKey"]
            return await self._fetch_all_pmids(
                webenv, query_key, count, max_results
            )

        return pmids

    async def _fetch_all_pmids(
        self,
        webenv: str,
        query_key: str,
        total: int,
        max_results: int
    ) -> List[str]:
        """Fetch all PMIDs using history server."""
        batch_size = 500
        pmids = []

        for start in range(0, min(total, max_results), batch_size):
            await self.rate_limiter.acquire()

            handle = Entrez.esearch(
                db="pubmed",
                retstart=start,
                retmax=batch_size,
                webenv=webenv,
                query_key=query_key
            )
            results = Entrez.read(handle)
            handle.close()

            pmids.extend(results["IdList"])
            logger.debug(f"Fetched PMIDs {start}-{start+batch_size}")

        return pmids

    async def fetch_article_details(
        self,
        pmids: List[str],
        batch_size: int = 200
    ) -> List[Dict]:
        """
        Fetch detailed metadata for PMIDs.

        Args:
            pmids: List of PubMed IDs
            batch_size: Articles per request (max 500)

        Returns:
            List of article dictionaries
        """
        articles = []

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i+batch_size]

            await self.rate_limiter.acquire()

            logger.info(
                f"Fetching articles {i+1}-{min(i+batch_size, len(pmids))} "
                f"of {len(pmids)}"
            )

            handle = Entrez.efetch(
                db="pubmed",
                id=batch,
                rettype="xml",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()

            # Parse each article
            for record in records["PubmedArticle"]:
                article = self._parse_article(record)
                articles.append(article)

        return articles

    def _parse_article(self, record: Dict) -> Dict:
        """Parse PubMed XML record to structured dict."""
        article = record["MedlineCitation"]["Article"]
        medline = record["MedlineCitation"]

        # Basic metadata
        pmid = str(medline["PMID"])
        title = article.get("ArticleTitle", "")

        # Abstract
        abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
        if isinstance(abstract_parts, list):
            abstract = " ".join(str(part) for part in abstract_parts)
        else:
            abstract = str(abstract_parts)

        # Journal info
        journal = article.get("Journal", {})
        journal_title = journal.get("Title", "")

        # Publication date
        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "")

        # Authors
        authors = []
        for author in article.get("AuthorList", []):
            author_dict = {
                "last_name": author.get("LastName", ""),
                "first_name": author.get("ForeName", ""),
                "initials": author.get("Initials", ""),
                "affiliations": []
            }

            # Affiliations
            affil_list = author.get("AffiliationInfo", [])
            for affil in affil_list:
                author_dict["affiliations"].append(
                    affil.get("Affiliation", "")
                )

            authors.append(author_dict)

        # DOI
        doi = None
        for id_item in article.get("ELocationID", []):
            if id_item.attributes.get("EIdType") == "doi":
                doi = str(id_item)

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal_title,
            "year": year,
            "doi": doi,
            "authors": authors
        }

    async def scrape_and_save(
        self,
        affiliation: str,
        output_file: str,
        max_results: int = 10000
    ):
        """
        Complete workflow: search, fetch, save.

        Args:
            affiliation: Institution to search
            output_file: Path to save JSON results
            max_results: Max articles to retrieve
        """
        import json
        from pathlib import Path

        # Search
        pmids = await self.search_by_affiliation(
            affiliation, max_results
        )

        if not pmids:
            logger.warning("No articles found")
            return

        # Fetch details
        articles = await self.fetch_article_details(pmids)

        # Save
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(articles, f, indent=2)

        logger.info(f"Saved {len(articles)} articles to {output_file}")
```

**3. Create CLI command**

```python
# Add to main.py

@cli.command()
@click.option(
    '--affiliation',
    default="Boston Children's Hospital[Affiliation]",
    help='PubMed affiliation query'
)
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    default="data/raw/pubmed_articles.json",
    help='Output JSON file'
)
@click.option(
    '--max-results',
    type=int,
    default=10000,
    help='Maximum articles to retrieve'
)
@click.option(
    '--date-range',
    type=str,
    help='Date range (e.g., "2020:2024[PDAT]")'
)
def scrape(affiliation: str, output: Path, max_results: int, date_range: str):
    """Scrape PubMed articles by affiliation."""
    from src.scrapers import PubMedScraper
    import os

    console.print(Panel.fit(
        f"[bold]PubMed Scraper[/bold]\n\n"
        f"Affiliation: {affiliation}\n"
        f"Max Results: {max_results}\n"
        f"Output: {output}",
        border_style="green"
    ))

    scraper = PubMedScraper(
        email=os.getenv("NCBI_EMAIL", "your_email@example.com"),
        api_key=os.getenv("NCBI_API_KEY")
    )

    # Run async
    asyncio.run(
        scraper.scrape_and_save(affiliation, str(output), max_results)
    )

    console.print(f"\n[green]✓ Scraping complete![/green]")
```

**4. Test the scraper**

```bash
# Set environment
export NCBI_EMAIL="your_email@example.com"
export NCBI_API_KEY="your_key"  # Optional but recommended

# Test on small dataset first
python main.py scrape \
    --affiliation "Boston Children's Hospital[Affiliation]" \
    --max-results 100 \
    --output data/raw/test_100.json

# Full dataset
python main.py scrape \
    --affiliation "Boston Children's Hospital[Affiliation]" \
    --max-results 25000 \
    --output data/raw/pubmed_full.json
```

#### Day 6-7: Database Integration

**1. Create database schema**

```sql
-- schema/init.sql

CREATE TABLE IF NOT EXISTS articles (
    pmid VARCHAR(20) PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    journal VARCHAR(255),
    publication_year INTEGER,
    doi VARCHAR(100),
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS authors (
    author_id SERIAL PRIMARY KEY,
    last_name VARCHAR(255),
    first_name VARCHAR(255),
    initials VARCHAR(10),
    orcid VARCHAR(50) UNIQUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS affiliations (
    affiliation_id SERIAL PRIMARY KEY,
    institution VARCHAR(500),
    department VARCHAR(255),
    city VARCHAR(100),
    country VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    normalized_name VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS article_authors (
    pmid VARCHAR(20) REFERENCES articles(pmid) ON DELETE CASCADE,
    author_id INTEGER REFERENCES authors(author_id) ON DELETE CASCADE,
    affiliation_id INTEGER REFERENCES affiliations(affiliation_id),
    author_position INTEGER,
    is_corresponding BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (pmid, author_id, author_position)
);

-- Indexes
CREATE INDEX idx_articles_year ON articles(publication_year);
CREATE INDEX idx_articles_journal ON articles(journal);
CREATE INDEX idx_authors_last_name ON authors(last_name);
CREATE INDEX idx_affiliations_institution ON affiliations(institution);

-- Full-text search
CREATE INDEX idx_articles_abstract_fts ON articles USING GIN(to_tsvector('english', abstract));
CREATE INDEX idx_articles_title_fts ON articles USING GIN(to_tsvector('english', title));
```

**2. Create database loader**

```python
# src/db/loader.py

import psycopg2
from psycopg2.extras import execute_batch
from typing import List, Dict
import json

class DatabaseLoader:
    """Load articles into PostgreSQL."""

    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)

    def load_articles(self, articles: List[Dict]) -> None:
        """Load articles and authors into database."""
        cursor = self.conn.cursor()

        for article in articles:
            # Insert article
            cursor.execute("""
                INSERT INTO articles (pmid, title, abstract, journal, publication_year, doi, raw_data)
                VALUES (%(pmid)s, %(title)s, %(abstract)s, %(journal)s, %(year)s, %(doi)s, %(raw)s)
                ON CONFLICT (pmid) DO UPDATE
                SET title = EXCLUDED.title,
                    abstract = EXCLUDED.abstract,
                    updated_at = NOW()
            """, {
                "pmid": article["pmid"],
                "title": article["title"],
                "abstract": article["abstract"],
                "journal": article["journal"],
                "year": int(article["year"]) if article["year"] else None,
                "doi": article["doi"],
                "raw": json.dumps(article)
            })

            # Insert authors
            for pos, author in enumerate(article.get("authors", [])):
                # Insert author
                cursor.execute("""
                    INSERT INTO authors (last_name, first_name, initials)
                    VALUES (%(last_name)s, %(first_name)s, %(initials)s)
                    ON CONFLICT (orcid) DO NOTHING
                    RETURNING author_id
                """, {
                    "last_name": author["last_name"],
                    "first_name": author["first_name"],
                    "initials": author["initials"]
                })

                result = cursor.fetchone()
                if result:
                    author_id = result[0]
                else:
                    # Get existing author
                    cursor.execute("""
                        SELECT author_id FROM authors
                        WHERE last_name = %(last_name)s
                        AND first_name = %(first_name)s
                        LIMIT 1
                    """, author)
                    author_id = cursor.fetchone()[0]

                # Insert affiliations
                affiliation_id = None
                for affil_str in author.get("affiliations", []):
                    cursor.execute("""
                        INSERT INTO affiliations (institution)
                        VALUES (%(institution)s)
                        ON CONFLICT DO NOTHING
                        RETURNING affiliation_id
                    """, {"institution": affil_str})

                    result = cursor.fetchone()
                    if result:
                        affiliation_id = result[0]

                # Link article-author
                cursor.execute("""
                    INSERT INTO article_authors (pmid, author_id, affiliation_id, author_position)
                    VALUES (%(pmid)s, %(author_id)s, %(affiliation_id)s, %(position)s)
                    ON CONFLICT DO NOTHING
                """, {
                    "pmid": article["pmid"],
                    "author_id": author_id,
                    "affiliation_id": affiliation_id,
                    "position": pos
                })

        self.conn.commit()
        cursor.close()

    def close(self):
        self.conn.close()
```

**3. CLI command to load data**

```python
# Add to main.py

@cli.command()
@click.option(
    '--input',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='JSON file from scraper'
)
def load(input: Path):
    """Load scraped articles into database."""
    import json
    from src.db import DatabaseLoader
    import os

    console.print(f"[yellow]Loading articles from {input}...[/yellow]")

    with open(input) as f:
        articles = json.load(f)

    console.print(f"Found {len(articles)} articles")

    db_url = os.getenv("POSTGRES_URL")
    loader = DatabaseLoader(db_url)

    try:
        loader.load_articles(articles)
        console.print(f"[green]✓ Loaded {len(articles)} articles[/green]")
    finally:
        loader.close()
```

---

### Week 2: Embeddings & Clustering

#### Day 8-10: Embedding Generation

**1. Install embedding model**

```bash
pip install sentence-transformers transformers torch

# Download model (will cache locally)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"
```

**2. Create embedder**

```python
# src/processing/embedder.py

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict
import numpy as np
from loguru import logger

class ArticleEmbedder:
    """Generate and store embeddings for articles."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        chroma_path: str = "./data/chroma",
        device: str = "cuda"
    ):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create collection
        self.collection = self.client.get_or_create_collection(
            name="articles",
            metadata={"description": "PubMed article embeddings"}
        )

    def embed_articles(
        self,
        articles: List[Dict],
        batch_size: int = 32
    ) -> None:
        """
        Generate embeddings and store in ChromaDB.

        Combines title + abstract for semantic richness.
        """
        documents = []
        metadatas = []
        ids = []

        for article in articles:
            # Combine title and abstract
            text = f"{article['title']} {article['abstract']}"
            documents.append(text)

            # Metadata
            metadatas.append({
                "pmid": article["pmid"],
                "year": article.get("year", ""),
                "journal": article.get("journal", "")
            })

            ids.append(article["pmid"])

        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(documents)} articles...")

        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Store in ChromaDB
        logger.info("Storing in ChromaDB...")

        # ChromaDB batch limit is 41666
        chroma_batch_size = 5000

        for i in range(0, len(documents), chroma_batch_size):
            end = min(i + chroma_batch_size, len(documents))

            self.collection.add(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end].tolist(),
                ids=ids[i:end]
            )

            logger.info(f"Stored batch {i//chroma_batch_size + 1}")

        logger.info(f"✓ Embeddings complete: {len(documents)} articles")

    def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        filter: Dict = None
    ) -> List[Dict]:
        """Semantic search across article corpus."""
        query_embedding = self.model.encode([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filter
        )

        return results
```

**3. CLI command**

```python
# Add to main.py

@cli.command()
@click.option(
    '--input',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='JSON file with articles'
)
@click.option(
    '--model',
    default="BAAI/bge-large-en-v1.5",
    help='Embedding model'
)
def embed(input: Path, model: str):
    """Generate embeddings for articles."""
    import json
    from src.processing import ArticleEmbedder

    console.print(f"[yellow]Generating embeddings...[/yellow]")

    with open(input) as f:
        articles = json.load(f)

    embedder = ArticleEmbedder(model_name=model)
    embedder.embed_articles(articles)

    console.print("[green]✓ Embeddings generated[/green]")
```

**4. Run embedding**

```bash
python main.py embed --input data/raw/pubmed_full.json
```

#### Day 11-14: Clustering

**1. Implement clustering**

```python
# src/processing/clustering.py

import umap
import hdbscan
import numpy as np
from typing import List, Dict, Tuple
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger

class ArticleClusterer:
    """Cluster articles using UMAP + HDBSCAN."""

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        min_cluster_size: int = 50
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.min_cluster_size = min_cluster_size

    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = 2
    ) -> np.ndarray:
        """
        UMAP dimension reduction.

        Args:
            embeddings: High-dimensional embeddings (N, D)
            n_components: Output dimensions (2 or 3 for viz)

        Returns:
            Reduced embeddings (N, n_components)
        """
        logger.info(f"Reducing {embeddings.shape} to {n_components}D...")

        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=n_components,
            metric='cosine',
            random_state=42
        )

        reduced = reducer.fit_transform(embeddings)

        logger.info(f"✓ Reduction complete: {reduced.shape}")
        return reduced

    def cluster_articles(
        self,
        reduced_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        HDBSCAN clustering.

        Returns:
            Cluster labels (N,)
            -1 indicates noise/outliers
        """
        logger.info("Clustering...")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=10,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        labels = clusterer.fit_predict(reduced_embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        logger.info(
            f"✓ Clustering complete: {n_clusters} clusters, "
            f"{n_noise} outliers"
        )

        return labels

    def visualize_clusters(
        self,
        reduced_embeddings: np.ndarray,
        labels: np.ndarray,
        metadata: List[Dict],
        output_file: str = "clusters.html"
    ):
        """Create interactive Plotly visualization."""
        import pandas as pd

        # Create DataFrame
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'cluster': labels,
            'title': [m.get('title', '') for m in metadata],
            'pmid': [m.get('pmid', '') for m in metadata],
            'year': [m.get('year', '') for m in metadata]
        })

        # Plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['title', 'pmid', 'year'],
            title='Article Clusters (UMAP + HDBSCAN)',
            labels={'cluster': 'Cluster ID'}
        )

        fig.update_traces(marker=dict(size=5))
        fig.write_html(output_file)

        logger.info(f"✓ Visualization saved to {output_file}")
```

**2. Full pipeline**

```python
# scripts/cluster_pipeline.py

import json
import numpy as np
from pathlib import Path
from src.processing import ArticleEmbedder, ArticleClusterer

def run_clustering_pipeline(articles_file: str):
    """Complete clustering pipeline."""

    # 1. Load articles
    with open(articles_file) as f:
        articles = json.load(f)

    # 2. Get embeddings from ChromaDB
    embedder = ArticleEmbedder()
    collection = embedder.collection

    # Get all embeddings
    results = collection.get(
        include=["embeddings", "metadatas"]
    )

    embeddings = np.array(results["embeddings"])
    metadata = results["metadatas"]

    # 3. Cluster
    clusterer = ArticleClusterer()

    reduced = clusterer.reduce_dimensions(embeddings)
    labels = clusterer.cluster_articles(reduced)

    # 4. Visualize
    clusterer.visualize_clusters(
        reduced, labels, metadata,
        output_file="output/clusters.html"
    )

    # 5. Save cluster assignments
    cluster_data = [
        {
            "pmid": m["pmid"],
            "cluster_id": int(labels[i]),
            "x": float(reduced[i, 0]),
            "y": float(reduced[i, 1])
        }
        for i, m in enumerate(metadata)
    ]

    with open("output/cluster_assignments.json", "w") as f:
        json.dump(cluster_data, f, indent=2)

    print(f"✓ Clustering complete: {len(set(labels))} clusters")

if __name__ == "__main__":
    run_clustering_pipeline("data/raw/pubmed_full.json")
```

**3. Run clustering**

```bash
python scripts/cluster_pipeline.py
```

---

## Next Steps (Week 3+)

After completing the first 2 weeks:

1. **Review clusters**: Open `output/clusters.html` and explore
2. **Label clusters**: Use LLM to generate cluster names
3. **Build agent system**: Start with investment evaluator
4. **Create dashboard**: Streamlit UI for exploration

See ARCHITECTURE_DESIGN.md for full roadmap.

---

## Troubleshooting

### Common Issues

**1. ChromaDB: "Batch size too large"**
```python
# Reduce batch size
chroma_batch_size = 5000  # Instead of 10000
```

**2. CUDA out of memory**
```python
# Use CPU or reduce batch size
embedder = ArticleEmbedder(device="cpu")
# Or
embeddings = model.encode(texts, batch_size=16)
```

**3. PubMed rate limiting**
```
# Get API key (free)
https://www.ncbi.nlm.nih.gov/account/settings/
```

---

## Testing Checklist

- [ ] Scraper retrieves articles correctly
- [ ] Database stores articles without errors
- [ ] Embeddings generate successfully
- [ ] ChromaDB stores and retrieves embeddings
- [ ] Clustering produces reasonable groups
- [ ] Visualization is interactive and informative
- [ ] All CLI commands work
- [ ] Docker services are running

---

## Resources

- **PubMed E-utilities**: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- **ChromaDB Docs**: https://docs.trychroma.com/
- **UMAP**: https://umap-learn.readthedocs.io/
- **HDBSCAN**: https://hdbscan.readthedocs.io/
- **Sentence Transformers**: https://www.sbert.net/

---

Questions? Review ARCHITECTURE_DESIGN.md for the big picture.
