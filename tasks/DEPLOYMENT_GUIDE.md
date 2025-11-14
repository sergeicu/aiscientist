# Deployment Guide: Production Deployment

This guide explains how to deploy the Research Intelligence Dashboard to production for use by hospital CMOs and administrators.

---

## Deployment Options

### Option 1: Streamlit Cloud (Recommended - Easiest)
- **Cost**: Free tier available
- **Complexity**: Very easy
- **Best for**: Small to medium datasets, quick deployment

### Option 2: Vercel + Streamlit
- **Cost**: Free tier available
- **Complexity**: Easy
- **Best for**: Professional deployment with custom domain

### Option 3: AWS/GCP Cloud (Enterprise)
- **Cost**: Pay-as-you-go
- **Complexity**: Medium
- **Best for**: Large datasets, enterprise security requirements

---

## Option 1: Deploy to Streamlit Cloud

### Prerequisites
- GitHub account
- Code committed to GitHub repository
- Streamlit Cloud account (free at share.streamlit.io)

### Step 1: Prepare Repository

```bash
# Ensure all code is in repository
git add -A
git commit -m "Prepare for deployment"
git push origin main
```

### Step 2: Create `requirements.txt`

```bash
# In repository root
cat > requirements.txt << EOF
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
networkx>=3.1
neo4j>=5.12.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
numpy>=1.24.0
scikit-learn>=1.3.0
umap-learn>=0.5.0
hdbscan>=0.8.0
requests>=2.31.0
pydantic>=2.0.0
pyyaml>=6.0
python-dotenv>=1.0.0
EOF
```

### Step 3: Create `.streamlit/config.toml`

```bash
mkdir -p .streamlit
cat > .streamlit/config.toml << EOF
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
EOF
```

### Step 4: Create `secrets.toml` Template

```bash
cat > .streamlit/secrets.toml.example << EOF
# Copy this to .streamlit/secrets.toml and fill in your values
# DO NOT commit secrets.toml to git!

[neo4j]
uri = "bolt://your-neo4j-instance.com:7687"
user = "neo4j"
password = "your-password"

[pubmed]
email = "your.email@example.com"
api_key = "optional"

[general]
data_dir = "./data"
EOF
```

### Step 5: Update `.gitignore`

```bash
cat >> .gitignore << EOF
# Secrets
.streamlit/secrets.toml
.env

# Data
data/
*.json
*.npy

# Logs
logs/
*.log

# Python
__pycache__/
*.pyc
.pytest_cache/

# Neo4j
neo4j/
EOF
```

### Step 6: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

2. **Connect GitHub:**
   - Click "New app"
   - Authorize Streamlit to access your GitHub repos

3. **Configure App:**
   ```
   Repository: your-username/aiscientist
   Branch: main
   Main file path: src/dashboard/app.py
   ```

4. **Add Secrets:**
   - Click "Advanced settings"
   - Paste content from `.streamlit/secrets.toml.example`
   - Fill in real values

5. **Deploy:**
   - Click "Deploy!"
   - Wait 2-3 minutes for deployment

6. **Access:**
   - URL: `https://your-app-name.streamlit.app`
   - Share this URL with hospital CMO

### Step 7: Configure Custom Domain (Optional)

```bash
# In Streamlit Cloud dashboard:
# Settings → Custom subdomain
# Set to: research-intelligence
# URL becomes: https://research-intelligence.streamlit.app
```

---

## Option 2: Deploy to Vercel

### Step 1: Create `vercel.json`

```json
{
  "version": 2,
  "builds": [
    {
      "src": "src/dashboard/app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "src/dashboard/app.py"
    }
  ],
  "env": {
    "STREAMLIT_SERVER_PORT": "8501",
    "STREAMLIT_SERVER_HEADLESS": "true"
  }
}
```

### Step 2: Create `api/index.py` (Vercel Entry Point)

```python
"""Vercel entry point for Streamlit."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import Streamlit app
from dashboard import app

# Vercel handler
handler = app.handler
```

### Step 3: Deploy to Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel

# Follow prompts:
# Project name: research-intelligence
# Framework: Other
# Build command: (leave empty)
# Output directory: (leave empty)
```

### Step 4: Configure Environment Variables

```bash
# Add secrets in Vercel dashboard
vercel env add NEO4J_URI
vercel env add NEO4J_PASSWORD
vercel env add PUBMED_EMAIL
```

---

## Option 3: AWS Deployment (Enterprise)

### Architecture

```
┌─────────────────────────────────────────┐
│          CloudFront (CDN)               │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│     Application Load Balancer           │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│      ECS Fargate (Streamlit)            │
│  ┌─────────────────────────────────┐   │
│  │  Container: streamlit-dashboard │   │
│  │  CPU: 2 vCPU                    │   │
│  │  Memory: 4 GB                   │   │
│  │  Auto-scaling: 1-10 tasks       │   │
│  └─────────────────────────────────┘   │
└───────────────┬─────────────────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
┌───▼────────────┐  ┌──────▼────────┐
│ RDS (Postgres) │  │ Managed Neo4j │
│ for metadata   │  │ (Neo4j Aura)  │
└────────────────┘  └───────────────┘
```

### Step 1: Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

### Step 2: Create `docker-compose.yml` (Local Testing)

```yaml
version: '3.8'

services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
    depends_on:
      - neo4j
    volumes:
      - ./data:/app/data

  neo4j:
    image: neo4j:latest
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

### Step 3: Test Locally

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:8501
```

### Step 4: Deploy to AWS ECS

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Create ECR repository
aws ecr create-repository --repository-name research-intelligence

# Build and push Docker image
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker build -t research-intelligence .

docker tag research-intelligence:latest \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/research-intelligence:latest

docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/research-intelligence:latest
```

### Step 5: Create ECS Task Definition

```json
{
  "family": "research-intelligence",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "streamlit-dashboard",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/research-intelligence:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "NEO4J_URI",
          "value": "bolt://your-neo4j-aura.databases.neo4j.io:7687"
        }
      ],
      "secrets": [
        {
          "name": "NEO4J_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:neo4j-password"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/research-intelligence",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Step 6: Create ECS Service

```bash
# Create cluster
aws ecs create-cluster --cluster-name research-intelligence-cluster

# Create service
aws ecs create-service \
  --cluster research-intelligence-cluster \
  --service-name research-intelligence-service \
  --task-definition research-intelligence \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}"
```

---

## Data Management in Production

### Strategy 1: Pre-load Data (Small Datasets)

```bash
# Include data in Docker image
COPY data/ ./data/

# Pros: Fast, simple
# Cons: Large image size, data becomes stale
```

### Strategy 2: S3 Storage (Recommended)

```python
# In dashboard/utils/data_loader.py
import boto3

class DataLoader:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.bucket = 'research-intelligence-data'

    @st.cache_data(ttl=3600)
    def load_unified_data(_self):
        # Download from S3
        _self.s3.download_file(
            _self.bucket,
            'processed/unified_dataset.json',
            '/tmp/unified_dataset.json'
        )

        with open('/tmp/unified_dataset.json') as f:
            return json.load(f)
```

### Strategy 3: Database Backend

```python
# Use RDS/Aurora for structured data
import psycopg2

class DataLoader:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv('RDS_HOST'),
            database='research',
            user=os.getenv('RDS_USER'),
            password=os.getenv('RDS_PASSWORD')
        )

    def search_papers(self, query):
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM papers WHERE title ILIKE %s",
                (f'%{query}%',)
            )
            return cur.fetchall()
```

---

## Security Considerations

### 1. Secrets Management

```python
# Use environment variables or cloud secret managers
import os
from pathlib import Path

# Local development: .env file
if Path('.env').exists():
    from dotenv import load_dotenv
    load_dotenv()

# Production: Environment variables
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
```

### 2. Authentication (Optional)

```python
# Add authentication to Streamlit
import streamlit_authenticator as stauth

# In app.py
authenticator = stauth.Authenticate(
    credentials,
    'research-dashboard',
    'auth-key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Show dashboard
    st.write('Welcome!')
elif authentication_status == False:
    st.error('Username/password is incorrect')
```

### 3. HTTPS/SSL

```bash
# Streamlit Cloud: Automatic HTTPS
# AWS: Use ALB with ACM certificate
# Custom: Use Let's Encrypt with Nginx

# Nginx config
server {
    listen 443 ssl;
    server_name research.yourcompany.com;

    ssl_certificate /etc/letsencrypt/live/research.yourcompany.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/research.yourcompany.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

---

## Monitoring & Logging

### CloudWatch Logs (AWS)

```python
# Add structured logging
import logging
import json

logger = logging.getLogger(__name__)

def log_search(query, results_count, user_id=None):
    logger.info(json.dumps({
        'event': 'search',
        'query': query,
        'results': results_count,
        'user': user_id,
        'timestamp': datetime.now().isoformat()
    }))
```

### Streamlit Analytics

```python
# Track usage with Streamlit built-in
import streamlit as st

# In config.toml
[browser]
gatherUsageStats = true

# Custom analytics
def track_event(event_name, properties=None):
    # Send to your analytics service
    pass

# In dashboard
if st.button("Search"):
    track_event('search', {'query': query})
```

---

## Cost Estimation

### Streamlit Cloud (Free Tier)
- **Cost**: $0
- **Limits**: 1 app, shared resources
- **Best for**: Proof of concept, small teams

### Streamlit Cloud (Team Plan)
- **Cost**: $250/month
- **Features**: 10 apps, private sharing, custom domains
- **Best for**: Professional deployment

### AWS (Medium Usage)
- **ECS Fargate**: ~$50/month (2 vCPU, 4GB RAM)
- **Neo4j Aura**: ~$65/month (Pro plan)
- **S3 Storage**: ~$5/month (100GB)
- **Load Balancer**: ~$20/month
- **Total**: ~$140/month

---

## Deployment Checklist

### Pre-Deployment
- [ ] All tests pass locally
- [ ] Integration test completed
- [ ] Data pipeline runs successfully
- [ ] Dashboard tested with real data
- [ ] Performance acceptable (< 3s page load)
- [ ] Mobile-responsive design verified

### Deployment
- [ ] Repository prepared and pushed
- [ ] `requirements.txt` complete and tested
- [ ] Secrets configured (not committed to git!)
- [ ] Environment variables set
- [ ] Custom domain configured (if needed)
- [ ] SSL/HTTPS enabled

### Post-Deployment
- [ ] Dashboard accessible at production URL
- [ ] All features working in production
- [ ] Search returns correct results
- [ ] Visualizations render properly
- [ ] Export features work
- [ ] Monitoring and logging active
- [ ] Backup procedures in place

### User Access
- [ ] Production URL shared with hospital CMO
- [ ] User guide provided
- [ ] Training session scheduled (if needed)
- [ ] Support contact established

---

## Quick Start: Production Deployment in 30 Minutes

For fastest deployment to Streamlit Cloud:

```bash
# 1. Commit code
git add -A
git commit -m "Production ready"
git push origin main

# 2. Create requirements.txt (use provided above)

# 3. Go to share.streamlit.io
# 4. Click "New app"
# 5. Select repo: aiscientist
# 6. Main file: src/dashboard/app.py
# 7. Click "Deploy"

# Done! URL: https://your-app.streamlit.app
```

---

**End of Deployment Guide**
